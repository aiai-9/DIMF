# code/ImageDifussionFake/train.py

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DiffusionFake training entrypoint (PyTorch Lightning)

Multi-GPU torchrun-safe version.

Features:
1) Robust dataset key normalization -> source/target/hint/txt/label
2) Robust image shape normalization -> per-sample CHW, batch BCHW
3) YAML-driven hyperparams
4) DDP-safe WandB init + code snapshot only on rank 0
5) Resume from:
   - auto       -> last.ckpt
   - auto_best  -> best_model_path from last.ckpt callback state, with fallback scan
   - explicit path
6) EarlyStopping + LearningRateMonitor from YAML
7) Works correctly with torchrun multi-GPU
8) Lightning handles distributed samplers correctly
"""

import os
import glob
import shutil
import random
import subprocess
from typing import Any, Dict, Optional

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from datasets import create_dataset
from utils.logger import Logger
from utils.init import setup
from utils.parameters import get_parameters
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from utils.checkpoint import CheckpointCfg, SaveAtSchedule


# ============================================================
# Helpers
# ============================================================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _is_rank0(args) -> bool:
    if "RANK" in os.environ:
        return int(os.environ["RANK"]) == 0
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"]) == 0
    return int(getattr(args, "local_rank", 0)) == 0


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _using_torchrun() -> bool:
    return _world_size() > 1


def to_chw_tensor(x: Any) -> Any:
    """
    Force a single image into CHW float tensor in [-1, 1] where possible.
    Supports np.ndarray or torch.Tensor.
    """
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        t = x
    else:
        return x

    if t.ndim != 3:
        return t.float()

    # HWC -> CHW if needed
    if t.shape[0] != 3 and 3 in t.shape:
        ch = list(t.shape).index(3)
        perm = [ch] + [i for i in range(3) if i != ch]
        t = t.permute(*perm).contiguous()

    t = t.float()

    mx = float(t.max()) if t.numel() > 0 else 0.0
    mn = float(t.min()) if t.numel() > 0 else 0.0

    if mx > 2.0:  # likely [0,255]
        t = (t / 127.5) - 1.0
    elif mn >= 0.0 and mx <= 1.0:
        t = (t * 2.0) - 1.0

    return t


class KeyAdapter(Dataset):
    """
    Normalize dataset samples into the dict keys the LightningModule expects:
    source / target / hint / txt / label (+ optional fields).
    Strips extra keys to avoid mixed-dataset collate mismatch.
    """

    REQUIRED_KEYS = {"source", "target", "hint", "txt", "label"}
    OPTIONAL_KEYS = {"hint_ori", "source_score", "target_score", "ori_path",
                     "video_id", "domain"}   # video_id needed for eval_generalization.py

    def __init__(self, ds: Dataset):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx) -> Dict[str, Any]:
        s = self.ds[idx]

        if isinstance(s, (list, tuple)):
            s = s[0]

        if s is None or not isinstance(s, dict):
            raise TypeError(f"Dataset must return dict, got {type(s)}")

        if "source" not in s:
            if "hint_ori" in s:
                s["source"] = s["hint_ori"]
            elif "image" in s:
                s["source"] = s["image"]
            elif "jpg" in s:
                s["source"] = s["jpg"]
            else:
                raise KeyError(f"Cannot build source; keys={list(s.keys())}")

        if "target" not in s:
            s["target"] = s["source"]
        if "hint" not in s:
            s["hint"] = s["source"]
        if "hint_ori" not in s:
            s["hint_ori"] = s["source"]
        if "txt" not in s:
            s["txt"] = ""
        if "label" not in s:
            s["label"] = 0.0
        if "ori_path" not in s:
            s["ori_path"] = ""
        if "video_id" not in s:
            s["video_id"] = ""

        s["source"] = to_chw_tensor(s["source"])
        s["target"] = to_chw_tensor(s["target"])
        s["hint"] = to_chw_tensor(s["hint"])
        s["hint_ori"] = to_chw_tensor(s["hint_ori"])

        for k in ("source", "target", "hint"):
            if not (isinstance(s[k], torch.Tensor) and s[k].ndim == 3 and s[k].shape[0] == 3):
                raise ValueError(f"{k} bad shape: {type(s[k])} {getattr(s[k], 'shape', None)}")

        if isinstance(s["label"], torch.Tensor):
            s["label"] = float(s["label"].item())
        else:
            s["label"] = float(s["label"])

        for sk in ("source_score", "target_score"):
            if sk in s:
                v = s[sk]
                s[sk] = float(v.item()) if isinstance(v, torch.Tensor) else float(v)
            else:
                s[sk] = 1.0

        s["ori_path"] = str(s.get("ori_path", ""))
        s["video_id"] = str(s.get("video_id", ""))

        allowed = self.REQUIRED_KEYS | self.OPTIONAL_KEYS
        return {k: s[k] for k in allowed if k in s}


def force_batch_bchw(batch_dict: Dict[str, Any], keys=("source", "target", "hint", "hint_ori")) -> Dict[str, Any]:
    """
    Final safety: ensure image batches are BCHW.
    """
    for k in keys:
        if k not in batch_dict:
            continue

        x = batch_dict[k]
        if not isinstance(x, torch.Tensor) or x.ndim != 4:
            continue

        if x.shape[1] == 3:
            continue

        if x.shape[-1] == 3:
            batch_dict[k] = x.permute(0, 3, 1, 2).contiguous()
            continue

        if x.shape[2] == 3:
            batch_dict[k] = x.permute(0, 2, 1, 3).contiguous()
            continue

        raise ValueError(f"{k}: unexpected batch shape {x.shape}")

    return batch_dict


def safe_collate(batch):
    """
    Collate that handles string fields like ori_path.
    """
    string_keys = set()
    for sample in batch:
        for k, v in sample.items():
            if isinstance(v, str):
                string_keys.add(k)

    string_data = {k: [s.pop(k) for s in batch] for k in string_keys}
    out = default_collate(batch)

    for k, vals in string_data.items():
        out[k] = vals

    return force_batch_bchw(out)


# ============================================================
# TrainValProgressBar — shows val progress in terminal
# ============================================================

class TrainValProgressBar(pl.callbacks.TQDMProgressBar):
    """
    Extends Lightning's TQDM bar to show validation batch progress.

    Without this, validation runs completely silently in DDP mode.
    52,500 val samples with no output = 9 minutes that looks like a freeze.
    Users kill the process, last.ckpt saves at epoch N, and training always
    restarts from the same epoch. This progress bar ends that confusion.
    """

    def on_validation_epoch_start(self, trainer, pl_module):
        super().on_validation_epoch_start(trainer, pl_module)
        if trainer.is_global_zero:
            try:
                if isinstance(trainer.val_dataloaders, list):
                    n = sum(len(dl.dataset) for dl in trainer.val_dataloaders)
                else:
                    n = len(trainer.val_dataloaders.dataset)
            except Exception:
                n = "?"
            print(
                f"\n[Val] Starting validation — {n} samples across "
                f"{trainer.world_size} ranks. Please wait...",
                flush=True,
            )

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
        if trainer.is_global_zero and batch_idx % 10 == 0:
            try:
                total = self.total_val_batches_current_dataloader
                total_str = str(total) if total is not None else "?"
            except Exception:
                total_str = "?"
            print(f"\r  [Val batch {batch_idx + 1}/{total_str}]", end="", flush=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        super().on_validation_epoch_end(trainer, pl_module)
        if trainer.is_global_zero:
            print()  # newline after \r progress


# ============================================================
# LogitSaturationMonitor — prevents eer_thr → 1.0 collapse
# ============================================================

class LogitSaturationMonitor(pl.Callback):
    """
    Two-mechanism defence against classifier logit saturation:
    1. Label smoothing decay: ε starts at label_smooth_start, decays to
       label_smooth_end over label_smooth_warmup epochs.
    2. Lambda rescue: if v/eer_thr > collapse_thr, reduces lambda_cls to
       lambda_rescue, then restores linearly over restore_epochs.
    """

    def __init__(
        self,
        collapse_thr=0.95,
        lambda_rescue=0.30,
        lambda_target=1.00,
        restore_epochs=4,
        label_smooth_start=0.10,
        label_smooth_end=0.05,
        label_smooth_warmup=5,
    ):
        super().__init__()
        self.collapse_thr        = collapse_thr
        self.lambda_rescue       = lambda_rescue
        self.lambda_target       = lambda_target
        self.restore_epochs      = restore_epochs
        self.label_smooth_start  = label_smooth_start
        self.label_smooth_end    = label_smooth_end
        self.label_smooth_warmup = label_smooth_warmup
        self._rescue_epoch       = None
        self._rescuing           = False

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch

        # Label smoothing decay
        if hasattr(pl_module, "label_smoothing"):
            t   = min(epoch / max(1, self.label_smooth_warmup), 1.0)
            eps = self.label_smooth_start + t * (
                self.label_smooth_end - self.label_smooth_start
            )
            pl_module.label_smoothing = float(eps)

        raw = trainer.callback_metrics.get("v/eer_thr")
        if raw is None:
            return
        thr = float(raw)

        if thr > 0.90 and trainer.is_global_zero:
            tag = "COLLAPSE" if thr > self.collapse_thr else "elevated"
            print(
                f"\n[LSM] epoch={epoch:03d}  eer_thr={thr:.4f} ({tag})"
                f"  lambda_cls={getattr(pl_module, 'lambda_cls', '?')}",
                flush=True,
            )

        if thr > self.collapse_thr and not self._rescuing:
            self._rescuing     = True
            self._rescue_epoch = epoch
            if hasattr(pl_module, "lambda_cls"):
                pl_module.lambda_cls = self.lambda_rescue
            if trainer.is_global_zero:
                print(f"[LSM] RESCUE: lambda_cls → {self.lambda_rescue}", flush=True)

        elif self._rescuing and self._rescue_epoch is not None:
            elapsed = epoch - self._rescue_epoch
            if elapsed >= self.restore_epochs:
                if hasattr(pl_module, "lambda_cls"):
                    pl_module.lambda_cls = self.lambda_target
                self._rescuing = False
                if trainer.is_global_zero:
                    print(f"[LSM] RESTORED lambda_cls={self.lambda_target}", flush=True)
            else:
                frac = elapsed / self.restore_epochs
                if hasattr(pl_module, "lambda_cls"):
                    pl_module.lambda_cls = float(
                        self.lambda_rescue + frac * (self.lambda_target - self.lambda_rescue)
                    )

        if trainer.is_global_zero:
            try:
                pl_module.log("alert/collapse", float(thr > self.collapse_thr),
                              on_step=False, on_epoch=True, sync_dist=False)
                pl_module.log("t/lambda_cls_eff",
                              float(getattr(pl_module, "lambda_cls", 1.0)),
                              on_step=False, on_epoch=True, sync_dist=False)
            except Exception:
                pass


def _get_lr_loggerfreq_from_yaml(args):
    lr = None
    if hasattr(args, "train"):
        if hasattr(args.train, "lr"):
            lr = args.train.lr
        elif hasattr(args.train, "learning_rate"):
            lr = args.train.learning_rate
    if lr is None:
        lr = 1e-5

    logger_freq = 300
    if hasattr(args, "train") and hasattr(args.train, "logger_freq"):
        logger_freq = int(args.train.logger_freq)

    return float(lr), int(logger_freq)


def _build_resume_ckpt(model_save_dir: str, resume_cfg_value: Any) -> Optional[str]:
    """
    Supports:
      - auto       -> last.ckpt
      - auto_best  -> best_model_path from last.ckpt callback state, else glob fallback
      - explicit path
    """
    if not isinstance(resume_cfg_value, str):
        return None

    v = resume_cfg_value.strip()
    v_lower = v.lower()

    if v_lower == "auto":
        candidate = os.path.join(model_save_dir, "last.ckpt")
        return candidate if os.path.isfile(candidate) else None

    if v_lower == "auto_best":
        last_ckpt = os.path.join(model_save_dir, "last.ckpt")
        best_path = None

        if os.path.isfile(last_ckpt):
            try:
                ckpt_obj = torch.load(last_ckpt, map_location="cpu")
                callbacks_state = ckpt_obj.get("callbacks", {})
                for _, cb_state in callbacks_state.items():
                    if isinstance(cb_state, dict) and "best_model_path" in cb_state:
                        cand = cb_state["best_model_path"]
                        if cand:
                            best_path = cand
                            break
            except Exception:
                best_path = None

        if best_path and os.path.isfile(best_path):
            return best_path

        pattern = os.path.join(model_save_dir, "best-eer-epoch=*.ckpt")
        best_ckpts = sorted(glob.glob(pattern))
        return best_ckpts[-1] if len(best_ckpts) > 0 else None

    if v not in ("", "null", "None"):
        return v

    return None


# ============================================================
# Main
# ============================================================
def main():
    args = get_parameters()
    setup(args)

    seed = int(getattr(args, "seed", 3407))
    seed_everything(seed)

    if not hasattr(args, "train") or not hasattr(args.train, "epochs"):
        raise ValueError("Missing train.epochs in YAML config.")
    max_epochs = int(args.train.epochs)

    learning_rate, logger_freq = _get_lr_loggerfreq_from_yaml(args)

    init_resume_path = (
        getattr(args, "resume_path", None)
        or getattr(args, "ckpt_path", None)
        or "./models/control_sd15_ini.ckpt"
    )

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    logger = None
    wandb_logger = None

    ckcfg = CheckpointCfg.from_args(args)
    args.exam_dir = os.path.join(ckcfg.exp_root, ckcfg.exp_name)
    os.makedirs(args.exam_dir, exist_ok=True)

    if _is_rank0(args):
        if getattr(args.wandb, "name", None) is None:
            exp_name = getattr(getattr(args, "experiment", None), "name", None)
            args.wandb.name = exp_name or os.path.basename(args.config).replace(".yaml", "")

        try:
            wandb_logger = WandbLogger(
                project=getattr(args.wandb, "project", None),
                name=getattr(args.wandb, "name", None),
                group=getattr(args.wandb, "group", None),
                job_type=getattr(args.wandb, "job_type", None),
                save_dir=args.exam_dir,
                log_model=False,
            )
            wandb_logger.experiment.config.update(
                {
                    "config_path": args.config,
                    "seed": seed,
                    "max_epochs": max_epochs,
                    "lr": learning_rate,
                },
                allow_val_change=True,
            )
            wandb_logger.experiment.save(args.config)
        except Exception:
            wandb_logger = None

        logger = Logger(name="train", log_path=f"{args.exam_dir}/train.log")
        logger.info(args)
        logger.info(f"exam_dir={args.exam_dir}")
        logger.info(
            f"max_epochs={max_epochs}, lr={learning_rate}, "
            f"logger_freq={logger_freq}, init_resume_path={init_resume_path}"
        )

        code_dir = os.path.join(args.exam_dir, "code")
        os.makedirs(code_dir, exist_ok=True)

        shutil.copytree("configs", os.path.join(code_dir, "configs"), dirs_exist_ok=True)
        shutil.copy2(__file__, os.path.join(code_dir, "train.py"))

        cldm_dir = os.path.join(code_dir, "cldm")
        os.makedirs(cldm_dir, exist_ok=True)

        if os.path.exists("cldm/diffusionfake.py"):
            shutil.copy2("cldm/diffusionfake.py", os.path.join(cldm_dir, "diffusionfake.py"))

        if os.path.exists("cldm/mamba_modules.py"):
            shutil.copy2("cldm/mamba_modules.py", os.path.join(cldm_dir, "mamba_modules.py"))

        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
            with open(os.path.join(code_dir, "git_commit.txt"), "w") as f:
                f.write(commit)
        except Exception:
            pass

    # ------------------------------------------------------------
    # Build base dataloaders from dataset factory
    # ------------------------------------------------------------
    base_train_dl = create_dataset(args, split="train")

    original_ds_name = args.dataset.name
    original_ds_configs = {}
    val_ds_cfg = getattr(args, "val_dataset", None)

    if val_ds_cfg is not None:
        val_ds_name = getattr(val_ds_cfg, "name", None)
        if val_ds_name is not None:
            args.dataset.name = val_ds_name
            val_sub = getattr(val_ds_cfg, val_ds_name, None)
            if val_sub is not None:
                original_ds_configs[val_ds_name] = getattr(args.dataset, val_ds_name, None)
                setattr(args.dataset, val_ds_name, val_sub)
            if _is_rank0(args):
                print(f"[Val] Using separate val dataset: {val_ds_name}")

    base_val_dl = create_dataset(args, split="val")

    if _is_rank0(args):
        print(f"[Train] dataset.name={original_ds_name}")
        print(f"[Val] dataset.name={args.dataset.name}")
        try:
            print(f"[Val] dataset_len={len(base_val_dl.dataset)}")
        except Exception:
            pass


    args.dataset.name = original_ds_name
    for k, v in original_ds_configs.items():
        if v is not None:
            setattr(args.dataset, k, v)

    train_ds = KeyAdapter(base_train_dl.dataset)
    val_ds = KeyAdapter(base_val_dl.dataset)

    train_bs = int(args.train.batch_size) if hasattr(args.train, "batch_size") else base_train_dl.batch_size
    val_bs = int(args.val.batch_size) if hasattr(args.val, "batch_size") else base_val_dl.batch_size

    train_workers = int(args.train.num_workers) if hasattr(args.train, "num_workers") else base_train_dl.num_workers
    val_workers = int(args.val.num_workers) if hasattr(args.val, "num_workers") else base_val_dl.num_workers

    # IMPORTANT:
    # Do NOT manually create DistributedSampler here.
    # Lightning will inject distributed samplers after process group init.
    train_dataloader = DataLoader(
        train_ds,
        batch_size=train_bs,
        shuffle=True,
        num_workers=train_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=safe_collate,
        persistent_workers=(train_workers > 0),
    )

    val_dataloader = DataLoader(
        val_ds,
        batch_size=val_bs,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=safe_collate,
        persistent_workers=(val_workers > 0),
    )

    if _is_rank0(args):
        b = next(iter(train_dataloader))
        print("YAML scheduler cfg:", getattr(args.train, "scheduler", None))
        print("FINAL TRAIN BATCH KEYS:", b.keys())
        print("source:", b["source"].shape, "target:", b["target"].shape, "hint:", b["hint"].shape)
        assert b["source"].ndim == 4 and b["source"].shape[1] == 3, f"source wrong: {b['source'].shape}"

    # ------------------------------------------------------------
    # Callbacks / checkpoint dir
    # ------------------------------------------------------------
    img_logger = ImageLogger(batch_frequency=logger_freq, save_dir=args.exam_dir)
    model_save_dir = os.path.join(args.exam_dir, "ckpt")
    os.makedirs(model_save_dir, exist_ok=True)

    resume_ckpt = None
    if hasattr(args, "train") and hasattr(args.train, "resume_ckpt"):
        resume_ckpt = _build_resume_ckpt(model_save_dir, args.train.resume_ckpt)

    if _is_rank0(args):
        print("RESUME CKPT:", resume_ckpt)
        if logger is not None:
            logger.info(f"resolved_resume_ckpt={resume_ckpt}")

    # ------------------------------------------------------------
    # Model
    # Always use diffusionfake_mixed.yaml:
    #   1. Logs v/eer correctly → ModelCheckpoint monitor fires
    #   2. NCCL-safe DDP gather with barrier()
    #   3. Backward-compatible with single-domain training
    # ------------------------------------------------------------
    model_cfg = "configs/diffusionfake_mixed.yaml"
    if not os.path.isfile(model_cfg):
        model_cfg = "configs/diffusionfake.yaml"
        if _is_rank0(args):
            print(f"[WARN] diffusionfake_mixed.yaml not found, falling back to {model_cfg}")

    model = create_model(model_cfg).cpu()
    model.args = args

    backbone = getattr(getattr(args, "model", None), "backbone", "convnextv2_base")
    model.control_model.define_feature_filter(backbone)
    if _is_rank0(args):
        print(f"[Backbone] Using: {backbone}")

    # ── Propagate ablation flags (MUST be after define_feature_filter) ────────
    # define_feature_filter() creates mamba_head — flags must be set after that.
    # When no ablation block in YAML, abl=None → all gates remain True → zero change.
    abl = getattr(args, "ablation", None)
    if abl is not None:
        model.ablation = abl
        model.control_model.ablation = abl
        if hasattr(model.control_model, "mamba_head"):
            ddp = float(getattr(abl, "dual_drop_prob", 0.15))
            model.control_model.mamba_head.dual_drop_prob = ddp
        if _is_rank0(args):
            flags = {k: getattr(abl, k, "?") for k in [
                "use_diffusion_loss", "use_dual_upsample", "use_mamba_head",
                "use_global_smc", "use_dimf", "use_igl", "use_gap_loss", "dual_drop_prob"
            ]}
            print(f"[Ablation] Flags applied: {flags}", flush=True)
    else:
        if _is_rank0(args):
            print("[Ablation] No ablation block in YAML — running full model.", flush=True)
    # ─────────────────────────────────────────────────────────────────────────

    # Tell DiffusionFakeMixed which domain this val loader belongs to
    model.val_domain_names = [getattr(getattr(args, "val_dataset", None), "name", None)
                              or getattr(args.dataset, "name", "val")]

    if resume_ckpt is None:
        state_dict = load_state_dict(init_resume_path, location="cpu")

        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        state_dict.pop("cond_stage_model.transformer.text_model.embeddings.position_ids", None)

        model_state = model.state_dict()
        filtered_state = {}
        skipped_keys = []

        for k, v in state_dict.items():
            if k in model_state and model_state[k].shape == v.shape:
                filtered_state[k] = v
            else:
                skipped_keys.append(k)

        missing, unexpected = model.load_state_dict(filtered_state, strict=False)

        if _is_rank0(args):
            print(f"[INFO] Loaded {len(filtered_state)} matching keys from checkpoint")
            print(f"[INFO] Skipped {len(skipped_keys)} keys")
            for k in skipped_keys[:20]:
                print("   SKIPPED:", k)

            print(f"[INFO] Missing keys after load: {len(missing)}")
            for k in missing[:20]:
                print("   MISSING:", k)

            print(f"[INFO] Unexpected keys after load: {len(unexpected)}")
            for k in unexpected[:20]:
                print("   UNEXPECTED:", k)

    model.learning_rate = learning_rate
    model.sd_locked = True
    model.only_mid_control = False

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=model_save_dir,
        filename=ckcfg.best_filename,
        save_top_k=ckcfg.save_top_k,
        monitor=ckcfg.monitor,
        mode=ckcfg.mode,
        save_last=ckcfg.save_last,
        auto_insert_metric_name=False,
        save_weights_only=False,
    )

    schedule_callback = SaveAtSchedule(
        dirpath=model_save_dir,
        save_every_n_epochs=ckcfg.save_every_n_epochs,
        save_epochs=ckcfg.save_epochs,
        filename_tmpl=ckcfg.every_filename,
        save_weights_only=False,
    )

    tqdm_bar = TrainValProgressBar(refresh_rate=10)

    # LogitSaturationMonitor: guards against eer_thr→1.0 collapse
    lsm_cfg = (
        getattr(getattr(args, "train", None), "logit_saturation_monitor", None)
        or getattr(args, "logit_saturation_monitor", None)
    )
    lsm_kw = {}
    if lsm_cfg:
        for k in ("collapse_thr","lambda_rescue","lambda_target","restore_epochs",
                  "label_smooth_start","label_smooth_end","label_smooth_warmup"):
            v = getattr(lsm_cfg, k, None)
            if v is not None:
                lsm_kw[k] = type(v)(v)
    lsm_callback = LogitSaturationMonitor(**lsm_kw)

    callbacks = [tqdm_bar, lsm_callback, checkpoint_callback, schedule_callback]

    if _is_rank0(args):
        callbacks.insert(1, img_logger)

    if hasattr(args, "early_stopping"):
        es_cfg = args.early_stopping
        callbacks.append(
            EarlyStopping(
                monitor=str(getattr(es_cfg, "monitor", "v/eer")),
                mode=str(getattr(es_cfg, "mode", "min")),
                patience=int(getattr(es_cfg, "patience", 4)),
                min_delta=float(getattr(es_cfg, "min_delta", 1e-3)),
                verbose=True,
            )
        )

    callbacks.append(LearningRateMonitor(logging_interval="step"))

    accum = 1
    if hasattr(args, "train") and hasattr(args.train, "accumulate_grad_batches"):
        accum = int(args.train.accumulate_grad_batches)

    # ------------------------------------------------------------
    # Trainer config
    # ------------------------------------------------------------
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    if _using_torchrun():
        # IMPORTANT:
        # Under torchrun, devices must match WORLD_SIZE
        want_devices = _world_size()
    else:
        num_visible = torch.cuda.device_count()
        want_devices = num_visible if num_visible > 0 else 0

    if want_devices < 1:
        raise RuntimeError("No GPU detected.")

    # "bf16-mixed" = AMP with bfloat16 (correct Lightning value)
    # "bf16"       = pure bf16 (no AMP, can cause instability with this model)
    if torch.cuda.is_bf16_supported():
        precision = "bf16-mixed"
    else:
        precision = "16-mixed"

    strategy = "auto"
    if _using_torchrun():
        try:
            from pytorch_lightning.strategies import DDPStrategy
            strategy = DDPStrategy(
                find_unused_parameters=True,
                gradient_as_bucket_view=True,
            )
        except Exception:
            strategy = "ddp_find_unused_parameters_true"

    trainer_kwargs = dict(
        accelerator="gpu",
        devices=want_devices,
        strategy=strategy,
        precision=precision,
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=wandb_logger,
        enable_progress_bar=True,
        log_every_n_steps=50,
        enable_checkpointing=True,
        benchmark=True,
        deterministic=False,
        num_sanity_val_steps=0,
        gradient_clip_val=float(getattr(getattr(args, "train", None), "grad_clip_val", 1.0)),
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=accum,
        enable_model_summary=False,
    )

    # Lightning version compatibility
    try:
        trainer = pl.Trainer(
            **trainer_kwargs,
            use_distributed_sampler=True,
        )
    except TypeError:
        trainer = pl.Trainer(
            **trainer_kwargs,
            replace_sampler_ddp=True,
        )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=resume_ckpt,
    )


if __name__ == "__main__":
    main()



# 1 GPU:
#   CUDA_VISIBLE_DEVICES=1 python train.py -c configs/train.yaml
#
# 4 GPUs (recommended):
#   CUDA_VISIBLE_DEVICES=1,5,6,7 torchrun --nproc_per_node=4 train.py -c configs/train.yaml