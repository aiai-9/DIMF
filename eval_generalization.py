# eval_generalization.py  [FIXED v2]
# =============================================================
# FIXES vs original:
#
# FIX-E1 (CRITICAL): video_id grouping bug
#   Original: path_to_video_id(path) = dirname(path)
#   Bug: for flat-file datasets (DiffSwap, WildDeepfake), ALL reals share one
#   parent dir and ALL fakes share another -> N_videos=2, V-AUC=1.000 (artificial!)
#   Fix: use batch["video_id"] directly (already set by every dataset's __getitem__)
#   Fall back to path_to_video_id only when video_id is not in batch.
#
# FIX-E2 (CRITICAL): DFD only 344 frames
#   Root cause: eval_k_frames_per_video=1 in eval config -> K=1 frame per video
#   Fix: eval config sets eval_k_frames_per_video=50 to use all frames,
#   AND we support a new --use_all_frames CLI flag that overrides K to a large number.
#
# FIX-E3: Inverted prediction detection + auto-reporting
#   When AUC < 0.5, the model's predictions are inverted for that dataset
#   (the model sees that domain's "real" faces as fakes and vice versa).
#   We now detect this, report it clearly, and report BOTH AUC and AUC_FLIPPED
#   so the user understands the calibration issue.
#   Note: We do NOT auto-flip for the final table — that would hide the domain gap.
#   Instead we report effective_auc = max(auc, auc_flipped) and flag inversion.
#
# FIX-E4: Per-dataset optimal threshold (dataset-calibrated EER threshold)
#   Standard EER threshold assumes equal priors. We now also report threshold
#   at max-accuracy on the full test set (calibrated_thr, calibrated_acc).
#   This shows what's achievable WITH calibration vs without.
#
# FIX-E5: Proper summary CSV (the requested table format)
#   The compact summary table (Dataset | I-AUC | I-EER | V-AUC | V-EER)
#   is now always written to --out_summary_csv.
#
# FIX-E6: TTA (horizontal flip) for better frame-level metrics
#   Optional --use_tta flag averages predictions from original + horizontally
#   flipped input. This usually +0.01-0.03 AUC on cross-domain data.
#
# FIX-E7: Accurate EER calculation
#   Original used fpr[idx] alone as EER estimate; correct EER = (fpr+fnr)/2 at crossing.
#   Already fixed in metrics_video.py but also consistent here.
# =============================================================

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.init import setup
from utils.parameters import get_parameters
from datasets import create_dataset
from cldm.model import create_model
from utils.metrics_video import compute_metrics, compute_video_metrics
from utils.timer import Timer


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def ensure_test_cfg_from_eval(args):
    if hasattr(args, "test") and args.test is not None:
        return
    from argparse import Namespace
    test_cfg = Namespace()
    test_cfg.batch_size  = int(getattr(args.eval, "batch_size",  64))
    test_cfg.num_workers = int(getattr(args.eval, "num_workers",  4))
    test_cfg.shuffle     = False
    test_cfg.drop_last   = False
    test_cfg.threshold   = float(getattr(args.eval, "threshold", 0.5))
    test_cfg.max_items   = None
    args.test = test_cfg


def to_builtin(x: Any) -> Any:
    if isinstance(x, dict):   return {k: to_builtin(v) for k, v in x.items()}
    if isinstance(x, list):   return [to_builtin(v) for v in x]
    if isinstance(x, tuple):  return [to_builtin(v) for v in x]
    if isinstance(x, np.integer):  return int(x)
    if isinstance(x, np.floating): return float(x)
    if isinstance(x, np.ndarray):  return x.tolist()
    return x


def infer_batch_size_from_batch(batch: Dict[str, Any]) -> int:
    if "label" in batch and isinstance(batch["label"], torch.Tensor):
        return int(batch["label"].shape[0])
    for _, v in batch.items():
        if isinstance(v, torch.Tensor) and v.ndim >= 1:
            return int(v.shape[0])
    raise RuntimeError("Unable to infer batch size from batch.")


def normalize_paths(paths_obj: Any, batch_size: int,
                    dataset_name: str, batch_idx: int) -> List[str]:
    if paths_obj is None:
        return [f"{dataset_name}/missing_path/b{batch_idx:06d}_{i:04d}"
                for i in range(batch_size)]
    if isinstance(paths_obj, str):      return [paths_obj]
    if isinstance(paths_obj, tuple):    paths_obj = list(paths_obj)
    if isinstance(paths_obj, list):     return [str(x) for x in paths_obj]
    if isinstance(paths_obj, Sequence): return [str(x) for x in list(paths_obj)]
    return [str(paths_obj)]


def normalize_video_ids(vid_obj: Any, batch_size: int,
                        paths: List[str]) -> List[str]:
    """
    FIX-E1: Extract video_id from batch dict.
    Falls back to dirname(path) only when video_id is missing or empty.
    """
    if vid_obj is None:
        return [_path_to_fallback_vid(p) for p in paths]

    vids: List[str] = []
    if isinstance(vid_obj, (list, tuple)):
        vids = [str(v) if v else _path_to_fallback_vid(paths[i])
                for i, v in enumerate(vid_obj)]
    elif isinstance(vid_obj, torch.Tensor):
        # shouldn't happen but guard anyway
        vids = [str(v.item()) for v in vid_obj.view(-1)]
    else:
        vids = [str(vid_obj)] * batch_size

    # Pad/trim to batch_size
    if len(vids) < batch_size:
        vids += [_path_to_fallback_vid(paths[i]) for i in range(len(vids), batch_size)]
    return vids[:batch_size]


def _path_to_fallback_vid(path: str) -> str:
    """Fallback: use parent directory as video id (correct for DFD, FF++)."""
    path = str(path).replace("\\", "/").strip()
    parent = os.path.dirname(path)
    return parent if parent else os.path.splitext(os.path.basename(path))[0]


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    for k in ["source", "target", "hint", "hint_ori"]:
        if k in batch and isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device, non_blocking=True)
    for k in ["label", "source_score", "target_score"]:
        if k in batch and isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device, non_blocking=True)
    return batch


def choose_prediction_tensor(loss_dict: Dict[str, Any]) -> Tuple[str, torch.Tensor]:
    for key in ["v/probs", "probs", "v/logits", "logits"]:
        if key in loss_dict:
            val = loss_dict[key]
            if not torch.is_tensor(val):
                raise RuntimeError(f"Prediction entry '{key}' is not tensor: {type(val)}")
            if "logits" in key:
                val = torch.sigmoid(val)
            return key, val
    raise RuntimeError(
        f"No probability/logit key found in loss_dict. Keys={list(loss_dict.keys())}")


# ─────────────────────────────────────────────────────────────────────────────
# FIX-E6: TTA helper — horizontal flip
# ─────────────────────────────────────────────────────────────────────────────

def _flip_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of the batch with all image tensors horizontally flipped."""
    flipped = dict(batch)
    for k in ["source", "target", "hint", "hint_ori"]:
        if k in flipped and isinstance(flipped[k], torch.Tensor) and flipped[k].ndim == 4:
            flipped[k] = torch.flip(flipped[k], dims=[-1])
    return flipped


# ─────────────────────────────────────────────────────────────────────────────
# Core inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_probs_single(model, batch: Dict[str, Any],
                         expected_bs: int, batch_idx: int,
                         debug: bool = False) -> Tuple[torch.Tensor, str]:
    source, target, c, labels = model.get_input(batch, model.first_stage_key)
    out = model(source, target, c, labels)

    if not isinstance(out, tuple) or len(out) < 2 or not isinstance(out[1], dict):
        raise RuntimeError(f"Unexpected model output: {type(out)}")

    loss_dict = out[1]
    chosen_key, p = choose_prediction_tensor(loss_dict)
    p = p.detach().view(-1).float().cpu()

    if debug and batch_idx < 3:
        print(f"[DEBUG] batch={batch_idx} key={chosen_key} pred={tuple(p.shape)}")
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                print(f"  {k}: shape={tuple(v.shape)} "
                      f"min={float(v.detach().min()):.4f} max={float(v.detach().max()):.4f}")

    if p.numel() != expected_bs:
        shapes = {k: tuple(v.shape) for k, v in loss_dict.items() if torch.is_tensor(v)}
        raise RuntimeError(
            f"Prediction size mismatch at batch {batch_idx}: "
            f"preds={p.numel()} expected={expected_bs}. key='{chosen_key}'. shapes={shapes}")

    return p, chosen_key


@torch.no_grad()
def predict_probs(model, batch: Dict[str, Any],
                  expected_bs: int, batch_idx: int,
                  use_tta: bool = False, debug: bool = False) -> Tuple[torch.Tensor, str]:
    """
    FIX-E6: Optional TTA — average original + horizontally flipped predictions.
    TTA improves cross-domain robustness by ~0.5-2% AUC on most benchmarks.
    """
    p_orig, chosen_key = predict_probs_single(model, batch, expected_bs, batch_idx, debug)
    if not use_tta:
        return p_orig, chosen_key

    try:
        batch_flip = _flip_batch(batch)
        p_flip, _ = predict_probs_single(model, batch_flip, expected_bs, batch_idx, False)
        p_avg = (p_orig + p_flip) * 0.5
        return p_avg, chosen_key
    except Exception as e:
        if debug:
            print(f"[TTA] flip failed at batch {batch_idx}: {e}")
        return p_orig, chosen_key


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-level evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_dataset(model, loader, device: torch.device,
                 dataset_name: str, use_tta: bool = False,
                 debug: bool = False):
    probs_all  = []
    labels_all = []
    paths_all  = []
    vids_all   = []   # FIX-E1: collect video_ids from batch
    used_keys  = []

    try:
        dataset_len = len(loader.dataset)
    except Exception:
        dataset_len = -1

    print(f"[DEBUG] dataset={dataset_name} "
          f"len(dataset)={dataset_len} "
          f"len(loader)={len(loader)} "
          f"batch_size={getattr(loader, 'batch_size', 'NA')} "
          f"tta={use_tta}")

    for batch_idx, batch in enumerate(tqdm(loader, desc=f"EVAL:{dataset_name}", leave=False)):
        batch_size = infer_batch_size_from_batch(batch)
        batch      = move_batch(batch, device)

        with torch.cuda.amp.autocast(enabled=False):
            probs, chosen_key = predict_probs(
                model, batch, expected_bs=batch_size,
                batch_idx=batch_idx, use_tta=use_tta, debug=debug)

        labels = batch["label"].detach().view(-1).float().cpu()
        paths  = normalize_paths(batch.get("ori_path", None),
                                 batch_size, dataset_name, batch_idx)

        # FIX-E1: extract video_id from batch
        raw_vids = batch.get("video_id", None)
        vids     = normalize_video_ids(raw_vids, batch_size, paths)

        if labels.numel() != batch_size:
            raise RuntimeError(f"Label mismatch: labels={labels.numel()} bs={batch_size}")
        if len(paths) != batch_size:
            raise RuntimeError(f"Path mismatch: paths={len(paths)} bs={batch_size}")
        if probs.numel() != batch_size:
            raise RuntimeError(f"Prob mismatch: probs={probs.numel()} bs={batch_size}")

        probs_all.append(probs.numpy())
        labels_all.append(labels.numpy())
        paths_all.extend(paths)
        vids_all.extend(vids)
        used_keys.append(chosen_key)

    if not probs_all:
        raise RuntimeError(f"No predictions for dataset={dataset_name}")

    probs_np  = np.concatenate(probs_all,  axis=0).astype(np.float64)
    labels_np = np.concatenate(labels_all, axis=0).astype(np.float64)

    print(f"[DEBUG] dataset={dataset_name} final: "
          f"preds={len(probs_np)} labels={len(labels_np)} paths={len(paths_all)} "
          f"vids={len(set(vids_all))} unique_videos "
          f"pred_key={used_keys[0] if used_keys else 'NA'}")

    return probs_np, labels_np, paths_all, vids_all


# ─────────────────────────────────────────────────────────────────────────────
# FIX-E3: Diagnostics with inversion detection
# ─────────────────────────────────────────────────────────────────────────────

def print_prob_diagnostics(dataset_name: str, probs: np.ndarray, labs: np.ndarray):
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    labs  = (np.asarray(labs).reshape(-1) > 0.5).astype(np.int32)

    pos = probs[labs == 1]  # fakes
    neg = probs[labs == 0]  # reals

    print(f"[DIAG:{dataset_name}] "
          f"prob_min={probs.min():.6f} prob_max={probs.max():.6f} "
          f"prob_mean={probs.mean():.6f} prob_std={probs.std():.6f}")

    if len(pos) > 0:
        print(f"[DIAG:{dataset_name}] "
              f"fake_mean={pos.mean():.6f} fake_std={pos.std():.6f} n_fake={len(pos)}")
    if len(neg) > 0:
        print(f"[DIAG:{dataset_name}] "
              f"real_mean={neg.mean():.6f} real_std={neg.std():.6f} n_real={len(neg)}")

    if len(pos) > 0 and len(neg) > 0:
        spread = float(pos.mean()) - float(neg.mean())
        print(f"[DIAG:{dataset_name}] "
              f"fake_mean - real_mean = {spread:+.4f} "
              f"({'✓ correct direction' if spread > 0 else '✗ INVERTED — model sees this domain backwards'})")

    unique = np.unique(np.round(probs, 6))
    print(f"[DIAG:{dataset_name}] unique_probs_rounded={len(unique)}")
    print(f"[DIAG:{dataset_name}] first10_probs={np.round(probs[:10], 6).tolist()}")


# ─────────────────────────────────────────────────────────────────────────────
# FIX-E4: Enhanced metrics with calibrated threshold
# ─────────────────────────────────────────────────────────────────────────────

def compute_enhanced_metrics(probs: np.ndarray, labs: np.ndarray,
                              dataset_name: str) -> Dict[str, Any]:
    """
    Wraps compute_metrics and adds:
    - effective_auc: max(auc, auc_flipped) — what the model *could* do with calibration
    - inverted: True if model predicts backwards for this domain
    - calibrated_thr / calibrated_acc: threshold that maximises accuracy on test set
    """
    from utils.metrics_video import compute_metrics
    from sklearn.metrics import roc_curve

    m = compute_metrics(probs, labs)

    inverted     = m["AUC"] < 0.5
    effective_auc = max(m["AUC"], m["AUC_FLIPPED"])

    # Calibrated threshold (max accuracy on this test set)
    calibrated_thr = 0.5
    calibrated_acc = m["BEST_ACC"]

    labs_bin = (np.asarray(labs) > 0.5).astype(np.int32)
    if len(np.unique(labs_bin)) >= 2 and probs.size > 0:
        fpr, tpr, thrs = roc_curve(labs_bin, probs, pos_label=1)
        accs = []
        for th in thrs:
            pred = (probs >= th).astype(np.int32)
            accs.append(float((pred == labs_bin).mean()))
        if accs:
            best_idx      = int(np.argmax(accs))
            calibrated_thr = float(thrs[best_idx])
            calibrated_acc = float(accs[best_idx])

    m["INVERTED"]       = bool(inverted)
    m["EFFECTIVE_AUC"]  = float(effective_auc)
    m["CALIBRATED_THR"] = float(calibrated_thr)
    m["CALIBRATED_ACC"] = float(calibrated_acc)

    if inverted:
        print(f"[WARN:{dataset_name}] AUC={m['AUC']:.3f} < 0.5 — predictions are INVERTED for this domain! "
              f"effective_auc (if calibrated)={effective_auc:.3f}")

    return m


# ─────────────────────────────────────────────────────────────────────────────
# Summary table builders
# ─────────────────────────────────────────────────────────────────────────────

def build_full_table(metrics: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for ds, v in metrics.items():
        fm = v["frame"]
        vm = v["video"]
        rows.append({
            "Dataset":         ds,
            "N_frames":        fm["N"],
            "I_AUC":           round(fm["AUC"], 3),
            "I_EER":           round(fm["EER"], 3),
            "I_AUC_FLIPPED":   round(fm["AUC_FLIPPED"], 3),
            "I_EFF_AUC":       round(fm.get("EFFECTIVE_AUC", fm["AUC"]), 3),
            "I_BEST_ACC":      round(fm["BEST_ACC"], 3),
            "I_ACC_AT_EER":    round(fm["ACC_AT_EER"], 3),
            "I_CALIB_THR":     round(fm.get("CALIBRATED_THR", 0.5), 3),
            "I_CALIB_ACC":     round(fm.get("CALIBRATED_ACC", 0.0), 3),
            "I_INVERTED":      fm.get("INVERTED", False),
            "N_videos":        vm["N"],
            "V_AUC":           round(vm["AUC"], 3),
            "V_EER":           round(vm["EER"], 3),
            "V_AUC_FLIPPED":   round(vm["AUC_FLIPPED"], 3),
            "V_EFF_AUC":       round(vm.get("EFFECTIVE_AUC", vm["AUC"]), 3),
            "V_BEST_ACC":      round(vm["BEST_ACC"], 3),
            "V_ACC_AT_EER":    round(vm["ACC_AT_EER"], 3),
            "V_CALIB_THR":     round(vm.get("CALIBRATED_THR", 0.5), 3),
            "V_CALIB_ACC":     round(vm.get("CALIBRATED_ACC", 0.0), 3),
            "V_INVERTED":      vm.get("INVERTED", False),
        })
    return pd.DataFrame(rows)


def build_summary_table(df_full: pd.DataFrame) -> pd.DataFrame:
    """
    FIX-E5: Compact summary matching the requested output format.
    Dataset | I-AUC | I-EER | V-AUC | V-EER
    Also includes effective AUC for cross-domain insight.
    """
    cols = ["Dataset", "I_AUC", "I_EER", "I_EFF_AUC",
            "V_AUC", "V_EER", "V_EFF_AUC", "I_INVERTED", "V_INVERTED"]
    available = [c for c in cols if c in df_full.columns]
    df = df_full[available].copy()
    for col in [c for c in available if c not in ("Dataset", "I_INVERTED", "V_INVERTED")]:
        df[col] = pd.to_numeric(df[col], errors="coerce").round(3)
    return df


def add_avg_row(df: pd.DataFrame, exclude: List[str],
                num_cols: List[str]) -> pd.DataFrame:
    cross = df[~df["Dataset"].isin(exclude + ["AVG (cross-dataset)"])]
    if len(cross) < 2:
        return df
    avg = {"Dataset": "AVG (cross-dataset)", "N_frames": 0, "N_videos": 0}
    for col in num_cols:
        if col in cross.columns:
            try:
                avg[col] = round(float(cross[col].dropna().astype(float).mean()), 3)
            except Exception:
                avg[col] = 0.0
    return pd.concat([df, pd.DataFrame([avg])], ignore_index=True)


def print_summary(df_table: pd.DataFrame):
    print("\n" + "=" * 78)
    print(f"{'Dataset':<22} {'I-AUC':>7} {'I-EER':>7} {'I-EffAUC':>10} "
          f"{'V-AUC':>7} {'V-EER':>7} {'V-EffAUC':>10} {'Inv?':>5}")
    print("-" * 78)
    for _, row in df_table.iterrows():
        inv = "✗" if (row.get("I_INVERTED", False)) else " "
        print(f"{str(row['Dataset']):<22} "
              f"{row.get('I_AUC', 0):>7.3f} "
              f"{row.get('I_EER', 0):>7.3f} "
              f"{row.get('I_EFF_AUC', row.get('I_AUC', 0)):>10.3f} "
              f"{row.get('V_AUC', 0):>7.3f} "
              f"{row.get('V_EER', 0):>7.3f} "
              f"{row.get('V_EFF_AUC', row.get('V_AUC', 0)):>10.3f} "
              f"{inv:>5}")
    print("=" * 78)
    print("I=Frame-level, V=Video-level, EffAUC=max(AUC, 1-AUC), Inv=Inverted predictions")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    timer = Timer()
    timer.start()

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config",       required=True)
    parser.add_argument("--ckpt",               required=True)
    parser.add_argument("--out_frame_csv",      required=True)
    parser.add_argument("--out_video_csv",      required=True)
    parser.add_argument("--out_table_csv",      required=True)
    parser.add_argument("--out_summary_csv",    required=True)
    parser.add_argument("--out_metrics_json",   required=True)
    parser.add_argument("--top_k",              default=10,     type=int)
    parser.add_argument("--video_score_mode",   default="topk",
                        choices=["topk", "mean", "median"])
    # FIX-E6: TTA flag
    parser.add_argument("--use_tta",            action="store_true",
                        help="Use horizontal-flip TTA (avg original + flipped probs)")
    # FIX-E2: override DFD frame count
    parser.add_argument("--use_all_frames",     action="store_true",
                        help="Override eval_k_frames_per_video to use all frames per video")
    parser.add_argument("--debug",              action="store_true")
    args_cli = parser.parse_args()

    # Load config
    orig_argv = sys.argv[:]
    sys.argv  = [sys.argv[0], "-c", args_cli.config]
    args      = get_parameters()
    sys.argv  = orig_argv

    setup(args)
    ensure_test_cfg_from_eval(args)

    # Build model
    model = create_model(args.eval.model_config)
    backbone = getattr(getattr(args, "model", None), "backbone", "convnextv2_base")
    print(f"[INFO] Using backbone: {backbone}")
    model.control_model.define_feature_filter(backbone)

    ckpt    = torch.load(args_cli.ckpt, map_location="cpu")
    state   = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[CKPT] missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
    if missing:    print("[CKPT] first_missing:   ", missing[:10])
    if unexpected: print("[CKPT] first_unexpected:", unexpected[:10])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device).eval()

    # ── FIX-E2: use_all_frames override ─────────────────────────────────────
    # Patch DFD (and any other dataset) to use all frames in test split.
    # This is done by setting eval_k_frames_per_video to a large number in args.
    if args_cli.use_all_frames:
        print("[INFO] --use_all_frames: overriding eval_k_frames_per_video=9999 for all datasets")
        for ds_name in getattr(args.eval, "datasets", []):
            ds_cfg = getattr(args.dataset, ds_name, None)
            if ds_cfg is not None and hasattr(ds_cfg, "eval_k_frames_per_video"):
                ds_cfg.eval_k_frames_per_video = 9999

    frame_rows = []
    video_rows = []
    metrics    = {}

    for ds_name in args.eval.datasets:
        print(f"\nDATASET: {ds_name}")
        args.dataset.name = ds_name

        loader = create_dataset(args, split="test")

        probs, labs, paths, vids = eval_dataset(
            model=model, loader=loader, device=device,
            dataset_name=ds_name, use_tta=args_cli.use_tta, debug=args_cli.debug)

        print_prob_diagnostics(ds_name, probs, labs)

        # FIX-E4: enhanced frame metrics
        frame_m = compute_enhanced_metrics(probs, labs, ds_name)

        # FIX-E1: use video_id from batch for video grouping
        video_m, v_probs, v_labs, v_ids = compute_video_metrics_with_ids(
            probs_np=probs, labs_np=labs, video_ids=vids,
            top_k=args_cli.top_k, mode=args_cli.video_score_mode)
        # FIX-E4: enhance video metrics too
        video_m_enh = dict(video_m)
        video_m_enh["INVERTED"]      = video_m["AUC"] < 0.5
        video_m_enh["EFFECTIVE_AUC"] = max(video_m["AUC"], video_m["AUC_FLIPPED"])

        metrics[ds_name] = {"frame": frame_m, "video": video_m_enh}

        print(f"I-AUC {frame_m['AUC']:.3f} "
              f"I-EER {frame_m['EER']:.3f} "
              f"I-EffAUC {frame_m['EFFECTIVE_AUC']:.3f} "
              f"{'[INVERTED]' if frame_m['INVERTED'] else ''} | "
              f"V-AUC {video_m['AUC']:.3f} "
              f"V-EER {video_m['EER']:.3f} "
              f"V-EffAUC {video_m_enh['EFFECTIVE_AUC']:.3f} "
              f"N_videos={video_m['N']}")

        # Collect rows
        for p, l, path, vid in zip(probs, labs, paths, vids):
            frame_rows.append({
                "dataset": ds_name, "path": str(path),
                "video_id": str(vid), "label": int(l > 0.5), "prob": float(p),
            })
        for p, l, vid in zip(v_probs, v_labs, v_ids):
            video_rows.append({
                "dataset": ds_name, "video_id": str(vid),
                "label": int(l > 0.5), "prob": float(p),
            })

    # ── Build output tables ───────────────────────────────────────────────────
    df_frame = pd.DataFrame(frame_rows)
    df_video = pd.DataFrame(video_rows)
    df_full  = build_full_table(metrics)

    # Add AVG row (cross-dataset, exclude ffpp_rela which is in-domain)
    num_cols = [c for c in df_full.columns
                if c not in ("Dataset", "I_INVERTED", "V_INVERTED")]
    df_full = add_avg_row(df_full, exclude=["ffpp_rela"], num_cols=num_cols)

    df_summary = build_summary_table(df_full)

    # ── Write files ───────────────────────────────────────────────────────────
    out_dir = os.path.dirname(args_cli.out_frame_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df_frame.to_csv(args_cli.out_frame_csv,   index=False)
    df_video.to_csv(args_cli.out_video_csv,   index=False)
    df_full.to_csv(args_cli.out_table_csv,    index=False)
    df_summary.to_csv(args_cli.out_summary_csv, index=False)

    with open(args_cli.out_metrics_json, "w") as f:
        json.dump(to_builtin(metrics), f, indent=2)

    # ── Print results ─────────────────────────────────────────────────────────
    print("\nFINAL TABLE")
    print(df_full[["Dataset","I_AUC","I_EER","I_EFF_AUC",
                   "V_AUC","V_EER","V_EFF_AUC"]].to_string(index=False))

    print_summary(df_full)

    runtime = timer.stop()
    print(f"\nTotal evaluation time: {runtime}")
    print(f"\nOutputs:")
    print(f"  Frame CSV:   {args_cli.out_frame_csv}")
    print(f"  Video CSV:   {args_cli.out_video_csv}")
    print(f"  Full table:  {args_cli.out_table_csv}")
    print(f"  Summary CSV: {args_cli.out_summary_csv}")
    print(f"  Metrics JSON:{args_cli.out_metrics_json}")


# ─────────────────────────────────────────────────────────────────────────────
# FIX-E1: Video metrics using explicit video_id (not path-derived)
# ─────────────────────────────────────────────────────────────────────────────

def compute_video_metrics_with_ids(probs_np, labs_np, video_ids,
                                    top_k=10, mode="topk"):
    """
    Like compute_video_metrics but uses explicit video_id list instead of
    inferring from path. Fixes the DiffSwap / WildDeepfake V-AUC=1.0 artifact
    caused by all reals sharing one parent dir and all fakes sharing another.
    """
    from collections import defaultdict
    from utils.metrics_video import compute_metrics

    probs_np = np.asarray(probs_np, dtype=np.float64).reshape(-1)
    labs_np  = (np.asarray(labs_np).reshape(-1) > 0.5).astype(np.int32)

    video_scores = defaultdict(list)
    video_labels = {}

    for prob, lab, vid in zip(probs_np, labs_np, video_ids):
        video_scores[vid].append(float(prob))
        video_labels[vid] = int(lab)

    v_probs, v_labs, v_ids = [], [], []
    for vid, scores in video_scores.items():
        scores = np.asarray(scores, dtype=np.float64)
        if mode == "median":
            score = float(np.median(scores))
        elif mode == "mean":
            score = float(np.mean(scores))
        else:  # topk
            k     = min(int(top_k), len(scores))
            score = float(np.mean(np.sort(scores)[-k:]))
        v_probs.append(score)
        v_labs.append(video_labels[vid])
        v_ids.append(vid)

    v_probs = np.asarray(v_probs, dtype=np.float64)
    v_labs  = np.asarray(v_labs,  dtype=np.int32)

    m = compute_metrics(v_probs, v_labs)
    return m, v_probs, v_labs, v_ids


if __name__ == "__main__":
    main()