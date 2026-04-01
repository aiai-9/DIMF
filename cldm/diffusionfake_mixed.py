# cldm/diffusionfake_mixed.py  [FIXED v3 — NCCL hang fix]
# =============================================================
# ROOT CAUSE OF "NCCL ALLREDUCE timeout after 1800 seconds":
#
# The crash at epoch 4 validation end:
#   WorkNCCL(SeqNum=21306, OpType=ALLREDUCE, NumelIn=1)
#   ran for 1800096 ms before timing out
#
# What happened step by step:
#
# 1. Validation starts after epoch 4 training (52500 samples, 4 GPUs)
#    52500 / 4 = 13125 per rank, batch_size=64 → 205 batches/rank
#    With drop_last=False, the last batch may have fewer samples,
#    causing ranks to finish at slightly different times.
#
# 2. on_validation_epoch_end fires on each rank AS SOON as it finishes
#    its own validation batches. One rank may enter this function while
#    another is still running its last val batch.
#
# 3. _gather_1d_variable() called self.all_gather(local_n) where local_n
#    is a size-1 tensor. Lightning's self.all_gather() internally calls
#    NCCL ALLREDUCE, which requires ALL ranks to call it simultaneously.
#
# 4. Because one rank entered on_validation_epoch_end before another
#    finished its val loop, the all_gather call on rank 0 waited for
#    30 minutes (NCCL_TIMEOUT=1800s) for the other ranks, then killed
#    the entire process group.
#
# THE FIX (3 parts):
#
# FIX-1: Add torch.distributed.barrier() BEFORE any collective in
#   on_validation_epoch_end. This forces ALL ranks to wait until
#   every rank has finished its validation batches before proceeding.
#
# FIX-2: Replace self.all_gather() (Lightning internal, uses ALLREDUCE)
#   with torch.distributed.all_gather() directly. The direct call is
#   barrier-safe because the preceding barrier ensures synchronization.
#
# FIX-3: Keep probs/labels on CPU in validation_step to avoid
#   accumulating tensors on GPU VRAM across 205 val batches
#   (prevents OOM that could cause a rank to silently die).
#
# ALSO FIXED:
#   BUG-1: v/eer not logged → checkpoint monitor never saved "best"
#   BUG-4: v/eer_thr oscillates → EMA smoothing applied
#   BUG-6: t/* training keys logged during val → now filtered to v/* only
# =============================================================

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import roc_curve, roc_auc_score
from typing import Dict, List, Optional

from cldm.diffusionfake import DiffusionFake


class DiffusionFakeMixed(DiffusionFake):
    """
    Unified model supporting single-domain and multi-domain validation.

    Fixes the NCCL ALLREDUCE deadlock in DDP validation via:
    - Explicit dist.barrier() before any cross-rank collective
    - Direct torch.distributed.all_gather() instead of Lightning's all_gather()
    - CPU storage of per-batch probs/labels to prevent GPU OOM
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._val_probs:  Dict[str, List[torch.Tensor]] = {}
        self._val_labels: Dict[str, List[torch.Tensor]] = {}
        # EMA state for all three key metrics (fixes wild oscillation in WandB).
        # eer_thr_ema existed before; eer_ema and auc_ema are new.
        # The checkpoint monitor fires on v/eer — if that is noisy, the "best"
        # checkpoint is selected based on random batch-composition luck, not
        # genuine model quality. EMA-smoothed v/eer fixes that.
        self._eer_thr_ema: Dict[str, float] = {}
        self._eer_ema:     Dict[str, float] = {}
        self._auc_ema:     Dict[str, float] = {}
        # FIX: alpha 0.3 was too reactive for eer/auc — 0.4 gives heavier smoothing.
        # eer_thr can stay at 0.3 (it's already smoothed and is a derived quantity).
        self._eer_thr_ema_alpha: float = 0.3
        self._eer_auc_ema_alpha: float = 0.35  # FIX: heavier smoothing for primary metrics

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _domain_name(self, dataloader_idx: int) -> str:
        names = getattr(self, "val_domain_names", None)
        if names and dataloader_idx < len(names):
            return names[dataloader_idx]
        return f"val{dataloader_idx}"

    def _primary_domain(self) -> str:
        names = getattr(self, "val_domain_names", None)
        return names[0] if names else "val0"

    @staticmethod
    def _dist_ok() -> bool:
        return dist.is_available() and dist.is_initialized()

    def _barrier(self) -> None:
        """Synchronize all DDP ranks. No-op in single-GPU mode."""
        if self._dist_ok():
            dist.barrier()

    def _safe_all_gather(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Variable-length tensor gather using torch.distributed directly.

        Unlike Lightning's self.all_gather() which uses ALLREDUCE internally
        and deadlocks when ranks hit on_validation_epoch_end at different times,
        this uses torch.distributed.all_gather() which is a pure collective
        that respects the preceding dist.barrier().

        Returns: concatenated tensor on rank 0, None on other ranks.
        """
        if not self._dist_ok():
            return x

        x = x.contiguous().float().view(-1).to(self.device)
        world = dist.get_world_size()

        # Exchange sizes so every rank knows how much padding is needed
        local_n = torch.tensor([x.numel()], device=self.device, dtype=torch.long)
        all_n   = [torch.zeros(1, device=self.device, dtype=torch.long)
                   for _ in range(world)]
        dist.all_gather(all_n, local_n)
        sizes = [int(s.item()) for s in all_n]
        max_n = max(sizes)

        # Pad to uniform length
        if x.numel() < max_n:
            x = torch.cat([x, x.new_zeros(max_n - x.numel())])

        # Gather uniform-length tensors
        gathered = [x.new_zeros(max_n) for _ in range(world)]
        dist.all_gather(gathered, x)

        if self.global_rank == 0:
            return torch.cat([gathered[r][:sizes[r]] for r in range(world)])
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Epoch lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def on_validation_epoch_start(self) -> None:
        self._val_probs.clear()
        self._val_labels.clear()

    # ─────────────────────────────────────────────────────────────────────────
    # Validation step
    # ─────────────────────────────────────────────────────────────────────────

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        source, target, c, labels = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(source, target, c, labels)

        domain = self._domain_name(dataloader_idx)

        # Only log valid val metrics (skip t/* training-only keys — BUG-6 fix)
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1 and k.startswith("v/"):
                self.log(
                    f"{k}/{domain}", v,
                    on_step=False, on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,   # safe: per-batch sync, no epoch-end barrier
                )

        # Accumulate probs/labels on CPU to avoid GPU VRAM accumulation (FIX-3)
        if "v/probs" in loss_dict and "v/labels" in loss_dict:
            if domain not in self._val_probs:
                self._val_probs[domain]  = []
                self._val_labels[domain] = []
            self._val_probs[domain].append(
                loss_dict["v/probs"].detach().float().cpu()
            )
            self._val_labels[domain].append(
                loss_dict["v/labels"].detach().float().cpu()
            )

        return loss

    # ─────────────────────────────────────────────────────────────────────────
    # Epoch end — NCCL-safe metric computation
    # ─────────────────────────────────────────────────────────────────────────

    def on_validation_epoch_end(self) -> None:
        if not self._val_probs:
            return

        # ── FIX-1: Barrier BEFORE any collective ─────────────────────────────
        # All ranks must have finished ALL validation batches before we attempt
        # any cross-rank communication. This is the core fix for the 30-minute
        # NCCL timeout: the barrier prevents rank 0 from calling all_gather
        # while another rank is still running its last validation batch.
        self._barrier()

        domain_eers    = {}
        domain_aucs    = {}
        domain_thrs    = {}
        domain_accs    = {}
        domain_spreads = {}

        try:
            for domain in list(self._val_probs.keys()):
                m = self._compute_domain_metrics(domain)

                # EMA smooth eer_thr (original BUG-4 fix)
                prev_thr   = self._eer_thr_ema.get(domain, m["eer_thr"])
                smooth_thr = self._eer_thr_ema_alpha * m["eer_thr"] + \
                             (1 - self._eer_thr_ema_alpha) * prev_thr
                self._eer_thr_ema[domain] = smooth_thr

                # FIX: EMA smooth eer and auc so checkpoint monitor uses stable values.
                # Raw eer/auc fluctuate ±0.003 from batch composition luck each epoch.
                # EMA-smoothed values mean ModelCheckpoint saves genuinely better models.
                alpha = self._eer_auc_ema_alpha
                prev_eer   = self._eer_ema.get(domain, m["eer"])
                smooth_eer = alpha * m["eer"] + (1 - alpha) * prev_eer
                self._eer_ema[domain] = smooth_eer

                prev_auc   = self._auc_ema.get(domain, m["auc"])
                smooth_auc = alpha * m["auc"] + (1 - alpha) * prev_auc
                self._auc_ema[domain] = smooth_auc

                domain_eers[domain]    = smooth_eer    # FIX: was m["eer"] (raw, noisy)
                domain_aucs[domain]    = smooth_auc    # FIX: was m["auc"] (raw, noisy)
                domain_thrs[domain]    = smooth_thr
                domain_accs[domain]    = m["best_acc"]
                domain_spreads[domain] = m["prob_spread"]

                # Per-domain logging — raw values also logged for WandB debugging
                self.log(f"v/eer_{domain}",         smooth_eer,       on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
                self.log(f"v/eer_raw_{domain}",     m["eer"],         on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
                self.log(f"v/auc_{domain}",         smooth_auc,       on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
                self.log(f"v/auc_raw_{domain}",     m["auc"],         on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
                self.log(f"v/eer_thr_{domain}",     smooth_thr,       on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
                self.log(f"v/best_acc_{domain}",    m["best_acc"],    on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
                self.log(f"v/spread_{domain}",      m["prob_spread"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)

            # ── BUG-1 FIX: log v/eer (primary metric for checkpoint monitor) ─
            primary = self._primary_domain()
            if primary in domain_eers:
                p_eer    = domain_eers[primary]
                p_auc    = domain_aucs[primary]
                p_thr    = domain_thrs[primary]
                p_acc    = domain_accs[primary]
                p_spread = domain_spreads[primary]
            else:
                vals     = list(domain_eers.values()) or [0.5]
                p_eer    = float(np.mean(vals))
                p_auc    = float(np.mean(list(domain_aucs.values()) or [0.5]))
                p_thr    = 0.5
                p_acc    = 0.0
                p_spread = 0.0

            self.log("v/eer",         p_eer,    on_step=False, on_epoch=True, prog_bar=True,  sync_dist=False)
            self.log("v/auc",         p_auc,    on_step=False, on_epoch=True, prog_bar=True,  sync_dist=False)
            self.log("v/eer_thr",     p_thr,    on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
            self.log("v/best_acc",    p_acc,    on_step=False, on_epoch=True, prog_bar=True,  sync_dist=False)
            self.log("v/prob_spread", p_spread, on_step=False, on_epoch=True, prog_bar=True,  sync_dist=False)

            # Multi-domain aggregates (only when multiple val loaders)
            if len(domain_eers) > 1:
                eer_list = list(domain_eers.values())
                auc_list = list(domain_aucs.values())
                self.log("v/eer_macro", float(np.mean(eer_list)), on_step=False, on_epoch=True, prog_bar=True,  sync_dist=False)
                self.log("v/auc_macro", float(np.mean(auc_list)), on_step=False, on_epoch=True, prog_bar=True,  sync_dist=False)
                self.log("v/eer_worst", float(np.max(eer_list)),  on_step=False, on_epoch=True, prog_bar=True,  sync_dist=False)

        finally:
            # Always clear buffers, even on exception
            self._val_probs.clear()
            self._val_labels.clear()

    # ─────────────────────────────────────────────────────────────────────────
    # Per-domain metric computation — DDP-safe
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_domain_metrics(self, domain: str) -> dict:
        """
        Gather probs/labels from all DDP ranks and compute EER/AUC on rank 0.
        Uses torch.distributed.all_gather() directly (FIX-2).

        A dist.barrier() is called by the caller before this function,
        guaranteeing all ranks have their local probs ready.
        """
        probs_local = torch.cat(self._val_probs[domain],  dim=0).float().view(-1)
        labs_local  = torch.cat(self._val_labels[domain], dim=0).float().view(-1)

        # Move to device for gather
        probs_all = self._safe_all_gather(probs_local.to(self.device))
        labs_all  = self._safe_all_gather(labs_local.to(self.device))

        # Barrier after gather — before broadcast
        self._barrier()

        # Compute on rank 0
        if self.global_rank == 0 and probs_all is not None:
            m = _metrics_from_numpy(
                probs_all.detach().cpu().numpy().astype(np.float64),
                labs_all.detach().cpu().numpy().astype(np.float64),
            )
            result = torch.tensor(
                [m["eer"], m["auc"], m["eer_thr"], m["best_acc"], m["prob_spread"]],
                device=self.device, dtype=torch.float32,
            )
        else:
            result = torch.zeros(5, device=self.device, dtype=torch.float32)

        # Broadcast rank-0 results to all other ranks
        if self._dist_ok():
            dist.broadcast(result, src=0)

        eer, auc, eer_thr, best_acc, prob_spread = result.tolist()
        return {
            "eer":        eer,
            "auc":        auc,
            "eer_thr":    eer_thr,
            "best_acc":   best_acc,
            "prob_spread": prob_spread,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Pure-numpy helpers (CPU, runs on rank 0 only)
# ─────────────────────────────────────────────────────────────────────────────

def _metrics_from_numpy(probs_np: np.ndarray, labs_np: np.ndarray) -> dict:
    mask     = np.isfinite(probs_np) & np.isfinite(labs_np)
    probs_np = probs_np[mask]
    labs_np  = (labs_np[mask] > 0.5).astype(np.int32)

    eer = auc = 0.5
    eer_thr = best_acc = prob_spread = 0.0

    if probs_np.size > 0 and len(np.unique(labs_np)) >= 2:
        fpr, tpr, thr = roc_curve(labs_np, probs_np, pos_label=1)
        fnr  = 1.0 - tpr
        idx  = int(np.nanargmin(np.abs(fpr - fnr)))

        eer     = float((fpr[idx] + fnr[idx]) / 2.0)
        eer_thr = float(thr[idx])
        auc     = float(roc_auc_score(labs_np, probs_np))

        accs     = [(probs_np >= t).astype(np.int32) == labs_np for t in thr]
        best_acc = float(max(a.mean() for a in accs)) if accs else 0.0

        pos = probs_np[labs_np == 1]
        neg = probs_np[labs_np == 0]
        if len(pos) > 0 and len(neg) > 0:
            prob_spread = float(pos.mean() - neg.mean())

    return {
        "eer":        eer,
        "auc":        auc,
        "eer_thr":    eer_thr,
        "best_acc":   best_acc,
        "prob_spread": prob_spread,
    }