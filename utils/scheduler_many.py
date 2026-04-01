# code/DiffusionFake/utils/scheduler_many.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import math
import torch


def _to_dict(cfg: Any) -> Dict[str, Any]:
    """Supports OmegaConf / argparse Namespace / dict."""
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    # OmegaConf
    if hasattr(cfg, "items") and callable(cfg.items):
        try:
            return {k: _to_dict(v) for k, v in cfg.items()}
        except Exception:
            pass
    # argparse Namespace / custom object
    if hasattr(cfg, "__dict__"):
        return {k: _to_dict(v) for k, v in cfg.__dict__.items()}
    return {}


def _get(cfg: Dict[str, Any], key: str, default=None):
    return cfg.get(key, default)


@dataclass
class SchedulerSpec:
    name: str = "none"          # onecycle | cosine | cosine_warmup | plateau | step | none
    interval: str = "epoch"     # step or epoch
    monitor: str = "v/eer"      # only for plateau


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    trainer,                      # pl.Trainer (only used for total_steps/max_epochs)
    sched_cfg_any: Any,
    base_lr: float,
) -> Union[torch.optim.Optimizer, Dict[str, Any]]:
    """
    Returns either:
      - optimizer (no scheduler)
      - dict {"optimizer": optimizer, "lr_scheduler": {...}}  (Lightning format)
    """
    sched_cfg = _to_dict(sched_cfg_any)

    # allow old style: scheduler: "CosineAnnealingLR"
    if isinstance(sched_cfg_any, str):
        name = str(sched_cfg_any).lower()
        spec = SchedulerSpec(name=("cosine" if "cosine" in name else name),
                             interval="epoch",
                             monitor="v/eer")
    else:
        spec = SchedulerSpec(
            name=str(_get(sched_cfg, "name", "none")).lower(),
            interval=str(_get(sched_cfg, "interval", "epoch")).lower(),
            monitor=str(_get(sched_cfg, "monitor", "v/eer")),
        )

    if spec.name in ("none", "null", "", "no"):
        return optimizer

    # ---------- OneCycleLR ----------
    if spec.name in ("onecycle", "onecyclelr"):
        block = _to_dict(_get(sched_cfg, "onecycle", {}))
        pct_start = float(_get(block, "pct_start", 0.15))
        div_factor = float(_get(block, "div_factor", 10.0))
        final_div_factor = float(_get(block, "final_div_factor", 1e4))
        anneal_strategy = str(_get(block, "anneal_strategy", "cos"))

        total_steps = int(getattr(trainer, "estimated_stepping_batches", 0))
        if total_steps <= 0:
            # fallback: epochs * steps_per_epoch
            max_epochs = int(getattr(trainer, "max_epochs", 1))
            steps_per_epoch = int(getattr(trainer, "num_training_batches", 0))
            total_steps = max(1, max_epochs * max(1, steps_per_epoch))

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(base_lr),
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    # ---------- CosineAnnealingLR ----------
    if spec.name in ("cosine", "cosineannealinglr"):
        block = _to_dict(_get(sched_cfg, "cosine", {}))
        eta_min_factor = float(_get(block, "eta_min_factor", 0.05))
        eta_min = float(base_lr) * eta_min_factor
        t_max = int(getattr(trainer, "max_epochs", 1))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, t_max), eta_min=eta_min
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    # ---------- Cosine warmup (LambdaLR) ----------
    if spec.name in ("cosine_warmup", "warmup_cosine", "cosinewarmup"):
        block = _to_dict(_get(sched_cfg, "cosine_warmup", {}))
        warmup_steps = int(_get(block, "warmup_steps", 2000))
        min_lr_factor = float(_get(block, "min_lr_factor", 0.05))

        total_steps = int(getattr(trainer, "estimated_stepping_batches", 0))
        if total_steps <= 0:
            max_epochs = int(getattr(trainer, "max_epochs", 1))
            steps_per_epoch = int(getattr(trainer, "num_training_batches", 0))
            total_steps = max(1, max_epochs * max(1, steps_per_epoch))

        min_lr = float(base_lr) * min_lr_factor

        def lr_lambda(step: int):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return (min_lr / float(base_lr)) + (1.0 - (min_lr / float(base_lr))) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    # ---------- ReduceLROnPlateau ----------
    if spec.name in ("plateau", "reducelronplateau"):
        block = _to_dict(_get(sched_cfg, "plateau", {}))
        factor = float(_get(block, "factor", 0.5))
        patience = int(_get(block, "patience", 2))
        threshold = float(_get(block, "threshold", 1e-4))
        min_lr_factor = float(_get(block, "min_lr_factor", 0.05))
        min_lr = float(base_lr) * min_lr_factor

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            threshold=threshold,
            min_lr=min_lr,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": spec.monitor,
            },
        }

    # ---------- StepLR ----------
    if spec.name in ("step", "steplr"):
        block = _to_dict(_get(sched_cfg, "step", {}))
        step_size = int(_get(block, "step_size", 10))
        gamma = float(_get(block, "gamma", 0.5))

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    # fallback: no scheduler
    return optimizer