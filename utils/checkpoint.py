# code/DiffusionFake/utils/checkpoint.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Optional

import pytorch_lightning as pl

from typing import List
from collections.abc import Iterable as _Iterable  # ✅ NEW

def _parse_int_list(x) -> List[int]:
    """
    Accept:
      - list/tuple/OmegaConf ListConfig: [2,3,10]
      - string: "2,3,10" or "2 3 10"
      - None/empty -> []
      - single int -> [int]
    """
    if x is None:
        return []

    # strings
    if isinstance(x, str):
        s = x.replace(",", " ").strip()
        if not s:
            return []
        return sorted({int(v) for v in s.split()})

    # any iterable container (covers list, tuple, OmegaConf ListConfig)
    if isinstance(x, _Iterable):
        vals = []
        for v in x:
            if v is None or (isinstance(v, str) and v.strip() == ""):
                continue
            vals.append(int(v))
        return sorted(set(vals))

    # single scalar
    return [int(x)]


@dataclass
class CheckpointCfg:
    # experiment folder
    exp_root: str = "experiments"
    exp_name: str = "experiment"

    # best checkpoint settings
    monitor: str = "t/acc_step"
    mode: str = "max"  # "min" or "max"
    save_top_k: int = 2
    save_last: bool = True

    # schedule saving
    save_every_n_epochs: int = 0        # 0 disables interval saving
    save_epochs: List[int] = None       # explicit epochs (1-indexed)

    # filename templates (without .ckpt)
    best_filename: str = "best-epoch={epoch:02d}"
    every_filename: str = "epoch={epoch:02d}"

    @classmethod
    def from_args(cls, args) -> "CheckpointCfg":
        exp_root = getattr(getattr(args, "experiment", None), "root_dir", "experiments")
        exp_name = (
            getattr(getattr(args, "experiment", None), "name", None)
            or getattr(getattr(args, "wandb", None), "name", None)
            or os.path.basename(getattr(args, "config", "experiment")).replace(".yaml", "")
        )

        ck = getattr(args, "checkpoint", None)

        monitor = getattr(ck, "monitor", "t/loss_epoch") if ck is not None else "t/loss_epoch"
        mode = getattr(ck, "mode", "min") if ck is not None else "min"
        save_top_k = int(getattr(ck, "save_top_k", 2)) if ck is not None else 2
        save_last = bool(getattr(ck, "save_last", True)) if ck is not None else True

        save_every_n_epochs = int(getattr(ck, "save_every_n_epochs", 0) or 0) if ck is not None else 0
        if save_every_n_epochs < 0:
            raise ValueError("checkpoint.save_every_n_epochs must be >= 0")

        save_epochs = []
        if ck is not None and hasattr(ck, "save_epochs"):
            save_epochs = _parse_int_list(getattr(ck, "save_epochs"))

        best_filename = getattr(ck, "best_filename", "best-epoch={epoch:02d}") if ck is not None else "best-epoch={epoch:02d}"
        every_filename = getattr(ck, "every_filename", "epoch={epoch:02d}") if ck is not None else "epoch={epoch:02d}"

        return cls(
            exp_root=exp_root,
            exp_name=exp_name,
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            save_last=save_last,
            save_every_n_epochs=save_every_n_epochs,
            save_epochs=save_epochs,
            best_filename=best_filename,
            every_filename=every_filename,
        )


# class SaveAtSchedule(pl.Callback):
#     """
#     Save checkpoints:
#       - every N epochs: save_every_n_epochs=5 -> epochs 5,10,15,...
#       - plus explicit epochs: save_epochs=[2,3,10]
#     Epochs are 1-indexed (human-readable).
#     """

#     def __init__(
#         self,
#         dirpath: str,
#         save_every_n_epochs: int = 0,
#         save_epochs: Optional[Iterable[int]] = None,
#         filename_tmpl: str = "epoch={epoch:02d}",
#     ):
#         super().__init__()
#         self.dirpath = dirpath
#         self.save_every_n_epochs = int(save_every_n_epochs or 0)
#         self.save_epochs = sorted({int(e) for e in (save_epochs or [])})
#         self.filename_tmpl = filename_tmpl

#         if self.save_every_n_epochs < 0:
#             raise ValueError("save_every_n_epochs must be >= 0")

#     def _should_save(self, epoch_1idx: int) -> bool:
#         if epoch_1idx in self.save_epochs:
#             return True
#         if self.save_every_n_epochs > 0 and epoch_1idx % self.save_every_n_epochs == 0:
#             return True
#         return False

#     def on_train_epoch_end(self, trainer, pl_module):
#         epoch_1idx = int(trainer.current_epoch) + 1
#         if not self._should_save(epoch_1idx):
#             return

#         os.makedirs(self.dirpath, exist_ok=True)
#         name = self.filename_tmpl.format(epoch=epoch_1idx) + ".ckpt"
#         out = os.path.join(self.dirpath, name)

#         if trainer.is_global_zero:
#             trainer.save_checkpoint(out)

#         # ✅ sync all ranks so nobody runs ahead / deadlocks later
#         if trainer.world_size > 1:
#             trainer.strategy.barrier()

#         os.makedirs(self.dirpath, exist_ok=True)
#         name = self.filename_tmpl.format(epoch=epoch_1idx) + ".ckpt"
#         out = os.path.join(self.dirpath, name)

#         # DDP-safe: only global rank 0 writes
#         if trainer.is_global_zero:
#             trainer.save_checkpoint(out)
            
import os
from typing import Optional, Iterable

import pytorch_lightning as pl


class SaveAtSchedule(pl.Callback):
    """
    Save checkpoints:
      - every N epochs: save_every_n_epochs=5 -> epochs 5,10,15,...
      - plus explicit epochs: save_epochs=[2,3,10]
    Epochs are 1-indexed (human-readable).
    """

    def __init__(
        self,
        dirpath: str,
        save_every_n_epochs: int = 0,
        save_epochs: Optional[Iterable[int]] = None,
        filename_tmpl: str = "epoch={epoch:02d}",
        save_weights_only: bool = False,   # ✅ important for huge models
        barrier_after_save: bool = True,  # ✅ prevents ranks from drifting
    ):
        super().__init__()
        self.dirpath = dirpath
        self.save_every_n_epochs = int(save_every_n_epochs or 0)
        self.save_epochs = sorted({int(e) for e in (save_epochs or [])})
        self.filename_tmpl = filename_tmpl
        self.save_weights_only = bool(save_weights_only)
        self.barrier_after_save = bool(barrier_after_save)

        if self.save_every_n_epochs < 0:
            raise ValueError("save_every_n_epochs must be >= 0")

    def _should_save(self, epoch_1idx: int) -> bool:
        if epoch_1idx in self.save_epochs:
            return True
        if self.save_every_n_epochs > 0 and (epoch_1idx % self.save_every_n_epochs == 0):
            return True
        return False

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch_1idx = int(trainer.current_epoch) + 1
        if not self._should_save(epoch_1idx):
            return

        # Build output path once
        os.makedirs(self.dirpath, exist_ok=True)
        out = os.path.join(self.dirpath, self.filename_tmpl.format(epoch=epoch_1idx) + ".ckpt")

        # Only rank0 writes
        if trainer.is_global_zero:
            # In PL 1.9.5, save_checkpoint supports weights_only kwarg
            trainer.save_checkpoint(out, weights_only=self.save_weights_only)

        # Make sure all ranks wait until the file write is done
        if self.barrier_after_save and getattr(trainer, "world_size", 1) > 1:
            trainer.strategy.barrier()