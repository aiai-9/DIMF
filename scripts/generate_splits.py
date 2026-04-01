#!/usr/bin/env python3
"""
scripts/generate_splits.py — Generate train/val/test CSVs for cross-dataset training
=====================================================================================
Run ONCE before starting mixed-domain training.

Usage:
    python scripts/generate_splits.py

Creates:
    celebdf/split_csv/celeb_df_v2_train.csv   (80% of test CSV → train)
    celebdf/split_csv/celeb_df_v2_val.csv     (20% of test CSV → val)
    wild_deepfake/split_csv/wilddeepfake_train.csv
    wild_deepfake/split_csv/wilddeepfake_val.csv
    diffswap/split_csv/diffswap_train.csv
    diffswap/split_csv/diffswap_val.csv

NOTE: Adjust BASE_DIR to match your dataset root.
"""

import os
import pandas as pd
import numpy as np

BASE_DIR = "/scratch/sahil/projects/img_deepfake/datasets"
SEED = 42
TRAIN_RATIO = 0.80


def split_csv(input_csv: str, output_dir: str, prefix: str,
              group_col: str = None, seed: int = SEED):
    """
    Split a CSV into train (80%) and val (20%).
    If group_col is provided, split by unique values in that column
    (so all frames from one video stay together).
    """
    df = pd.read_csv(input_csv)
    print(f"  Input: {input_csv} -> {len(df)} rows")

    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.RandomState(seed)

    if group_col and group_col in df.columns:
        groups = df[group_col].unique()
        rng.shuffle(groups)
        n_train = int(len(groups) * TRAIN_RATIO)
        train_groups = set(groups[:n_train])
        val_groups = set(groups[n_train:])

        train_df = df[df[group_col].isin(train_groups)]
        val_df = df[df[group_col].isin(val_groups)]
    else:
        indices = np.arange(len(df))
        rng.shuffle(indices)
        n_train = int(len(df) * TRAIN_RATIO)
        train_df = df.iloc[indices[:n_train]]
        val_df = df.iloc[indices[n_train:]]

    train_path = os.path.join(output_dir, f"{prefix}_train.csv")
    val_path = os.path.join(output_dir, f"{prefix}_val.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"  Train: {train_path} -> {len(train_df)} rows")
    print(f"  Val:   {val_path} -> {len(val_df)} rows")


def main():
    print("=" * 60)
    print("Generating train/val splits for cross-dataset training")
    print("=" * 60)

    # ── Celeb-DF ──
    print("\n[1] Celeb-DF")
    celebdf_csv = os.path.join(BASE_DIR, "celebdf", "split_csv", "celeb_df_v2_test.csv")
    if os.path.exists(celebdf_csv):
        split_csv(
            celebdf_csv,
            os.path.join(BASE_DIR, "celebdf", "split_csv"),
            "celeb_df_v2",
            group_col="path",  # split by video path
        )
    else:
        print(f"  SKIP: {celebdf_csv} not found")

    # ── WildDeepfake ──
    print("\n[2] WildDeepfake")
    wdf_csv = os.path.join(BASE_DIR, "wild_deepfake", "split_csv", "wilddeepfake_test.csv")
    if os.path.exists(wdf_csv):
        split_csv(
            wdf_csv,
            os.path.join(BASE_DIR, "wild_deepfake", "split_csv"),
            "wilddeepfake",
            group_col=None,  # no video grouping
        )
    else:
        print(f"  SKIP: {wdf_csv} not found")

    # ── DiffSwap ──
    print("\n[3] DiffSwap")
    ds_csv = os.path.join(BASE_DIR, "diffswap", "split_csv", "diffswap_mixed_full.csv")
    if os.path.exists(ds_csv):
        split_csv(
            ds_csv,
            os.path.join(BASE_DIR, "diffswap", "split_csv"),
            "diffswap",
            group_col=None,
        )
    else:
        print(f"  SKIP: {ds_csv} not found")

    print("\n" + "=" * 60)
    print("Done. You can now use configs/train_mixed.yaml")
    print("=" * 60)


if __name__ == "__main__":
    main()
