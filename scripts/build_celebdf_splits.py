#!/usr/bin/env python3
"""
scripts/build_celebdf_splits.py
================================
Generate train / val / test CSV splits for Celeb-DF v2.

Input:
  - List_of_training_videos.txt  (at celebdf root)
  - List_of_testing_videos.txt   (at celebdf root)

Output (in split_csv2/):
  - celeb_df_v2_train.csv   (90% of training list)
  - celeb_df_v2_val.csv     (10% of training list)
  - celeb_df_v2_test.csv    (100% of testing list)

Frame folders expected at:
  celebdf/<source>-mtcnn/<vid>/*.png
  e.g. celebdf/Celeb-real-mtcnn/id0_0001/*.png

Usage:
  python scripts/build_celebdf_splits.py
"""

import os
import csv
import glob
import random
from pathlib import Path
from typing import List, Dict, Tuple

# ═══════════════════════════════════════════════
# CONFIG — adjust these paths if needed
# ═══════════════════════════════════════════════
CELEBDF_ROOT = "/scratch/sahil/projects/img_deepfake/datasets/celebdf"
TRAIN_LIST   = os.path.join(CELEBDF_ROOT, "List_of_training_videos.txt")
TEST_LIST    = os.path.join(CELEBDF_ROOT, "List_of_testing_videos.txt")
OUT_DIR      = os.path.join(CELEBDF_ROOT, "split_csv2")
MTCNN_SUFFIX = "-mtcnn"
TRAIN_RATIO  = 0.90   # 90% train, 10% val
SEED         = 42


def parse_list_file(path: str) -> List[Tuple[int, str]]:
    """
    Parse Celeb-DF list file.
    Format: '<label> <source>/<vid>.mp4'
    e.g.:   '0 Celeb-real/id0_0001.mp4'
            '1 Celeb-synthesis/id0_id1_0000.mp4'
    Returns: [(label, rel_path), ...]
    """
    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[0] in ("0", "1"):
                label = int(parts[0])
                rel = parts[1]
            elif len(parts) >= 2:
                # fallback: last token is path
                rel = parts[-1]
                label = 1 if "synthesis" in rel.lower() else 0
            else:
                continue
            entries.append((label, rel))
    return entries


def expand_to_frames(entries: List[Tuple[int, str]], root: str,
                     num_frames: int = 50) -> List[Dict]:
    """
    For each video entry, find its extracted frame folder and expand into
    per-frame rows. Each row has: path, label, source, video_id

    Frame folder layout: <root>/<source>-mtcnn/<vid>/*.png
    """
    rows = []
    missing_folders = 0
    empty_folders = 0

    for label, rel in entries:
        # rel = "Celeb-real/id0_0001.mp4"
        parts = rel.split("/")
        if len(parts) < 2:
            continue

        source = parts[0]                       # "Celeb-real"
        vid = os.path.splitext(parts[-1])[0]    # "id0_0001"

        # Frame folder: <root>/Celeb-real-mtcnn/id0_0001/
        frame_dir = os.path.join(root, f"{source}{MTCNN_SUFFIX}", vid)

        if not os.path.isdir(frame_dir):
            missing_folders += 1
            continue

        frames = sorted(
            glob.glob(os.path.join(frame_dir, "*.png")) +
            glob.glob(os.path.join(frame_dir, "*.jpg"))
        )

        if len(frames) == 0:
            empty_folders += 1
            continue

        # Uniform subsample if too many frames
        if len(frames) > num_frames:
            import numpy as np
            idx = np.linspace(0, len(frames) - 1, num_frames, endpoint=True, dtype=int)
            frames = [frames[i] for i in idx]

        for fp in frames:
            rows.append({
                "path":     fp,
                "label":    label,
                "source":   source,
                "video_id": vid,
                "dataset":  "celebdf",
            })

    if missing_folders > 0:
        print(f"  [WARN] missing frame folders: {missing_folders}")
    if empty_folders > 0:
        print(f"  [WARN] empty frame folders: {empty_folders}")

    return rows


def write_csv(rows: List[Dict], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = ["path", "label", "source", "video_id", "dataset"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  Written: {out_path} ({len(rows)} rows)")


def split_by_video(rows: List[Dict], ratio: float, seed: int
                   ) -> Tuple[List[Dict], List[Dict]]:
    """Split rows by video_id so all frames from one video stay together."""
    vid_to_rows = {}
    for r in rows:
        vid_to_rows.setdefault(r["video_id"], []).append(r)

    vids = sorted(vid_to_rows.keys())
    rng = random.Random(seed)
    rng.shuffle(vids)

    n_train = int(len(vids) * ratio)
    train_vids = set(vids[:n_train])

    train_rows = []
    val_rows = []
    for vid in vids:
        if vid in train_vids:
            train_rows.extend(vid_to_rows[vid])
        else:
            val_rows.extend(vid_to_rows[vid])

    return train_rows, val_rows


def main():
    print("=" * 60)
    print("Building Celeb-DF v2 splits → split_csv2/")
    print("=" * 60)

    assert os.path.exists(TRAIN_LIST), f"Missing: {TRAIN_LIST}"
    assert os.path.exists(TEST_LIST), f"Missing: {TEST_LIST}"

    # ── Parse list files ──
    print(f"\nParsing {TRAIN_LIST}")
    train_entries = parse_list_file(TRAIN_LIST)
    print(f"  Videos in training list: {len(train_entries)}")

    print(f"\nParsing {TEST_LIST}")
    test_entries = parse_list_file(TEST_LIST)
    print(f"  Videos in testing list: {len(test_entries)}")

    # ── Expand to frames ──
    print(f"\nExpanding training videos to frames...")
    train_all_rows = expand_to_frames(train_entries, CELEBDF_ROOT)
    n_real = sum(1 for r in train_all_rows if r["label"] == 0)
    n_fake = sum(1 for r in train_all_rows if r["label"] == 1)
    print(f"  Total training frames: {len(train_all_rows)} (real={n_real}, fake={n_fake})")

    print(f"\nExpanding testing videos to frames...")
    test_rows = expand_to_frames(test_entries, CELEBDF_ROOT)
    n_real_t = sum(1 for r in test_rows if r["label"] == 0)
    n_fake_t = sum(1 for r in test_rows if r["label"] == 1)
    print(f"  Total testing frames: {len(test_rows)} (real={n_real_t}, fake={n_fake_t})")

    # ── Split training 90:10 by video ──
    print(f"\nSplitting training → 90% train / 10% val (by video, seed={SEED})")
    train_rows, val_rows = split_by_video(train_all_rows, TRAIN_RATIO, SEED)
    print(f"  Train: {len(train_rows)} frames")
    print(f"  Val:   {len(val_rows)} frames")

    # ── Write CSVs ──
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\nWriting CSVs to {OUT_DIR}/")
    write_csv(train_rows, os.path.join(OUT_DIR, "celeb_df_v2_train.csv"))
    write_csv(val_rows,   os.path.join(OUT_DIR, "celeb_df_v2_val.csv"))
    write_csv(test_rows,  os.path.join(OUT_DIR, "celeb_df_v2_test.csv"))

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("DONE. Split summary:")
    print(f"  Train: {len(train_rows):>8} frames")
    print(f"  Val:   {len(val_rows):>8} frames")
    print(f"  Test:  {len(test_rows):>8} frames")
    print(f"\nCSV files in: {OUT_DIR}/")
    for fn in sorted(os.listdir(OUT_DIR)):
        if fn.endswith(".csv"):
            print(f"  - {fn}")
    print("=" * 60)


if __name__ == "__main__":
    main()
