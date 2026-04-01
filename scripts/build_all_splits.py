#!/usr/bin/env python3
"""
scripts/build_all_splits.py
===========================

Generate train / val / test CSV splits for:
  1. Celeb-DF v2
  2. DFD (DeepFakeDetection)
  3. DiffSwap
  4. WildDeepfake

Output:
  split_csv2/ inside each dataset root

Design goals:
- No accidental row drops
- Grouped splitting to reduce leakage
- Prefer pre-extracted frames if available
- Fall back to raw videos when frames are missing
- Respect official train/valid/test structure when provided
"""

import os
import csv
import json
import glob
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
BASE = "/scratch/sahil/projects/img_deepfake/datasets"
SEED = 42
MAX_FRAMES_PER_VIDEO = 50

CELEBDF_ROOT = os.path.join(BASE, "celebdf")
DFD_ROOT = os.path.join(BASE, "dfd")
DIFFSWAP_ROOT = os.path.join(BASE, "diffswap")
WILDDEEPFAKE_ROOT = os.path.join(BASE, "wild_deepfake")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ============================================================
# HELPERS
# ============================================================
def write_csv(rows: List[Dict], out_path: str, fieldnames: List[str]):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_real = sum(1 for r in rows if int(r["label"]) == 0)
    n_fake = len(rows) - n_real
    print(f"    Written: {out_path}  ({len(rows)} rows, real={n_real}, fake={n_fake})")


def sample_frames(frame_paths: List[str], max_frames: int = MAX_FRAMES_PER_VIDEO) -> List[str]:
    if len(frame_paths) <= max_frames:
        return frame_paths
    idx = np.linspace(0, len(frame_paths) - 1, max_frames, endpoint=True, dtype=int)
    return [frame_paths[i] for i in idx]


def collect_images_recursive(directory: str) -> List[str]:
    paths = []
    if not os.path.isdir(directory):
        return paths
    for p in Path(directory).rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            paths.append(str(p))
    return sorted(paths)


def collect_videos_recursive(directory: str) -> List[str]:
    paths = []
    if not os.path.isdir(directory):
        return paths
    for p in Path(directory).rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            paths.append(str(p))
    return sorted(paths)


def make_row(
    path: str,
    label: int,
    source: str,
    video_id: str,
    dataset: str,
    group_id: Optional[str] = None,
) -> Dict:
    row = {
        "path": path,
        "label": int(label),
        "source": source,
        "video_id": video_id,
        "dataset": dataset,
    }
    if group_id is not None:
        row["group_id"] = group_id
    return row


def split_rows_by_group_3way(
    rows: List[Dict],
    group_key: str,
    train_ratio: float,
    val_ratio: float,
    seed: int = SEED,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    groups = {}
    for r in rows:
        gid = r[group_key]
        groups.setdefault(gid, []).append(r)

    group_ids = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    n = len(group_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = set(group_ids[:n_train])
    val_ids = set(group_ids[n_train:n_train + n_val])
    test_ids = set(group_ids[n_train + n_val:])

    train_rows = [r for gid in train_ids for r in groups[gid]]
    val_rows = [r for gid in val_ids for r in groups[gid]]
    test_rows = [r for gid in test_ids for r in groups[gid]]
    return train_rows, val_rows, test_rows


def split_rows_by_group_2way(
    rows: List[Dict],
    group_key: str,
    train_ratio: float,
    seed: int = SEED,
) -> Tuple[List[Dict], List[Dict]]:
    groups = {}
    for r in rows:
        gid = r[group_key]
        groups.setdefault(gid, []).append(r)

    group_ids = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    n_train = int(len(group_ids) * train_ratio)
    train_ids = set(group_ids[:n_train])
    val_ids = set(group_ids[n_train:])

    train_rows = [r for gid in train_ids for r in groups[gid]]
    val_rows = [r for gid in val_ids for r in groups[gid]]
    return train_rows, val_rows


def assert_no_group_overlap(
    train_rows: List[Dict],
    val_rows: List[Dict],
    test_rows: Optional[List[Dict]] = None,
    group_key: str = "group_id",
    name: str = "dataset",
):
    test_rows = test_rows or []

    tr = {r[group_key] for r in train_rows}
    va = {r[group_key] for r in val_rows}
    te = {r[group_key] for r in test_rows}

    tv = tr & va
    tt = tr & te
    vt = va & te

    if tv or tt or vt:
        raise RuntimeError(
            f"[{name}] group leakage detected: "
            f"train∩val={len(tv)}, train∩test={len(tt)}, val∩test={len(vt)}"
        )


def print_split_summary(name: str, train_rows: List[Dict], val_rows: List[Dict], test_rows: List[Dict]):
    total = len(train_rows) + len(val_rows) + len(test_rows)
    print(f"  {name} total rows after split: {total}")


# ============================================================
# 1. CELEB-DF v2
# ============================================================
def build_celebdf():
    print("\n" + "=" * 60)
    print("[1] Celeb-DF v2")
    print("=" * 60)

    root = CELEBDF_ROOT
    out_dir = os.path.join(root, "split_csv2")
    fields = ["path", "label", "source", "video_id", "dataset", "group_id"]

    train_list = os.path.join(root, "List_of_training_videos.txt")
    test_list = os.path.join(root, "List_of_testing_videos.txt")

    def parse_list(list_path: str):
        entries = []
        with open(list_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2 and parts[0] in ("0", "1"):
                    label = int(parts[0])
                    rel = parts[1]
                else:
                    rel = parts[-1]
                    rel_l = rel.lower()
                    label = 1 if "synthesis" in rel_l or "fake" in rel_l else 0
                entries.append((label, rel))
        return entries

    def expand_entries(entries):
        rows = []
        missing = 0

        for label, rel in entries:
            rel_path = rel.replace("\\", "/")
            parts = rel_path.split("/")
            if len(parts) < 2:
                continue

            source = parts[0]
            video_file = parts[-1]
            vid = os.path.splitext(video_file)[0]
            group_id = f"{source}/{vid}"

            mtcnn_dir = os.path.join(root, f"{source}-mtcnn", vid)
            frame_paths = []
            if os.path.isdir(mtcnn_dir):
                frame_paths = sorted(
                    glob.glob(os.path.join(mtcnn_dir, "*.png")) +
                    glob.glob(os.path.join(mtcnn_dir, "*.jpg")) +
                    glob.glob(os.path.join(mtcnn_dir, "*.jpeg"))
                )

            if frame_paths:
                frame_paths = sample_frames(frame_paths, MAX_FRAMES_PER_VIDEO)
                for fp in frame_paths:
                    rows.append(make_row(fp, label, source, vid, "celebdf", group_id))
                continue

            raw_video_path = os.path.join(root, rel_path)
            if os.path.isfile(raw_video_path):
                rows.append(make_row(raw_video_path, label, source, vid, "celebdf", group_id))
            else:
                missing += 1

        if missing > 0:
            print(f"  WARN: missing {missing} listed videos/frame dirs")
        return rows

    if not os.path.exists(train_list):
        print(f"  SKIP: missing {train_list}")
        return
    if not os.path.exists(test_list):
        print(f"  SKIP: missing {test_list}")
        return

    train_entries = parse_list(train_list)
    test_entries = parse_list(test_list)

    print(f"  Training list: {len(train_entries)} videos")
    print(f"  Testing list:  {len(test_entries)} videos")

    train_all_rows = expand_entries(train_entries)
    test_rows = expand_entries(test_entries)

    print(f"  Training rows before split: {len(train_all_rows)}")
    print(f"  Testing rows:              {len(test_rows)}")

    # 2-way split only; no row drop
    train_rows, val_rows = split_rows_by_group_2way(
        train_all_rows,
        group_key="group_id",
        train_ratio=0.90,
        seed=SEED,
    )

    assert_no_group_overlap(train_rows, val_rows, test_rows, name="CelebDF")

    # Cross-check no rows lost
    if len(train_rows) + len(val_rows) != len(train_all_rows):
        raise RuntimeError(
            f"[CelebDF] row mismatch: train+val={len(train_rows)+len(val_rows)} "
            f"!= train_all={len(train_all_rows)}"
        )

    print_split_summary("CelebDF", train_rows, val_rows, test_rows)

    write_csv(train_rows, os.path.join(out_dir, "celeb_df_v2_train.csv"), fields)
    write_csv(val_rows,   os.path.join(out_dir, "celeb_df_v2_val.csv"),   fields)
    write_csv(test_rows,  os.path.join(out_dir, "celeb_df_v2_test.csv"),  fields)


# ============================================================
# 2. DFD (DeepFakeDetection)
# ============================================================
def build_dfd():
    print("\n" + "=" * 60)
    print("[2] DFD (DeepFakeDetection)")
    print("=" * 60)

    root = DFD_ROOT
    out_dir = os.path.join(root, "split_csv2")
    fields = ["path", "label", "source", "video_id", "dataset", "group_id"]

    real_dir = os.path.join(root, "DFD_original_sequences")
    fake_dir = os.path.join(root, "DFD_manipulated_sequences")

    def collect_items(base_dir: str, label: int, source_name: str):
        rows = []

        if not os.path.exists(base_dir):
            print(f"  WARN: missing {base_dir}")
            return rows

        video_paths = collect_videos_recursive(base_dir)

        if video_paths:
            for vp in video_paths:
                vid = Path(vp).stem
                group_id = f"{source_name}/{vid}"

                mtcnn_dir = os.path.join(root, f"{source_name}-mtcnn", vid)
                frame_paths = []
                if os.path.isdir(mtcnn_dir):
                    frame_paths = sorted(
                        glob.glob(os.path.join(mtcnn_dir, "*.png")) +
                        glob.glob(os.path.join(mtcnn_dir, "*.jpg")) +
                        glob.glob(os.path.join(mtcnn_dir, "*.jpeg"))
                    )

                if frame_paths:
                    frame_paths = sample_frames(frame_paths, MAX_FRAMES_PER_VIDEO)
                    for fp in frame_paths:
                        rows.append(make_row(fp, label, source_name, vid, "dfd", group_id))
                else:
                    rows.append(make_row(vp, label, source_name, vid, "dfd", group_id))
            return rows

        image_paths = collect_images_recursive(base_dir)
        for ip in image_paths:
            rel_parent = str(Path(ip).parent.relative_to(base_dir))
            vid = Path(ip).stem
            group_id = f"{source_name}/{rel_parent}/{vid}"
            rows.append(make_row(ip, label, source_name, vid, "dfd", group_id))

        return rows

    real_rows = collect_items(real_dir, 0, "DFD_original_sequences")
    fake_rows = collect_items(fake_dir, 1, "DFD_manipulated_sequences")
    all_rows = real_rows + fake_rows

    print(f"  Total rows before split: {len(all_rows)} (real={len(real_rows)}, fake={len(fake_rows)})")

    if not all_rows:
        print("  SKIP: no DFD data found")
        return

    train_rows, val_rows, test_rows = split_rows_by_group_3way(
        all_rows,
        group_key="group_id",
        train_ratio=0.80,
        val_ratio=0.10,
        seed=SEED,
    )

    assert_no_group_overlap(train_rows, val_rows, test_rows, name="DFD")

    if len(train_rows) + len(val_rows) + len(test_rows) != len(all_rows):
        raise RuntimeError(
            f"[DFD] row mismatch: split sum={len(train_rows)+len(val_rows)+len(test_rows)} "
            f"!= all_rows={len(all_rows)}"
        )

    print_split_summary("DFD", train_rows, val_rows, test_rows)

    write_csv(train_rows, os.path.join(out_dir, "dfd_train.csv"), fields)
    write_csv(val_rows,   os.path.join(out_dir, "dfd_val.csv"),   fields)
    write_csv(test_rows,  os.path.join(out_dir, "dfd_test.csv"),  fields)


# ============================================================
# 3. DiffSwap
# ============================================================
def build_diffswap():
    print("\n" + "=" * 60)
    print("[3] DiffSwap")
    print("=" * 60)

    root = DIFFSWAP_ROOT
    out_dir = os.path.join(root, "split_csv2")
    fields = ["path", "label", "source", "video_id", "dataset", "group_id"]

    rows = []

    # --------------------------------------------------------
    # Native fake images from DiffSwap Wild subset
    # --------------------------------------------------------
    wild_img_dir = os.path.join(root, "Wild", "images_256")
    base_json = os.path.join(root, "Wild", "base.json")

    if not os.path.isdir(wild_img_dir):
        print(f"  SKIP: missing fake image directory: {wild_img_dir}")
        return

    if not os.path.isfile(base_json):
        print(f"  SKIP: missing metadata file: {base_json}")
        return

    with open(base_json, "r") as f:
        base = json.load(f)

    img_map = {}
    for p in Path(wild_img_dir).rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            img_map[p.stem] = str(p)

    print(f"  base.json entries: {len(base)}, images found: {len(img_map)}")

    missing_fake = 0
    fake_rows = []
    for k, v in base.items():
        img_id = str(k)
        p = img_map.get(img_id)

        if p is None and isinstance(v, dict) and "id" in v:
            p = img_map.get(str(v["id"]))

        if p is None:
            missing_fake += 1
            continue

        # Native fake group id: one image per fake id
        group_id = f"fake/{img_id}"
        fake_rows.append(
            make_row(
                path=p,
                label=1,
                source="diffswap_wild",
                video_id=img_id,
                dataset="diffswap",
                group_id=group_id,
            )
        )

    if missing_fake > 0:
        print(f"  WARN: missing images for {missing_fake} fake ids")

    # --------------------------------------------------------
    # Native real images ONLY from DiffSwap root
    # --------------------------------------------------------
    real_rows = []
    native_real_dirs = [
        "real",
        "Real",
        "original",
        "Original",
        "CelebA-HQ",
        "CelebA_HQ",
        "source",
        "sources",
    ]

    found_real_dirs = []
    for real_dir_name in native_real_dirs:
        real_dir = os.path.join(root, real_dir_name)
        if not os.path.isdir(real_dir):
            continue

        found_real_dirs.append(real_dir_name)

        for p in Path(real_dir).rglob("*"):
            if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
                continue

            rel_parent = str(p.parent.relative_to(real_dir)) if p.parent != Path(real_dir) else ""
            vid = p.stem

            # Group by native path structure, not by outside dataset fallback
            if rel_parent:
                group_id = f"real/{real_dir_name}/{rel_parent}"
            else:
                group_id = f"real/{real_dir_name}/{vid}"

            real_rows.append(
                make_row(
                    path=str(p),
                    label=0,
                    source=real_dir_name,
                    video_id=vid,
                    dataset="diffswap",
                    group_id=group_id,
                )
            )

    print(f"  Native fake rows: {len(fake_rows)}")
    print(f"  Native real rows: {len(real_rows)}")

    if found_real_dirs:
        print("  Native real dirs found:", ", ".join(found_real_dirs))
    else:
        print("  Native real dirs found: none")

    # --------------------------------------------------------
    # Strict behavior: do NOT mix with CelebDF fallback
    # --------------------------------------------------------
    if len(fake_rows) == 0:
        print("  SKIP: no native fake DiffSwap images found")
        return

    if len(real_rows) == 0:
        print("  SKIP: no native real DiffSwap images found under DiffSwap root")
        print("  NOTE: old fallback to CelebDF real images is intentionally disabled")
        return

    rows = real_rows + fake_rows

    n_real = len(real_rows)
    n_fake = len(fake_rows)
    print(f"  Total rows before split: {len(rows)}")
    print(f"  Real: {n_real}, Fake: {n_fake}")

    train_rows, val_rows, test_rows = split_rows_by_group_3way(
        rows,
        group_key="group_id",
        train_ratio=0.80,
        val_ratio=0.10,
        seed=SEED,
    )

    assert_no_group_overlap(train_rows, val_rows, test_rows, name="DiffSwap")

    if len(train_rows) + len(val_rows) + len(test_rows) != len(rows):
        raise RuntimeError(
            f"[DiffSwap] row mismatch: split sum={len(train_rows)+len(val_rows)+len(test_rows)} "
            f"!= all_rows={len(rows)}"
        )

    print_split_summary("DiffSwap", train_rows, val_rows, test_rows)

    write_csv(train_rows, os.path.join(out_dir, "diffswap_train.csv"), fields)
    write_csv(val_rows,   os.path.join(out_dir, "diffswap_val.csv"),   fields)
    write_csv(test_rows,  os.path.join(out_dir, "diffswap_test.csv"),  fields)
    
    
def build_diffswap_mixed():
    print("\n" + "=" * 60)
    print("[3] DiffSwap-Mixed (DiffSwap fake + CelebDF real)")
    print("=" * 60)

    out_dir = os.path.join(DIFFSWAP_ROOT, "split_csv2")
    os.makedirs(out_dir, exist_ok=True)
    fields = ["path", "label", "source", "video_id", "dataset", "group_id"]

    rows = []

    # --------------------------------------------------------
    # Fake: native DiffSwap fake images
    # --------------------------------------------------------
    wild_img_dir = os.path.join(DIFFSWAP_ROOT, "Wild", "images_256")
    base_json = os.path.join(DIFFSWAP_ROOT, "Wild", "base.json")

    if not os.path.isdir(wild_img_dir):
        print(f"  SKIP: missing fake image directory: {wild_img_dir}")
        return
    if not os.path.isfile(base_json):
        print(f"  SKIP: missing metadata file: {base_json}")
        return

    with open(base_json, "r") as f:
        base = json.load(f)

    img_map = {}
    for p in Path(wild_img_dir).rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            img_map[p.stem] = str(p)

    fake_rows = []
    missing_fake = 0
    for k, v in base.items():
        img_id = str(k)
        p = img_map.get(img_id)
        if p is None and isinstance(v, dict) and "id" in v:
            p = img_map.get(str(v["id"]))
        if p is None:
            missing_fake += 1
            continue

        group_id = f"diffswap_fake/{img_id}"
        fake_rows.append(make_row(
            path=p,
            label=1,
            source="diffswap_wild",
            video_id=img_id,
            dataset="diffswap_mixed_celebdfreal",
            group_id=group_id,
        ))

    # --------------------------------------------------------
    # Real: CelebDF real only
    # --------------------------------------------------------
    celeb_real_dir = os.path.join(CELEBDF_ROOT, "Celeb-real-mtcnn")
    if not os.path.isdir(celeb_real_dir):
        print(f"  SKIP: missing CelebDF real dir: {celeb_real_dir}")
        return

    real_rows = []
    for vid_dir in sorted(Path(celeb_real_dir).iterdir()):
        if not vid_dir.is_dir():
            continue

        frame_paths = sorted(
            glob.glob(os.path.join(str(vid_dir), "*.png")) +
            glob.glob(os.path.join(str(vid_dir), "*.jpg")) +
            glob.glob(os.path.join(str(vid_dir), "*.jpeg"))
        )
        if not frame_paths:
            continue

        frame_paths = sample_frames(frame_paths, MAX_FRAMES_PER_VIDEO)
        vid = vid_dir.name
        group_id = f"celebdf_real/{vid}"

        for fp in frame_paths:
            real_rows.append(make_row(
                path=fp,
                label=0,
                source="celebdf_real",
                video_id=vid,
                dataset="diffswap_mixed_celebdfreal",
                group_id=group_id,
            ))

    rows = real_rows + fake_rows

    print(f"  Fake rows (DiffSwap): {len(fake_rows)}")
    print(f"  Real rows (CelebDF):  {len(real_rows)}")
    print(f"  Total rows before split: {len(rows)}")

    if not rows:
        print("  SKIP: no rows found")
        return

    train_rows, val_rows, test_rows = split_rows_by_group_3way(
        rows,
        group_key="group_id",
        train_ratio=0.80,
        val_ratio=0.10,
        seed=SEED,
    )

    assert_no_group_overlap(train_rows, val_rows, test_rows, name="DiffSwap-Mixed")

    if len(train_rows) + len(val_rows) + len(test_rows) != len(rows):
        raise RuntimeError(
            f"[DiffSwap-Mixed] row mismatch: split sum={len(train_rows)+len(val_rows)+len(test_rows)} "
            f"!= all_rows={len(rows)}"
        )

    print_split_summary("DiffSwap-Mixed", train_rows, val_rows, test_rows)

    write_csv(train_rows, os.path.join(out_dir, "diffswap_mixed_celebdfreal_train.csv"), fields)
    write_csv(val_rows,   os.path.join(out_dir, "diffswap_mixed_celebdfreal_val.csv"),   fields)
    write_csv(test_rows,  os.path.join(out_dir, "diffswap_mixed_celebdfreal_test.csv"),  fields)




# ============================================================
# 4. WildDeepfake
# ============================================================
def build_wilddeepfake():
    print("\n" + "=" * 60)
    print("[4] WildDeepfake")
    print("=" * 60)

    root = WILDDEEPFAKE_ROOT
    out_dir = os.path.join(root, "split_csv2")
    fields = ["path", "label", "source", "video_id", "dataset", "group_id"]

    def collect_images_from_class(split_dir: str, class_name: str, label: int, split_name: str):
        rows = []
        class_dir = os.path.join(split_dir, class_name)

        if not os.path.isdir(class_dir):
            print(f"  WARN: missing folder: {class_dir}")
            return rows

        for p in sorted(Path(class_dir).rglob("*")):
            if not p.is_file():
                continue
            if p.suffix.lower() not in IMAGE_EXTS:
                continue

            rel_path = p.relative_to(split_dir)
            video_id = str(rel_path.with_suffix(""))
            group_id = f"{split_name}/{video_id}"

            rows.append(make_row(str(p), label, split_name, video_id, "wilddeepfake", group_id))

        return rows

    def collect_split(split_name: str):
        split_dir = os.path.join(root, split_name)
        if not os.path.isdir(split_dir):
            print(f"  WARN: split folder not found: {split_dir}")
            return []

        real_rows = collect_images_from_class(split_dir, "real", 0, split_name)
        fake_rows = collect_images_from_class(split_dir, "fake", 1, split_name)
        rows = real_rows + fake_rows

        print(
            f"  {split_name.capitalize()} dir: {len(rows)} images "
            f"(real={len(real_rows)}, fake={len(fake_rows)})"
        )
        return rows

    train_rows = collect_split("train")
    val_rows = collect_split("valid")
    test_rows = collect_split("test")

    if len(train_rows) == 0 and len(val_rows) == 0 and len(test_rows) == 0:
        print("  SKIP: no WildDeepfake data found")
        return

    assert_no_group_overlap(train_rows, val_rows, test_rows, name="WildDeepfake")

    print_split_summary("WildDeepfake", train_rows, val_rows, test_rows)

    write_csv(train_rows, os.path.join(out_dir, "wilddeepfake_train.csv"), fields)
    write_csv(val_rows,   os.path.join(out_dir, "wilddeepfake_val.csv"),   fields)
    write_csv(test_rows,  os.path.join(out_dir, "wilddeepfake_test.csv"),  fields)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Building ALL dataset splits -> split_csv2/")
    print("=" * 60)

    build_celebdf()
    build_dfd()
    # build_diffswap()
    # build_diffswap_mixed()
    build_wilddeepfake()

    print("\n" + "=" * 60)
    print("ALL DONE. CSVs created in split_csv2/ for each dataset.")
    print("=" * 60)



# python scripts/build_all_splits.py