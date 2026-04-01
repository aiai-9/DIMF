
#!/usr/bin/env python3
"""
Download and build the official DiffSwap binary benchmark from DiffFace part1.

Downloads:
  - DiffSwap.tar
  - Real.tar

Source:
  DiffFace repo says dataset is released in DiffFace-part1 / part2.
  DiffFace-part1 contains DiffSwap.tar and Real.tar.

Outputs:
  /scratch/sahil/projects/img_deepfake/datasets/diffswap/
    ├── downloads/
    │    ├── DiffSwap.tar
    │    └── Real.tar
    ├── extracted/
    │    ├── DiffSwap/...
    │    └── Real/...
    └── split_csv2/
         ├── diffswap_train.csv
         ├── diffswap_val.csv
         └── diffswap_test.csv
"""

import os
import csv
import tarfile
import random
from pathlib import Path
from typing import List, Dict, Tuple

import requests
from tqdm.auto import tqdm

# ============================================================
# CONFIG
# ============================================================
BASE = "/scratch/sahil/projects/img_deepfake/datasets"
DATASET_ROOT = os.path.join(BASE, "diffswap")
DOWNLOAD_DIR = os.path.join(DATASET_ROOT, "downloads")
EXTRACT_DIR = os.path.join(DATASET_ROOT, "extracted")
SPLIT_DIR = os.path.join(DATASET_ROOT, "split_csv2")

SEED = 42
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Official DiffFace part1 record files
FILES = {
    "DiffSwap.tar": "https://zenodo.org/records/10865300/files/DiffSwap.tar?download=1",
    "Real.tar": "https://zenodo.org/records/10865300/files/Real.tar?download=1",
}


# ============================================================
# HELPERS
# ============================================================
def ensure_dirs():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    os.makedirs(SPLIT_DIR, exist_ok=True)


def write_csv(rows: List[Dict], out_path: str, fieldnames: List[str]):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_real = sum(1 for r in rows if int(r["label"]) == 0)
    n_fake = len(rows) - n_real
    print(f"Written: {out_path} ({len(rows)} rows, real={n_real}, fake={n_fake})")


def download_file(url: str, out_path: str):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"[skip] already exists: {out_path}")
        return

    print(f"[download] {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with open(out_path, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=os.path.basename(out_path),
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))


def extract_tar(tar_path: str, out_dir: str):
    marker = os.path.join(out_dir, ".done")
    if os.path.exists(marker):
        print(f"[skip] already extracted: {out_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)
    print(f"[extract] {tar_path} -> {out_dir}")

    with tarfile.open(tar_path, "r") as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc=f"extract:{os.path.basename(tar_path)}"):
            tar.extract(member, path=out_dir)

    Path(marker).touch()


def collect_images_recursive(directory: str) -> List[str]:
    out = []
    if not os.path.isdir(directory):
        return out
    for p in Path(directory).rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out.append(str(p))
    return sorted(out)


def choose_group_id_for_real(img_path: Path, real_root: Path) -> str:
    """
    Try to group real images conservatively to reduce leakage.
    Strategy:
      - if path is Real/person_x/frame.png  -> group by Real/person_x
      - if path is deeper, group by first 1-2 levels
      - if flat, group by stem
    """
    rel = img_path.relative_to(real_root)
    parts = rel.parts

    # Real/<id>/<frame>.png
    if len(parts) >= 3:
        return f"real/{parts[0]}/{parts[1]}"
    # <id>/<frame>.png
    if len(parts) == 2:
        return f"real/{parts[0]}"
    # flat
    return f"real/{img_path.stem}"


def choose_group_id_for_fake(img_path: Path, fake_root: Path) -> str:
    """
    Fake side in DiffFace DiffSwap is typically image-based.
    We group by the immediate identity-like folder if present, otherwise by stem.
    """
    rel = img_path.relative_to(fake_root)
    parts = rel.parts

    if len(parts) >= 3:
        return f"fake/{parts[0]}/{parts[1]}"
    if len(parts) == 2:
        return f"fake/{parts[0]}"
    return f"fake/{img_path.stem}"


def split_rows_by_group_3way(
    rows: List[Dict],
    group_key: str,
    train_ratio: float,
    val_ratio: float,
    seed: int = SEED,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    groups = {}
    for r in rows:
        groups.setdefault(r[group_key], []).append(r)

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


def assert_no_group_overlap(train_rows, val_rows, test_rows, group_key="group_id"):
    tr = {r[group_key] for r in train_rows}
    va = {r[group_key] for r in val_rows}
    te = {r[group_key] for r in test_rows}

    tv = tr & va
    tt = tr & te
    vt = va & te

    if tv or tt or vt:
        raise RuntimeError(
            f"Leakage detected: train∩val={len(tv)}, train∩test={len(tt)}, val∩test={len(vt)}"
        )


def detect_single_extracted_subdir(root: str) -> str:
    """
    After extracting a tar, sometimes files land in:
      extracted/DiffSwap/DiffSwap/...
    or
      extracted/DiffSwap/...
    This returns the deepest obvious content root if there is exactly one subdir.
    """
    p = Path(root)
    if not p.exists():
        return root

    children = [x for x in p.iterdir() if x.is_dir()]
    if len(children) == 1:
        nested = children[0]
        # if nested actually contains images or more dirs, prefer nested
        return str(nested)
    return root


# ============================================================
# BUILD
# ============================================================
def download_all():
    ensure_dirs()
    for fname, url in FILES.items():
        out_path = os.path.join(DOWNLOAD_DIR, fname)
        download_file(url, out_path)


def extract_all():
    diffswap_tar = os.path.join(DOWNLOAD_DIR, "DiffSwap.tar")
    real_tar = os.path.join(DOWNLOAD_DIR, "Real.tar")

    diffswap_out = os.path.join(EXTRACT_DIR, "DiffSwap")
    real_out = os.path.join(EXTRACT_DIR, "Real")

    extract_tar(diffswap_tar, diffswap_out)
    extract_tar(real_tar, real_out)


def build_csvs():
    fields = ["path", "label", "source", "video_id", "dataset", "group_id"]

    fake_root = detect_single_extracted_subdir(os.path.join(EXTRACT_DIR, "DiffSwap"))
    real_root = detect_single_extracted_subdir(os.path.join(EXTRACT_DIR, "Real"))

    print(f"[scan] fake root: {fake_root}")
    print(f"[scan] real root: {real_root}")

    fake_imgs = collect_images_recursive(fake_root)
    real_imgs = collect_images_recursive(real_root)

    print(f"[scan] fake images found: {len(fake_imgs)}")
    print(f"[scan] real images found: {len(real_imgs)}")

    if len(fake_imgs) == 0:
        raise RuntimeError("No fake images found after extracting DiffSwap.tar")
    if len(real_imgs) == 0:
        raise RuntimeError("No real images found after extracting Real.tar")

    rows = []

    # Fake rows
    fake_root_p = Path(fake_root)
    for fp in fake_imgs:
        p = Path(fp)
        video_id = p.stem
        group_id = choose_group_id_for_fake(p, fake_root_p)
        rows.append({
            "path": fp,
            "label": 1,
            "source": "DiffSwap",
            "video_id": video_id,
            "dataset": "diffswap",
            "group_id": group_id,
        })

    # Real rows
    real_root_p = Path(real_root)
    for fp in real_imgs:
        p = Path(fp)
        video_id = p.stem
        group_id = choose_group_id_for_real(p, real_root_p)
        rows.append({
            "path": fp,
            "label": 0,
            "source": "Real",
            "video_id": video_id,
            "dataset": "diffswap",
            "group_id": group_id,
        })

    n_real = sum(1 for r in rows if r["label"] == 0)
    n_fake = len(rows) - n_real
    print(f"[build] total rows before split: {len(rows)}")
    print(f"[build] real={n_real}, fake={n_fake}")

    train_rows, val_rows, test_rows = split_rows_by_group_3way(
        rows,
        group_key="group_id",
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        seed=SEED,
    )

    assert_no_group_overlap(train_rows, val_rows, test_rows)

    if len(train_rows) + len(val_rows) + len(test_rows) != len(rows):
        raise RuntimeError("Row mismatch after split")

    print(
        f"[split] train={len(train_rows)} val={len(val_rows)} test={len(test_rows)} "
        f"(sum={len(train_rows)+len(val_rows)+len(test_rows)})"
    )

    write_csv(train_rows, os.path.join(SPLIT_DIR, "diffswap_train.csv"), fields)
    write_csv(val_rows,   os.path.join(SPLIT_DIR, "diffswap_val.csv"),   fields)
    write_csv(test_rows,  os.path.join(SPLIT_DIR, "diffswap_test.csv"),  fields)


def main():
    print("=" * 70)
    print("Download + extract + build official DiffSwap benchmark")
    print("=" * 70)
    download_all()
    extract_all()
    build_csvs()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()



# python scripts/download_and_build_diffswap_official.py
