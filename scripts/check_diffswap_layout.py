# scripts/check_diffswap_layout.py

#!/usr/bin/env python3
import os
import json
from pathlib import Path
from collections import Counter, defaultdict

BASE = "/scratch/sahil/projects/img_deepfake/datasets/diffswap"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def collect_images(directory):
    out = []
    if not os.path.isdir(directory):
        return out
    for p in Path(directory).rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out.append(p)
    return sorted(out)

def print_top(counter, title, k=20):
    print(f"\n{title}")
    for x, c in counter.most_common(k):
        print(f"  {x}: {c}")

def main():
    root = BASE
    print("=" * 70)
    print("Checking DiffSwap layout")
    print("=" * 70)
    print("Root:", root)
    print("Exists:", os.path.isdir(root))

    if not os.path.isdir(root):
        print("Dataset root not found.")
        return

    # show shallow tree
    print("\nTop-level contents:")
    for p in sorted(Path(root).iterdir()):
        kind = "DIR " if p.is_dir() else "FILE"
        print(f"  {kind} {p.name}")

    # fake side
    wild_img_dir = os.path.join(root, "Wild", "images_256")
    base_json = os.path.join(root, "Wild", "base.json")

    fake_imgs = collect_images(wild_img_dir)
    print(f"\nFake images in Wild/images_256: {len(fake_imgs)}")

    fake_stems = [p.stem for p in fake_imgs]
    fake_stem_counts = Counter(fake_stems)
    dup_fake_stems = {k: v for k, v in fake_stem_counts.items() if v > 1}
    print(f"Unique fake stems: {len(fake_stem_counts)}")
    print(f"Duplicate fake stems: {len(dup_fake_stems)}")
    if dup_fake_stems:
        print_top(Counter(dup_fake_stems), "Top duplicate fake stems")

    if os.path.isfile(base_json):
        with open(base_json, "r") as f:
            base = json.load(f)
        print(f"Entries in base.json: {len(base)}")

        key_types = Counter(type(k).__name__ for k in base.keys())
        val_types = Counter(type(v).__name__ for v in base.values())
        print("Key types:", dict(key_types))
        print("Value types:", dict(val_types))

        # summarize metadata fields if dict values
        field_counter = Counter()
        sample_dicts = 0
        for v in base.values():
            if isinstance(v, dict):
                sample_dicts += 1
                field_counter.update(v.keys())

        print(f"Dict-like entries in base.json: {sample_dicts}")
        if field_counter:
            print_top(field_counter, "Most common metadata fields in base.json")

        # detect useful identity/pair fields
        candidate_fields = [
            "id", "src", "source", "source_id", "target", "target_id",
            "pair_id", "person_id", "identity", "image_id", "orig_id"
        ]

        present = [f for f in candidate_fields if field_counter[f] > 0]
        print("\nCandidate identity/pair fields present:", present if present else "None obvious")

        # analyze many-to-one mapping if candidate fields exist
        for fld in present:
            counter = Counter()
            for v in base.values():
                if isinstance(v, dict) and fld in v:
                    counter[str(v[fld])] += 1
            repeated = {k: c for k, c in counter.items() if c > 1}
            print(f"\nField '{fld}': unique={len(counter)}, repeated_values={len(repeated)}")
            if repeated:
                print_top(Counter(repeated), f"Top repeated values for '{fld}'")

    # real side
    real_dirs = ["real", "Real", "original", "CelebA-HQ"]
    all_real_imgs = []
    real_dir_hits = []
    for d in real_dirs:
        p = os.path.join(root, d)
        if os.path.isdir(p):
            imgs = collect_images(p)
            all_real_imgs.extend(imgs)
            real_dir_hits.append((d, len(imgs)))

    print("\nReal-image folders found:")
    if real_dir_hits:
        for d, n in real_dir_hits:
            print(f"  {d}: {n}")
    else:
        print("  None")

    real_stems = [p.stem for p in all_real_imgs]
    real_stem_counts = Counter(real_stems)
    dup_real_stems = {k: v for k, v in real_stem_counts.items() if v > 1}
    print(f"\nTotal real images: {len(all_real_imgs)}")
    print(f"Unique real stems: {len(real_stem_counts)}")
    print(f"Duplicate real stems: {len(dup_real_stems)}")
    if dup_real_stems:
        print_top(Counter(dup_real_stems), "Top duplicate real stems")

    # path-parent analysis
    fake_parents = Counter(str(p.parent.relative_to(Path(wild_img_dir))) for p in fake_imgs) if fake_imgs else Counter()
    real_parents = Counter()
    for p in all_real_imgs:
        for d in real_dirs:
            droot = Path(root) / d
            if droot in p.parents:
                rel_parent = str(p.parent.relative_to(droot))
                real_parents[rel_parent] += 1
                break

    if fake_parents:
        print_top(fake_parents, "Top fake parent folders")
    if real_parents:
        print_top(real_parents, "Top real parent folders")

    # overlap of stems across real/fake
    overlap = set(fake_stems) & set(real_stems)
    print(f"\nStem overlap between real and fake files: {len(overlap)}")
    if overlap:
        print("Examples:", sorted(list(overlap))[:20])

    print("\nInterpretation guide:")
    print("1) If base.json has identity/source/pair fields with repeated values,")
    print("   splitting by img_id alone is probably too weak.")
    print("2) If many real images share identity folders, split by identity/folder instead of stem.")
    print("3) If real/fake share common stems or common pair fields, use that as group_id.")
    print("4) If every fake img_id is truly unique and independent, current split is likely acceptable.")

if __name__ == "__main__":
    main()
