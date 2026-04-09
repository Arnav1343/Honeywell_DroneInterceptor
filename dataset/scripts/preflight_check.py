#!/usr/bin/env python3
"""
preflight_check.py — Full GO/NO-GO validation gate.
Usage:
    python preflight_check.py              # checks Dataset_Fixed
    python preflight_check.py thermal      # checks Dataset_Thermal
    python preflight_check.py /abs/path    # checks any dataset directory
"""
import sys
from pathlib import Path
from collections import Counter

DETECTION = Path(__file__).parent.parent.resolve()
DATASET_FIXED = DETECTION / "Dataset_Fixed"
DATASET_THERMAL = DETECTION / "Dataset_Thermal"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def run_preflight(data_root: Path):
    print("=" * 50)
    print(f"  PREFLIGHT CHECK — {data_root.name}")
    print("=" * 50)
    errors = 0

    # 1. Image-label pair validation
    print("\n[1/3] Image-Label Pair Validation...")
    pair_errors = 0
    for split in ("train", "val"):
        imgs_dir = data_root / split / "images"
        lbls_dir = data_root / split / "labels"
        if not imgs_dir.exists():
            continue
        img_stems = {p.stem for p in imgs_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
        lbl_stems = {p.stem for p in lbls_dir.glob("*.txt") if p.name != "classes.txt"}

        missing_lbl = img_stems - lbl_stems
        orphan_lbl = lbl_stems - img_stems
        if missing_lbl:
            print(f"  [!] {split}: {len(missing_lbl)} images missing labels")
            pair_errors += len(missing_lbl)
        if orphan_lbl:
            print(f"  [!] {split}: {len(orphan_lbl)} orphan label files")
            pair_errors += len(orphan_lbl)
    if pair_errors == 0:
        print("  All image-label pairs matched.")
    else:
        errors += pair_errors

    # 2. Bounds & class validity
    print("\n[2/3] Coordinate Bounds & Class Validity...")
    total_lines = 0
    bound_fails = 0
    counts: Counter = Counter()
    for f in data_root.rglob("*/labels/*.txt"):
        if f.name == "classes.txt":
            continue
        try:
            lines = f.read_text().splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if not parts:
                continue
            total_lines += 1
            if len(parts) >= 5:
                try:
                    cid = int(parts[0])
                    counts[cid] += 1
                    cx, cy, w, h = map(float, parts[1:5])
                    if cid not in (0, 1, 2):
                        if bound_fails < 5:
                            print(f"  [!] Invalid class {cid} in {f.name}:{i}")
                        bound_fails += 1
                    if any(x < 0.0 or x > 1.0 for x in (cx, cy, w, h)):
                        if bound_fails < 5:
                            print(f"  [!] OOB {parts} in {f.name}:{i}")
                        bound_fails += 1
                except ValueError:
                    bound_fails += 1
    if bound_fails:
        print(f"  {bound_fails} violations found!")
        errors += bound_fails
    else:
        print(f"  {total_lines:,} annotations verified. Zero violations.")

    # 3. Class distribution
    print("\n[3/3] Class Distribution...")
    total_obj = sum(counts.values())
    if total_obj == 0:
        print("  Zero objects detected!")
        errors += 1
    else:
        for cid, name in [(0, "drone"), (1, "bird"), (2, "helicopter")]:
            c = counts.get(cid, 0)
            print(f"  {name:12s} ({cid}): {c:>8,}  ({c/total_obj*100:5.1f}%)")
        minority = counts.get(1, 0) + counts.get(2, 0)
        ratio = minority / total_obj * 100
        if ratio < 10.0:
            print(f"  WARNING: minority ratio {ratio:.1f}% < 10% — class imbalance is high")

    print("\n" + "=" * 50)
    if errors == 0:
        print("STATUS: GO")
    else:
        print(f"STATUS: NO-GO ({errors} errors)")
    print("=" * 50)
    return errors == 0


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "thermal":
        target = DATASET_THERMAL
    elif arg and Path(arg).is_dir():
        target = Path(arg)
    else:
        target = DATASET_FIXED
    ok = run_preflight(target)
    sys.exit(0 if ok else 1)
