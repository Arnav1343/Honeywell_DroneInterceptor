#!/usr/bin/env python3
"""
convert_thermal.py — Pseudo-thermal conversion for Dataset_Fixed/train images.

Finds all images in Dataset_Fixed/train/images that do NOT already have a
pseudo_<stem>.jpg in Dataset_Thermal/train/images and applies an INFERNO
colormap to simulate thermal appearance.

Labels are copied unchanged (bounding boxes are identical).

Usage:
    python convert_thermal.py           # process all pending images
    python convert_thermal.py --dry-run # report counts without converting
"""
import sys
import shutil
from pathlib import Path

DETECTION = Path(__file__).parent.parent.resolve()
SRC_IMG_DIR = DETECTION / "Dataset_Fixed" / "train" / "images"
SRC_LBL_DIR = DETECTION / "Dataset_Fixed" / "train" / "labels"
DST_IMG_DIR = DETECTION / "Dataset_Thermal" / "train" / "images"
DST_LBL_DIR = DETECTION / "Dataset_Thermal" / "train" / "labels"

BATCH_SIZE = 1000


def to_thermal(img_path: Path, out_path: Path) -> bool:
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise RuntimeError("opencv-python required — pip install opencv-python")
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(out_path), thermal)
    return True


def main(dry_run: bool = False):
    DST_IMG_DIR.mkdir(parents=True, exist_ok=True)
    DST_LBL_DIR.mkdir(parents=True, exist_ok=True)

    # Build set of already-processed stems
    existing = {p.stem[len("pseudo_"):] for p in DST_IMG_DIR.glob("pseudo_*.jpg")}
    print(f"Already converted : {len(existing):,} pseudo_ images")

    # Find pending images
    all_src = [p for p in SRC_IMG_DIR.iterdir()
               if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
               and p.stem not in existing]
    total_pending = len(all_src)
    print(f"Pending           : {total_pending:,} images")

    if dry_run:
        print("(dry-run — no files written)")
        return

    if total_pending == 0:
        print("Nothing to do. All images already converted.")
        return

    # Import opencv once (raises cleanly if missing)
    try:
        import cv2  # noqa: F401
    except ImportError:
        print("ERROR: opencv-python not installed — pip install opencv-python")
        sys.exit(1)

    processed = skipped = failed = 0

    for batch_start in range(0, total_pending, BATCH_SIZE):
        batch = all_src[batch_start: batch_start + BATCH_SIZE]
        batch_end = min(batch_start + BATCH_SIZE, total_pending)
        print(f"  Processing {batch_start + 1:>6,}–{batch_end:>6,} / {total_pending:,}...",
              end="", flush=True)

        b_ok = b_skip = b_fail = 0
        for src_img in batch:
            dst_img = DST_IMG_DIR / f"pseudo_{src_img.stem}.jpg"
            dst_lbl = DST_LBL_DIR / f"pseudo_{src_img.stem}.txt"

            # Skip if already done (race-condition guard)
            if dst_img.exists():
                b_skip += 1
                continue

            src_lbl = SRC_LBL_DIR / f"{src_img.stem}.txt"
            if not src_lbl.exists():
                b_skip += 1
                continue

            if to_thermal(src_img, dst_img):
                shutil.copy2(src_lbl, dst_lbl)
                b_ok += 1
            else:
                b_fail += 1

        processed += b_ok
        skipped += b_skip
        failed += b_fail
        print(f"  done ({b_ok} ok, {b_skip} skip, {b_fail} fail)")

    print(f"\nConversion complete.")
    print(f"  Processed : {processed:,}")
    print(f"  Skipped   : {skipped:,}")
    print(f"  Failed    : {failed:,}")
    print(f"  Total pseudo_ : {len(existing) + processed:,}")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    main(dry_run=dry_run)
