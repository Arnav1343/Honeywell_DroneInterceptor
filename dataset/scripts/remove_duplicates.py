#!/usr/bin/env python3
"""
remove_duplicates.py — Perceptual-hash deduplication.
Usage:
    python remove_duplicates.py              # deduplicates Dataset_Fixed/train
    python remove_duplicates.py thermal      # deduplicates Dataset_Thermal/train
    python remove_duplicates.py /abs/path    # deduplicates any images directory
"""
import sys
from pathlib import Path

DETECTION = Path(__file__).parent.parent.resolve()
DATASET_FIXED = DETECTION / "Dataset_Fixed"
DATASET_THERMAL = DETECTION / "Dataset_Thermal"


def remove_duplicates(images_dir: Path, labels_dir: Path):
    try:
        import imagehash
        from PIL import Image
        from tqdm import tqdm
    except ImportError:
        print("ERROR: requires imagehash, Pillow, tqdm — pip install imagehash pillow tqdm")
        return

    print(f"Deduplication sweep: {images_dir}")
    img_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    print(f"Found {len(img_files):,} images")

    hashes: dict = {}
    duplicates = []
    for img_path in tqdm(img_files, desc="Hashing"):
        try:
            h = imagehash.phash(Image.open(img_path))
            if h in hashes:
                duplicates.append(img_path)
            else:
                hashes[h] = img_path
        except Exception as e:
            print(f"  Error {img_path.name}: {e}")

    print(f"\nFound {len(duplicates):,} duplicates.")
    if not duplicates:
        print("Dataset is perceptually unique.")
        return

    print("Removing duplicates...")
    removed = 0
    for dup in tqdm(duplicates, desc="Deleting"):
        lbl = labels_dir / f"{dup.stem}.txt"
        try:
            dup.unlink()
            removed += 1
            if lbl.exists():
                lbl.unlink()
        except Exception as e:
            print(f"  Could not remove {dup.name}: {e}")
    print(f"Removed {removed:,} duplicate images (and matching labels).")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "thermal":
        imgs = DATASET_THERMAL / "train" / "images"
        lbls = DATASET_THERMAL / "train" / "labels"
    elif arg and Path(arg).is_dir():
        imgs = Path(arg)
        lbls = imgs.parent.parent / "labels"
    else:
        imgs = DATASET_FIXED / "train" / "images"
        lbls = DATASET_FIXED / "train" / "labels"
    remove_duplicates(imgs, lbls)
