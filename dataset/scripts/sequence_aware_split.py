#!/usr/bin/env python3
"""
sequence_aware_split.py — Rebuild train/val split without sequence leakage.

Groups images by their video/sequence origin so no two frames from the same
clip appear in both train and val. Independent web-scraped images are treated
as single-image groups and distributed randomly.

Usage:
    python sequence_aware_split.py              # splits Dataset_Fixed  (90/10)
    python sequence_aware_split.py thermal      # splits Dataset_Thermal (90/10, pseudo_ train-only)
    python sequence_aware_split.py /abs/path    # splits any dataset directory

The script MOVES files on disk. It first pools all images from both train/ and
val/ into a single candidate set, then re-assigns them. This is safe to re-run.
"""
import sys
import random
import shutil
from pathlib import Path
from collections import defaultdict

DETECTION = Path(__file__).parent.parent.resolve()
DATASET_FIXED = DETECTION / "Dataset_Fixed"
DATASET_THERMAL = DETECTION / "Dataset_Thermal"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
RANDOM_SEED = 42

# Sources whose frames must stay together (video sequences)
# Map prefix → how many underscore-parts form the sequence ID
VIDEO_SEQUENCE_PREFIXES = {
    "uavdt": 2,    # uavdt_M0203_... → group = uavdt_M0203
    "mavvid": 2,   # mavvid_pic_0012_... → group = mavvid_pic
    "dvb": 2,      # dvb_seq01_... → group = dvb_seq01
    "dut": 2,      # dut_seq01_... → group = dut_seq01
    "yt": 2,       # yt_abc_... → group = yt_abc
    "vd": 2,       # vd_seq001_... → group = vd_seq001
}

# Sources whose images are train-only (never move to val)
TRAIN_ONLY_PREFIXES = {"pseudo"}


def get_group(stem: str) -> tuple[str, bool]:
    """Return (group_id, train_only).
    group_id: string identifying the sequence/source group.
    train_only: True means this image must stay in train.
    """
    prefix = stem.split("_")[0].lower()

    if prefix in TRAIN_ONLY_PREFIXES:
        return stem, True  # unique group, forced train

    if prefix in VIDEO_SEQUENCE_PREFIXES:
        n = VIDEO_SEQUENCE_PREFIXES[prefix]
        parts = stem.split("_")
        group = "_".join(parts[:n]) if len(parts) >= n else stem
        return group, False

    # Independent images — group by source prefix to allow balanced distribution
    # but still shard large sources so bin-packing stays granular
    import hashlib
    shard = int(hashlib.md5(stem.encode()).hexdigest()[:6], 16) % 500
    return f"{prefix}_{shard:03d}", False


def split_dataset(dataset_dir: Path, val_fraction: float = 0.10):
    train_img = dataset_dir / "train" / "images"
    train_lbl = dataset_dir / "train" / "labels"
    val_img = dataset_dir / "val" / "images"
    val_lbl = dataset_dir / "val" / "labels"

    for d in (train_img, train_lbl, val_img, val_lbl):
        d.mkdir(parents=True, exist_ok=True)

    # Pool all images from both splits
    all_images = []
    for d in (train_img, val_img):
        all_images.extend(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)

    total = len(all_images)
    print(f"\nDataset  : {dataset_dir.name}")
    print(f"Pooled   : {total:,} images (train + existing val)")

    # Group images
    groups: dict[str, list[Path]] = defaultdict(list)
    forced_train: set[str] = set()

    for img in all_images:
        gid, train_only = get_group(img.stem)
        groups[gid].append(img)
        if train_only:
            forced_train.add(gid)

    # Greedy bin-packing: fill val until we reach target fraction
    target_val = int(total * val_fraction)
    group_ids = [g for g in groups if g not in forced_train]
    random.seed(RANDOM_SEED)
    random.shuffle(group_ids)

    val_set: list[Path] = []
    train_set: list[Path] = []

    for gid in group_ids:
        imgs = groups[gid]
        if len(val_set) + len(imgs) <= target_val * 1.05:
            val_set.extend(imgs)
        else:
            train_set.extend(imgs)

    # Add forced-train groups
    for gid in forced_train:
        train_set.extend(groups[gid])

    print(f"Target   : val = {target_val:,} ({val_fraction*100:.0f}%)")
    print(f"Assigned : train = {len(train_set):,}, val = {len(val_set):,}  "
          f"(val = {len(val_set)/total*100:.1f}%)")
    print(f"Train-only (e.g. pseudo_): {sum(len(groups[g]) for g in forced_train):,} images")

    # Move files to correct locations
    moved = 0

    def move_pair(img_path: Path, dst_img_dir: Path, dst_lbl_dir: Path):
        nonlocal moved
        # Determine source label dir
        if img_path.parent == train_img:
            src_lbl = train_lbl / f"{img_path.stem}.txt"
        else:
            src_lbl = val_lbl / f"{img_path.stem}.txt"

        dst_img = dst_img_dir / img_path.name
        dst_lbl = dst_lbl_dir / f"{img_path.stem}.txt"

        if img_path != dst_img:
            shutil.move(str(img_path), str(dst_img))
            moved += 1
        if src_lbl.exists() and src_lbl != dst_lbl:
            shutil.move(str(src_lbl), str(dst_lbl))

    print("Moving files...", flush=True)
    for img in train_set:
        move_pair(img, train_img, train_lbl)
    for img in val_set:
        move_pair(img, val_img, val_lbl)

    # Final counts
    final_train = sum(1 for p in train_img.iterdir() if p.suffix.lower() in IMG_EXTS)
    final_val = sum(1 for p in val_img.iterdir() if p.suffix.lower() in IMG_EXTS)
    final_total = final_train + final_val
    print(f"\nFinal    : train = {final_train:,}  val = {final_val:,}  "
          f"({final_val/final_total*100:.1f}% val)")
    print(f"Files moved: {moved:,}")
    return final_train, final_val


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "thermal":
        split_dataset(DATASET_THERMAL)
    elif arg and Path(arg).is_dir():
        split_dataset(Path(arg))
    else:
        split_dataset(DATASET_FIXED)
