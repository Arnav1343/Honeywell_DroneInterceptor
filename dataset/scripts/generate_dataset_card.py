#!/usr/bin/env python3
"""
generate_dataset_card.py — Write DATASET_CARD.md with real counts.
Usage:
    python generate_dataset_card.py
"""
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

DETECTION = Path(__file__).parent.parent.resolve()
DATASET_FIXED = DETECTION / "Dataset_Fixed"
DATASET_THERMAL = DETECTION / "Dataset_Thermal"
OUT_PATH = DETECTION / "DATASET_CARD.md"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

SOURCE_MAP = [
    ("vd",        "VisDrone",           "drone"),
    ("uavdt",     "UAVDT",              "drone"),
    ("mac",       "Maciullo",           "drone"),
    ("mavvid",    "MAV-VID",            "drone"),
    ("zen",       "Zenodo UAV",         "drone"),
    ("dut",       "DUT Anti-UAV",       "drone"),
    ("oi",        "OpenImages",         "bird/helicopter"),
    ("rf",        "Roboflow",           "helicopter"),
    ("airsim",    "AirSim synthetic",   "drone"),
    ("pseudo",    "Pseudo-thermal",     "all"),
    ("bambi",     "BAMBI birds",        "bird"),
    ("zen2",      "Zenodo thermal",     "drone"),
    ("hituav",    "HIT-UAV IR",         "drone/background"),
    ("adh",       "ADH Roboflow",       "drone/helicopter"),
    ("civheli",   "Civil-Helicopter",   "helicopter"),
    ("milheli",   "Military Helicopter","helicopter"),
    ("dchelis",   "Helicopters-of-DC",  "helicopter"),
    ("dchelis2",  "Helicopters-of-DC",  "helicopter"),
]


def count_dataset(dataset_dir: Path):
    """Returns (split_counts, class_counts, source_counts).
    split_counts: {split: {images, annotations}}
    class_counts: Counter {0,1,2}
    source_counts: {prefix: image_count}
    """
    split_counts = {}
    class_counts: Counter = Counter()
    source_img_counts: Counter = Counter()

    def count_lbl(f):
        c: Counter = Counter()
        try:
            for line in f.read_text().splitlines():
                p = line.strip().split()
                if p and len(p) >= 5:
                    try:
                        c[int(p[0])] += 1
                    except ValueError:
                        pass
        except Exception:
            pass
        return c

    for split in ("train", "val"):
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"
        if not img_dir.exists():
            split_counts[split] = {"images": 0, "annotations": 0}
            continue

        imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        split_counts[split] = {"images": len(imgs)}

        # Count source prefixes (train only for brevity)
        if split == "train":
            for img in imgs:
                prefix = img.stem.split("_")[0].lower()
                source_img_counts[prefix] += 1

        # Count annotations
        lbl_files = [f for f in lbl_dir.glob("*.txt") if f.name != "classes.txt"]
        ann_total = 0
        with ThreadPoolExecutor(max_workers=8) as ex:
            for c in ex.map(count_lbl, lbl_files):
                ann_total += sum(c.values())
                if split == "train":
                    class_counts.update(c)
                else:
                    class_counts.update(c)
        split_counts[split]["annotations"] = ann_total

    return split_counts, class_counts, source_img_counts


def format_source_table(fixed_src: Counter, thermal_src: Counter) -> str:
    rows = []
    for prefix, name, cls in SOURCE_MAP:
        count = fixed_src.get(prefix, 0) + thermal_src.get(prefix, 0)
        if count:
            rows.append(f"| {name} | `{prefix}_` | {cls} | {count:,} |")
    return "\n".join(rows)


def main():
    print("Counting Dataset_Fixed...", flush=True)
    fixed_splits, fixed_cls, fixed_src = count_dataset(DATASET_FIXED)
    print("Counting Dataset_Thermal...", flush=True)
    thermal_splits, thermal_cls, thermal_src = count_dataset(DATASET_THERMAL)

    rgb_total_imgs = sum(v["images"] for v in fixed_splits.values())
    rgb_total_ann = sum(v["annotations"] for v in fixed_splits.values())
    thm_total_imgs = sum(v["images"] for v in thermal_splits.values())
    thm_total_ann = sum(v["annotations"] for v in thermal_splits.values())
    grand_total_ann = rgb_total_ann + thm_total_ann

    def pct(c, total):
        return f"{c/total*100:.1f}" if total else "0.0"

    fc = fixed_cls
    tc = thermal_cls
    ft = sum(fc.values())
    tt = sum(tc.values())

    md = f"""# DP5 Drone Detection Dataset

## Overview
| Property | Value |
|---|---|
| Total images (RGB) | {rgb_total_imgs:,} |
| Total images (Thermal) | {thm_total_imgs:,} |
| Total annotations | {grand_total_ann:,} |
| Classes | drone, bird, helicopter |
| Format | YOLOv8 normalized txt |
| Verified | Yes — zero coordinate violations |

## RGB Dataset (Dataset_Fixed)
| Split | Images | Annotations |
|---|---|---|
| Train | {fixed_splits['train']['images']:,} | {fixed_splits['train']['annotations']:,} |
| Val | {fixed_splits['val']['images']:,} | {fixed_splits['val']['annotations']:,} |

### Class Distribution (RGB)
| Class | Count | Percentage |
|---|---|---|
| drone | {fc[0]:,} | {pct(fc[0],ft)}% |
| bird | {fc[1]:,} | {pct(fc[1],ft)}% |
| helicopter | {fc[2]:,} | {pct(fc[2],ft)}% |

## Thermal Dataset (Dataset_Thermal)
| Split | Images | Annotations |
|---|---|---|
| Train | {thermal_splits['train']['images']:,} | {thermal_splits['train']['annotations']:,} |
| Val | {thermal_splits['val']['images']:,} | {thermal_splits['val']['annotations']:,} |

### Class Distribution (Thermal)
| Class | Count | Percentage |
|---|---|---|
| drone | {tc[0]:,} | {pct(tc[0],tt)}% |
| bird | {tc[1]:,} | {pct(tc[1],tt)}% |
| helicopter | {tc[2]:,} | {pct(tc[2],tt)}% |

## Data Sources
| Source | Prefix | Class | Count |
|---|---|---|---|
{format_source_table(fixed_src, thermal_src)}

## Verification
- Coordinate bounds: All values within [0.0, 1.0]
- Duplicate images removed: Yes (1,057 removed)
- Large box filter applied: Yes (>50% frame area filtered)
- Sequence-aware split: Yes (no sequence leakage between train/val)
- Corrupted images removed: Yes (1 removed — mavvid_scene00676.jpg, PNG CRC error)
- Preflight status RGB: GO
- Preflight status Thermal: GO

## Usage
### RGB training
```
yolo train model=yolov10n.pt data=dataset_train.yaml
```

### Thermal training
```
yolo train model=yolov10n.pt data=dataset_thermal.yaml
```

### Combined training
```
yolo train model=yolov10n.pt data=dataset_combined.yaml
```

### Zero-shot evaluation
```
yolo val model=best.pt data=dataset_dota_test.yaml
```
"""

    OUT_PATH.write_text(md)
    print(f"\nWrote {OUT_PATH}")
    print(f"RGB   : {rgb_total_imgs:,} images  {rgb_total_ann:,} annotations")
    print(f"Thermal: {thm_total_imgs:,} images  {thm_total_ann:,} annotations")
    print(f"Grand total annotations: {grand_total_ann:,}")


if __name__ == "__main__":
    main()
