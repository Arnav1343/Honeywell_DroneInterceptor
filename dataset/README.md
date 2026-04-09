# DP5 Drone Detection Dataset

> **Task:** Multi-class aerial object detection — Drone · Bird · Helicopter  
> **Format:** YOLOv8 normalized bounding box (`.txt`)  
> **Modalities:** RGB + Pseudo-Thermal  
> **Status:** Verified · Submission-Ready · OS-Portable

---

## Overview

| Property | Value |
|---|---|
| Total images (RGB) | 120,375 |
| Total images (Thermal) | 166,350 |
| **Combined total images** | **286,725** |
| Total annotations | 1,063,701 |
| Classes | `drone` (0) · `bird` (1) · `helicopter` (2) |
| Annotation format | YOLOv8 normalized `cx cy w h` |
| Coordinate violations | **Zero** |
| Duplicate images removed | 1,057 |
| Corrupted images removed | 1 |
| Sequence-aware split | Yes |
| Preflight status (RGB) | **GO** |
| Preflight status (Thermal) | **GO** |

---

## Dataset Statistics

### Class Distribution — RGB vs Thermal

![Class Distribution](charts/01_class_distribution.png)

| Class | RGB Count | RGB % | Thermal Count | Thermal % |
|---|---|---|---|---|
| Drone (0) | 442,749 | 85.8% | 365,443 | 66.8% |
| Bird (1) | 24,970 | 4.8% | 136,544 | 24.9% |
| Helicopter (2) | 48,558 | 9.4% | 45,438 | 8.3% |
| **Total** | **516,277** | | **547,425** | |

> The thermal dataset has improved bird/helicopter representation due to BAMBI bird injection (40,982 images) and the full helicopter corpus applied during pseudo-thermal conversion.

---

### Train / Validation Split

![Train Val Split](charts/02_train_val_split.png)

| Dataset | Train | Val | Val % |
|---|---|---|---|
| RGB (Dataset_Fixed) | 107,737 | 12,638 | 10.5% |
| Thermal (Dataset_Thermal) | 148,884 | 17,466 | 10.5% |

Splits are **sequence-aware** — no two frames from the same video clip appear in both train and val. Pseudo-thermal images are locked to train only (no temporal leakage risk from the RGB→thermal mapping).

---

### Source Breakdown

![Source Breakdown](charts/03_source_breakdown.png)

| Source | Prefix | Primary Class | Images |
|---|---|---|---|
| Maciullo Dataset | `mac_pos_`, `mac_` | Drone | 50,403 |
| ADH Roboflow | `adh_` | Drone / Helicopter | 16,782 |
| Helicopters-of-DC | `dchelis2_`, `dchelis_` | Helicopter | 14,670 |
| OpenImages v7 | `oi_` | Bird / Helicopter | 12,455 |
| BAMBI Birds | `bambi_` | Bird | 40,982 |
| VisDrone | `vd_` | Drone | 6,468 |
| Military Helicopter | `milheli_` | Helicopter | 6,768 |
| HIT-UAV IR | `hituav_` | Drone / Background | 2,906 |
| Civil Helicopter | `civheli_` | Helicopter | 3,590 |
| Zenodo Thermal | `zen2_` | Drone | 8,160 |
| Roboflow Delivery | `rf_` | Helicopter | 1,883 |
| AirSim Synthetic | `airsim_` | Drone | 1,799 |
| MAV-VID | `mavvid_` | Drone | 1,191 |
| Zenodo UAV | `zen_` | Drone | 890 |

---

### Thermal Dataset Composition

![Thermal Composition](charts/04_thermal_composition.png)

The thermal dataset blends four distinct sources:

| Source | Images | Description |
|---|---|---|
| Pseudo-thermal (RGB→IR) | 114,302 | OpenCV INFERNO colormap applied to all RGB train images |
| BAMBI Birds | 40,982 | Real bird footage — forced class 1 |
| Zenodo Thermal | 8,160 | Real drone thermal imagery (Zenodo record) |
| HIT-UAV IR | 2,906 | Real infrared UAV surveillance backgrounds |
| **Total** | **166,350** | |

> Pseudo-thermal images are **train-only** to prevent any RGB→thermal leakage into the validation set.

---

### Annotations by Split and Class

![Annotations by Split](charts/05_annotations_by_split.png)

---

## Folder Structure

```
dataset/
├── README.md                    ← this file
├── DATASET_CARD.md              ← machine-readable summary
├── configs/
│   ├── dataset_train.yaml       ← RGB training config
│   ├── dataset_thermal.yaml     ← Thermal training config
│   ├── dataset_combined.yaml    ← Combined RGB + Thermal config
│   └── dataset_dota_test.yaml   ← Zero-shot evaluation config
├── scripts/
│   ├── fast_audit.py            ← Multi-threaded class distribution audit
│   ├── preflight_check.py       ← Full GO/NO-GO validation gate
│   ├── verify_bounds.py         ← Coordinate bounds checker
│   ├── sequence_aware_split.py  ← Rebuild train/val splits (no leakage)
│   ├── convert_thermal.py       ← RGB → pseudo-thermal conversion
│   ├── audit_classes.py         ← Per-source class breakdown
│   ├── remove_duplicates.py     ← Perceptual hash deduplication
│   ├── spot_check_labels.py     ← Visual annotation spot-checker
│   └── generate_dataset_card.py ← Regenerate DATASET_CARD.md
└── charts/
    ├── 01_class_distribution.png
    ├── 02_train_val_split.png
    ├── 03_source_breakdown.png
    ├── 04_thermal_composition.png
    └── 05_annotations_by_split.png
```

The actual image data lives in `Dataset_Fixed/` and `Dataset_Thermal/` one level above this folder (not committed to git — see **Accessing the Images** below).

---

## YAML Configs

All configs use **relative paths** and work on Windows, Linux, and macOS without modification.

### RGB Training
```yaml
# configs/dataset_train.yaml
path: ./Dataset_Fixed
train: train/images
val:   val/images
nc: 3
names:
  0: drone
  1: bird
  2: helicopter
```

### Thermal Training
```yaml
# configs/dataset_thermal.yaml
path: ./Dataset_Thermal
train: train/images
val:   val/images
nc: 3
names:
  0: drone
  1: bird
  2: helicopter
```

### Combined RGB + Thermal
```yaml
# configs/dataset_combined.yaml
path: .
train:
  - Dataset_Fixed/train/images
  - Dataset_Thermal/train/images
val:
  - Dataset_Fixed/val/images
  - Dataset_Thermal/val/images
nc: 3
names:
  0: drone
  1: bird
  2: helicopter
```

---

## Scripts Reference

All scripts use `pathlib` throughout — zero hardcoded paths.

```bash
# Run from the Detection/ root (one level above dataset/)

# Full validation gate (returns GO or NO-GO)
python dataset/scripts/preflight_check.py
python dataset/scripts/preflight_check.py thermal

# Class distribution audit
python dataset/scripts/fast_audit.py
python dataset/scripts/fast_audit.py thermal

# Bounds verification
python dataset/scripts/verify_bounds.py
python dataset/scripts/verify_bounds.py thermal

# Rebuild train/val splits (sequence-aware, 90/10)
python dataset/scripts/sequence_aware_split.py
python dataset/scripts/sequence_aware_split.py thermal

# Complete/resume pseudo-thermal conversion
python dataset/scripts/convert_thermal.py --dry-run
python dataset/scripts/convert_thermal.py

# Regenerate DATASET_CARD.md
python dataset/scripts/generate_dataset_card.py

# Per-source class breakdown
python dataset/scripts/audit_classes.py
```

---

## Accessing the Images

The image data (31 GB) is not stored in this repository. It is available at:

| Location | Contents | Size |
|---|---|---|
| *(HuggingFace / Zenodo — link TBD)* | `Dataset_Fixed/` — RGB, 120K images | ~12 GB |
| *(HuggingFace / Zenodo — link TBD)* | `Dataset_Thermal/` — IR/pseudo-thermal, 166K images | ~19 GB |

To use this dataset locally, place `Dataset_Fixed/` and `Dataset_Thermal/` in the same directory as this `dataset/` folder so the relative paths in the YAML configs resolve correctly:

```
Detection/
├── Dataset_Fixed/
│   ├── train/images/   (107,737 files)
│   ├── train/labels/   (107,737 files)
│   ├── val/images/     (12,638 files)
│   └── val/labels/     (12,638 files)
├── Dataset_Thermal/
│   ├── train/images/   (148,884 files)
│   ├── train/labels/   (148,884 files)
│   ├── val/images/     (17,466 files)
│   └── val/labels/     (17,466 files)
└── dataset/            ← this repo
```

---

## Training Examples

```bash
# YOLOv8/v10 RGB baseline
yolo train model=yolov10n.pt data=dataset/configs/dataset_train.yaml \
     imgsz=640 epochs=100 batch=32

# Thermal model
yolo train model=yolov10n.pt data=dataset/configs/dataset_thermal.yaml \
     imgsz=640 epochs=100 batch=32

# Combined modality training
yolo train model=yolov10n.pt data=dataset/configs/dataset_combined.yaml \
     imgsz=640 epochs=150 batch=32

# Zero-shot evaluation on DOTA v1.0
yolo val model=runs/detect/train/weights/best.pt \
     data=dataset/configs/dataset_dota_test.yaml
```

---

## Verification

Run the full preflight gate before any training run:

```bash
python dataset/scripts/preflight_check.py         # RGB
python dataset/scripts/preflight_check.py thermal  # Thermal
```

Expected output:
```
STATUS: GO
```

Both datasets currently return **GO**:
- All 286,725 image-label pairs matched
- Zero coordinate violations across 1,063,701 annotations
- No invalid class IDs

---

## Citation / Acknowledgements

This dataset aggregates and re-annotates imagery from the following public sources:

- **Maciullo** — Drone detection sequences
- **VisDrone** — UAV surveillance benchmark
- **UAVDT** — Urban aerial tracking
- **MAV-VID** — Micro aerial vehicle video
- **HIT-UAV** — Infrared thermal UAV dataset
- **BAMBI** — Bird flight dataset (re-annotated as class 1)
- **Zenodo Thermal** — Thermal drone detection imagery
- **OpenImages v7** — Bird and aircraft imagery (Google)
- **Roboflow Universe** — ADH, Civil Helicopter, Military Helicopter, Helicopters-of-DC
- **AirSim** — Synthetic drone imagery (Microsoft Research)

> This dataset is assembled for research use within the Honeywell Drone Interceptor project. Please respect the individual licences of each upstream source when redistributing.
