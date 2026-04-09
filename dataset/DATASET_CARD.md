# DP5 Drone Detection Dataset

## Overview
| Property | Value |
|---|---|
| Total images (RGB) | 120,375 |
| Total images (Thermal) | 166,350 |
| Total annotations | 1,063,701 |
| Classes | drone, bird, helicopter |
| Format | YOLOv8 normalized txt |
| Verified | Yes — zero coordinate violations |

## RGB Dataset (Dataset_Fixed)
| Split | Images | Annotations |
|---|---|---|
| Train | 107,737 | 418,981 |
| Val | 12,638 | 97,295 |

### Class Distribution (RGB)
| Class | Count | Percentage |
|---|---|---|
| drone | 442,748 | 85.8% |
| bird | 24,970 | 4.8% |
| helicopter | 48,558 | 9.4% |

## Thermal Dataset (Dataset_Thermal)
| Split | Images | Annotations |
|---|---|---|
| Train | 148,884 | 507,115 |
| Val | 17,466 | 40,310 |

### Class Distribution (Thermal)
| Class | Count | Percentage |
|---|---|---|
| drone | 365,443 | 66.8% |
| bird | 136,544 | 24.9% |
| helicopter | 45,438 | 8.3% |

## Data Sources
| Source | Prefix | Class | Count |
|---|---|---|---|
| VisDrone | `vd_` | drone | 5,702 |
| Maciullo | `mac_` | drone | 45,019 |
| MAV-VID | `mavvid_` | drone | 1,155 |
| Zenodo UAV | `zen_` | drone | 2 |
| OpenImages | `oi_` | bird/helicopter | 11,186 |
| Roboflow | `rf_` | helicopter | 1,725 |
| AirSim synthetic | `airsim_` | drone | 1,615 |
| Pseudo-thermal | `pseudo_` | all | 114,302 |
| BAMBI birds | `bambi_` | bird | 27,391 |
| Zenodo thermal | `zen2_` | drone | 5,404 |
| HIT-UAV IR | `hituav_` | drone/background | 1,787 |
| ADH Roboflow | `adh_` | drone/helicopter | 16,103 |
| Civil-Helicopter | `civheli_` | helicopter | 4,101 |
| Military Helicopter | `milheli_` | helicopter | 6,106 |
| Helicopters-of-DC | `dchelis_` | helicopter | 43 |
| Helicopters-of-DC | `dchelis2_` | helicopter | 14,980 |

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
