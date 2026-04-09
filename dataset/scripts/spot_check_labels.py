#!/usr/bin/env python3
"""
spot_check_labels.py — Visual spot-check of YOLO annotations.
Requires a display. Press any key to advance, Q to quit.
Usage:
    python spot_check_labels.py              # spot-checks Dataset_Fixed/train
    python spot_check_labels.py thermal      # spot-checks Dataset_Thermal/train
    python spot_check_labels.py /abs/path    # spot-checks any images directory
"""
import sys
import random
from pathlib import Path

DETECTION = Path(__file__).parent.parent.resolve()
DATASET_FIXED = DETECTION / "Dataset_Fixed"
DATASET_THERMAL = DETECTION / "Dataset_Thermal"

CLASS_MAP = {0: "Drone", 1: "Bird", 2: "Helicopter"}
COLORS = {0: (0, 200, 255), 1: (0, 255, 100), 2: (255, 80, 0)}  # BGR


def spot_check(images_dir: Path, n: int = 20):
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("ERROR: requires opencv-python — pip install opencv-python")
        return

    labels_dir = images_dir.parent.parent / "labels"
    imgs = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    if not imgs:
        print(f"No images found in {images_dir}")
        return

    sample = random.sample(imgs, min(n, len(imgs)))
    print(f"Spot-checking {len(sample)} random images. Press any key to advance, Q to quit.")

    for img_path in sample:
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cid = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)
                    color = COLORS.get(cid, (200, 200, 200))
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, CLASS_MAP.get(cid, str(cid)), (x1, max(y1 - 4, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                except ValueError:
                    pass

        cv2.imshow(f"Spot Check — {img_path.name}", img)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "thermal":
        target = DATASET_THERMAL / "train" / "images"
    elif arg and Path(arg).is_dir():
        target = Path(arg)
    else:
        target = DATASET_FIXED / "train" / "images"
    spot_check(target)
