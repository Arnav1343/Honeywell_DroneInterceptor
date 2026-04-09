#!/usr/bin/env python3
"""
verify_bounds.py — Check all YOLO labels for out-of-range coordinates.
Usage:
    python verify_bounds.py              # checks Dataset_Fixed
    python verify_bounds.py thermal      # checks Dataset_Thermal
    python verify_bounds.py /abs/path    # checks any dataset directory
"""
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

DETECTION = Path(__file__).parent.parent.resolve()
DATASET_FIXED = DETECTION / "Dataset_Fixed"
DATASET_THERMAL = DETECTION / "Dataset_Thermal"


def check_file(f):
    issues = []
    try:
        with open(f) as fh:
            for i, line in enumerate(fh, 1):
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) < 5:
                    issues.append(f"line {i}: too few fields — {line.strip()!r}")
                    continue
                try:
                    cid = int(parts[0])
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                except ValueError:
                    issues.append(f"line {i}: non-numeric — {line.strip()!r}")
                    continue
                if cid not in (0, 1, 2):
                    issues.append(f"line {i}: invalid class {cid}")
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    issues.append(f"line {i}: OOB {parts[1:]}")
    except Exception as e:
        issues.append(str(e))
    return f, issues


def verify(dataset_dir: Path):
    files = [f for f in dataset_dir.rglob("labels/*.txt") if f.name != "classes.txt"]
    print(f"Checking {len(files):,} label files in {dataset_dir.name}...")

    total_issues = 0
    bad_files = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        for path, issues in ex.map(check_file, files):
            if issues:
                total_issues += len(issues)
                bad_files.append((path, issues))

    if bad_files:
        print(f"\nOut-of-bounds / malformed: {total_issues} annotations in {len(bad_files)} files")
        for p, issues in bad_files[:20]:
            print(f"  {p.relative_to(dataset_dir)}: {issues[0]}")
        if len(bad_files) > 20:
            print(f"  ... and {len(bad_files) - 20} more")
        return False
    else:
        print(f"All bounding boxes valid. Zero violations.")
        return True


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "thermal":
        target = DATASET_THERMAL
    elif arg and Path(arg).is_dir():
        target = Path(arg)
    else:
        target = DATASET_FIXED
    ok = verify(target)
    sys.exit(0 if ok else 1)
