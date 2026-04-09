#!/usr/bin/env python3
"""
fast_audit.py — Multi-threaded class-distribution audit.
Usage:
    python fast_audit.py              # audits Dataset_Fixed
    python fast_audit.py thermal      # audits Dataset_Thermal
    python fast_audit.py /abs/path    # audits any dataset directory
"""
import sys
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

DETECTION = Path(__file__).parent.parent.resolve()
DATASET_FIXED = DETECTION / "Dataset_Fixed"
DATASET_THERMAL = DETECTION / "Dataset_Thermal"


def count_file(f):
    c = Counter()
    try:
        with open(f) as fh:
            for line in fh:
                parts = line.strip().split()
                if parts:
                    c[parts[0]] += 1
    except Exception:
        pass
    return c


def audit(dataset_dir: Path):
    files = [f for f in dataset_dir.rglob("labels/*.txt") if f.name != "classes.txt"]
    print(f"Scanning {len(files):,} label files in {dataset_dir.name}...")

    counts: Counter = Counter()
    with ThreadPoolExecutor(max_workers=16) as ex:
        for r in ex.map(count_file, files):
            counts.update(r)

    valid = {"0", "1", "2"}
    total = sum(counts[k] for k in valid)
    print(f"\n--- CLASS DISTRIBUTION ({dataset_dir.name}) ---")
    for cid, name in [("0", "drone"), ("1", "bird"), ("2", "helicopter")]:
        c = counts[cid]
        pct = (c / total * 100) if total else 0.0
        print(f"  {name:12s} (class {cid}): {c:>8,}  ({pct:5.1f}%)")
    bad = {k: v for k, v in counts.items() if k not in valid}
    print(f"\nTotal valid annotations : {total:,}")
    if bad:
        print(f"INVALID class IDs found : {bad}")
        return False
    print("No invalid class IDs.")
    return True


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "thermal":
        target = DATASET_THERMAL
    elif arg and Path(arg).is_dir():
        target = Path(arg)
    else:
        target = DATASET_FIXED
    ok = audit(target)
    sys.exit(0 if ok else 1)
