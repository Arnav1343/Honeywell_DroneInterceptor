#!/usr/bin/env python3
"""
audit_classes.py — Detailed per-source class breakdown.
Usage:
    python audit_classes.py              # audits Dataset_Fixed
    python audit_classes.py thermal      # audits Dataset_Thermal
    python audit_classes.py /abs/path    # audits any dataset directory
"""
import sys
from pathlib import Path
from collections import Counter, defaultdict

DETECTION = Path(__file__).parent.parent.resolve()
DATASET_FIXED = DETECTION / "Dataset_Fixed"
DATASET_THERMAL = DETECTION / "Dataset_Thermal"


def audit(dataset_dir: Path):
    files = [f for f in dataset_dir.rglob("labels/*.txt") if f.name != "classes.txt"]
    print(f"Auditing {len(files):,} label files in {dataset_dir.name}...\n")

    total = Counter()
    per_source: dict[str, Counter] = defaultdict(Counter)
    negatives = 0

    for f in files:
        # Derive source prefix (everything up to first digit run)
        stem = f.stem
        prefix = stem.split("_")[0] if "_" in stem else stem
        try:
            lines = f.read_text().splitlines()
        except Exception:
            continue
        if not any(l.strip() for l in lines):
            negatives += 1
            continue
        for line in lines:
            parts = line.strip().split()
            if parts and len(parts) >= 5:
                try:
                    cid = int(parts[0])
                    total[cid] += 1
                    per_source[prefix][cid] += 1
                except ValueError:
                    pass

    total_ann = sum(total.values())
    print(f"--- CLASS DISTRIBUTION ({dataset_dir.name}) ---")
    for cid, name in [(0, "drone"), (1, "bird"), (2, "helicopter")]:
        c = total[cid]
        pct = (c / total_ann * 100) if total_ann else 0
        print(f"  {name:12s} (class {cid}): {c:>8,}  ({pct:5.1f}%)")
    print(f"  {'backgrounds':12s}        : {negatives:>8,}")
    print(f"  {'TOTAL ann.':12s}        : {total_ann:>8,}")

    print(f"\n--- PER-SOURCE BREAKDOWN ---")
    for src in sorted(per_source.keys()):
        c = per_source[src]
        total_src = sum(c.values())
        print(f"  {src:15s}: drone={c[0]:6,}  bird={c[1]:6,}  heli={c[2]:6,}  total={total_src:6,}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "thermal":
        target = DATASET_THERMAL
    elif arg and Path(arg).is_dir():
        target = Path(arg)
    else:
        target = DATASET_FIXED
    audit(target)
