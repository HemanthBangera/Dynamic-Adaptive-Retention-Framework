"""Characterise differences between archived and replayed JSON/JSONL outputs.

For every leaf: non-numeric leaves must be equal; numeric leaves are compared by absolute and relative
difference. Prints, per file, the number of leaves compared, the number of differing numeric leaves, the
largest absolute and relative difference, and any non-numeric difference. Writes a JSON summary.

    python numeric_diff.py OUT.json LABEL ARCHIVED_FILE REPLAYED_FILE [LABEL A B ...]
"""
import json
import math
import re
import sys
from pathlib import Path

IGNORE = {"provenance", "created_unix", "created", "elapsed_s", "llm_usage"}


def load(path: Path):
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(x) for x in text.splitlines() if x.strip()]
    return json.loads(text)


def walk(a, b, path, stats):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in IGNORE:
                continue
            if k not in a or k not in b:
                stats["structural"].append(f"{path}.{k}: present on one side only")
                continue
            walk(a[k], b[k], f"{path}.{k}", stats)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            stats["structural"].append(f"{path}: length {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", stats)
        return
    stats["leaves"] += 1
    num = (int, float)
    if isinstance(a, num) and isinstance(b, num) and not isinstance(a, bool) and not isinstance(b, bool):
        if a != b:
            if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
                return
            d = abs(a - b)
            rel = d / max(abs(a), abs(b))
            stats["numeric_diffs"] += 1
            field = re.sub(r"\[\d+\]", "[]", path)
            f = stats["by_field"].setdefault(field, {"count": 0, "max_abs": 0.0})
            f["count"] += 1
            f["max_abs"] = max(f["max_abs"], d)
            stats["max_abs"] = max(stats["max_abs"], d)
            stats["max_rel"] = max(stats["max_rel"], rel)
        return
    if a != b:
        stats["non_numeric"].append(f"{path}: {str(a)[:60]!r} vs {str(b)[:60]!r}")


def main() -> None:
    out = Path(sys.argv[1])
    args = sys.argv[2:]
    report = {}
    for i in range(0, len(args), 3):
        label, fa, fb = args[i], Path(args[i + 1]), Path(args[i + 2])
        stats = {"leaves": 0, "numeric_diffs": 0, "max_abs": 0.0, "max_rel": 0.0, "non_numeric": [], "structural": [], "by_field": {}}
        walk(load(fa), load(fb), "", stats)
        stats["non_numeric_count"] = len(stats["non_numeric"])
        stats["non_numeric"] = stats["non_numeric"][:5]
        stats["structural_count"] = len(stats["structural"])
        stats["structural"] = stats["structural"][:5]
        report[label] = stats
        print(f"{label}: leaves={stats['leaves']} numeric_diffs={stats['numeric_diffs']} max_abs={stats['max_abs']:.3g} "
              f"max_rel={stats['max_rel']:.3g} non_numeric={stats['non_numeric_count']} structural={stats['structural_count']} "
              f"{stats['non_numeric'][:2]} {stats['structural'][:2]}")
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
