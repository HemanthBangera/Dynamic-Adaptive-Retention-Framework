"""Compare a replayed output directory with the archived one.

Per-question and item files must be byte-identical. JSON summaries must be equal after removing keys that
record when or where a run happened (provenance, timings, API usage counters). Manifests are compared the
same way. Exit status 1 if anything differs or is missing.

    python compare_outputs.py ARCHIVED_DIR REPLAYED_DIR [--label NAME]
"""
import argparse
import json
import sys
from pathlib import Path

VOLATILE = {"provenance", "created_unix", "created", "elapsed_s", "runtime_s", "llm_usage", "timestamp",
            "started", "finished", "host", "wall_time_s", "latency_median_ms", "latency_ms"}


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k not in VOLATILE}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


def first_diff(a, b, path=""):
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}.{k}: missing on one side"
            d = first_diff(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_diff(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if a == b else f"{path}: {a!r} vs {b!r}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("archived")
    p.add_argument("replayed")
    p.add_argument("--label", default="")
    p.add_argument("--only", nargs="*", default=None, help="relative file names to compare (default: all)")
    p.add_argument("--ignore", nargs="*", default=[], help="extra keys to ignore (e.g. absolute input paths)")
    args = p.parse_args()
    VOLATILE.update(args.ignore)
    a_root, b_root = Path(args.archived), Path(args.replayed)
    files = sorted(f for f in a_root.rglob("*") if f.is_file() and f.suffix in (".json", ".jsonl"))
    if args.only:
        files = [a_root / f for f in args.only]
    bad = 0
    for fa in files:
        rel = fa.relative_to(a_root)
        fb = b_root / rel
        if not fb.exists():
            print(f"[{args.label}] MISSING {rel}")
            bad += 1
            continue
        if fa.suffix == ".jsonl":
            if fa.read_bytes() == fb.read_bytes():
                status = "identical bytes"
            else:
                la = [strip(json.loads(x)) for x in fa.read_text(encoding="utf-8").splitlines() if x.strip()]
                lb = [strip(json.loads(x)) for x in fb.read_text(encoding="utf-8").splitlines() if x.strip()]
                d = first_diff(la, lb)
                status = "equal (volatile fields aside)" if d is None else f"DIFFERS {d}"
        else:
            d = first_diff(strip(json.loads(fa.read_text(encoding="utf-8"))), strip(json.loads(fb.read_text(encoding="utf-8"))))
            status = "equal" if d is None else f"DIFFERS {d}"
        if status.startswith("DIFFERS"):
            bad += 1
        print(f"[{args.label}] {rel}: {status[:300]}")
    print(f"[{args.label}] {'PASS' if bad == 0 else 'FAIL'} ({len(files)} files, {bad} problems)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
