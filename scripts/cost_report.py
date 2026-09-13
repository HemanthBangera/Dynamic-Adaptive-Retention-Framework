"""
API usage and cost of the study, computed from the LLM response cache (one record per distinct request).

Run managers only see their own calls, and replays report zero, so the cache is the complete record. Batch-API
answers are tagged ``"via": "batch"`` and billed at half the synchronous price.

    python scripts/cost_report.py --cache benchmark_runs/_llm_cache --out cost_keys.json
    python scripts/cost_report.py --merge laptop.json vm.json --out cost_report.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def scan(cache: Path) -> dict:
    out = {}
    for path in cache.rglob("*.json"):
        try:
            r = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        usage = r.get("usage") or {}
        out[path.stem] = {"model": r.get("model") or (r.get("request") or {}).get("model"),
                          "cost_usd": float(r.get("cost_usd") or 0.0), "via": r.get("via", "sync"),
                          "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                          "completion_tokens": int(usage.get("completion_tokens") or 0)}
    return out


def summarise(records: dict) -> dict:
    agg = defaultdict(lambda: {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd_list_price": 0.0,
                               "cost_usd_billed": 0.0})
    for r in records.values():
        key = f"{r['model']}|{r['via']}"
        a = agg[key]
        a["requests"] += 1
        a["prompt_tokens"] += r["prompt_tokens"]
        a["completion_tokens"] += r["completion_tokens"]
        a["cost_usd_list_price"] += r["cost_usd"]
        a["cost_usd_billed"] += r["cost_usd"] * (0.5 if r["via"] == "batch" else 1.0)
    total = {k: sum(a[k] for a in agg.values()) for k in ("requests", "prompt_tokens", "completion_tokens",
                                                          "cost_usd_list_price", "cost_usd_billed")}
    return {"by_model_and_channel": dict(agg), "total": total}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache")
    p.add_argument("--merge", nargs="+")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.cache:
        records = scan(Path(args.cache))
        Path(args.out).write_text(json.dumps(records), encoding="utf-8")
        print(f"{len(records)} records")
        return
    merged = {}
    for f in args.merge:
        merged.update(json.loads(Path(f).read_text(encoding="utf-8")))
    report = summarise(merged)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report["total"], indent=1))


if __name__ == "__main__":
    main()
