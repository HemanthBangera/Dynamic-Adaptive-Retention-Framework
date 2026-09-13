"""
Make a publishable copy of the LLM response cache that does not redistribute benchmark text.

Every cached record keeps what a replay needs (the response text, finish reason, usage, model,
system fingerprint and cost) and the request parameters (model, temperature, seed, token limit).
Each message's content is replaced by its SHA-256, so the record still documents exactly which
prompt produced the answer: anyone who rebuilds the prompts from the original datasets (under
those datasets' own licences) gets the same hashes and the same cache keys.

Replays are unaffected, because the transport looks records up by the cache key (a hash of the
full request, computed from the rebuilt prompt) and reads only response fields.

    python scripts/redact_cache.py --src benchmark_runs/_llm_cache --dst archive/llm_cache_redacted
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def redact_record(record: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(record)
    req = dict(record.get("request") or {})
    if "messages" in req:
        req["messages"] = [
            {"role": m.get("role"),
             "content_sha256": hashlib.sha256(str(m.get("content", "")).encode("utf-8")).hexdigest(),
             "content_chars": len(str(m.get("content", "")))}
            for m in req["messages"]
        ]
    out["request"] = req
    out["redacted"] = True
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Redact prompt text from the LLM response cache")
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    args = p.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    n = 0
    for path in src.rglob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        target = dst / path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(redact_record(record), ensure_ascii=False), encoding="utf-8")
        n += 1
    print(f"redacted {n} records -> {dst}")


if __name__ == "__main__":
    main()
