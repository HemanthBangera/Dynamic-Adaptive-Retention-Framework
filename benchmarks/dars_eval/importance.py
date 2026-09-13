"""
LLM importance ratings for the Generative Agents retention baseline.

Each distinct memory text is rated once, at write time, with the importance prompt of Park et
al. (2023) on gpt-4.1-nano (temperature 0, seed 0). The rating does not depend on retrieval,
feedback or later sessions. Calls go through the cached transport, so they can be collected and
delivered by the Batch API like every other stage, and replayed exactly.

    python -m benchmarks.dars_eval.importance --facts <run>/facts.jsonl [--facts ...] --out <file.jsonl>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from benchmarks.dars_eval.provenance import collect_provenance
from benchmarks.dars_eval.retention_signals import ga_importance_prompt, parse_rating, parse_rating_verbose

logger = logging.getLogger(__name__)
MAX_TOKENS = 4
NEUTRAL_RATING = 5          # used by analyses for an unparseable reply; the count is always reported


def distinct_texts(paths: Iterable[Path]) -> List[str]:
    seen: Dict[str, None] = {}
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if line.strip():
                seen.setdefault(json.loads(line)["text"], None)
    return list(seen)


async def rate(texts: List[str], transport: Any, chunk: int = 2000, max_tokens: int = MAX_TOKENS) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for lo in range(0, len(texts), chunk):
        part = texts[lo:lo + chunk]
        recs = await asyncio.gather(*(transport.complete(ga_importance_prompt(t), max_tokens=max_tokens)
                                      for t in part))
        parse = parse_rating if max_tokens <= MAX_TOKENS else parse_rating_verbose
        for t, r in zip(part, recs):
            out.append({"text": t, "rating": parse(r.get("text")), "raw": r.get("text", ""),
                        "collected": bool(r.get("collected", False))})
        logger.info("importance: %d/%d", min(lo + chunk, len(texts)), len(texts))
    return out


def load_ratings(path: Path) -> Dict[str, int]:
    """text -> rating, with unparseable replies mapped to the neutral rating."""
    ratings = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            ratings[r["text"]] = r["rating"] if r["rating"] is not None else NEUTRAL_RATING
    return ratings


def main(argv: Optional[List[str]] = None) -> None:
    from config.settings import DARSConfig
    from core.llm_transport import OpenAITransport

    p = argparse.ArgumentParser(description="Generative Agents importance ratings (gpt-4.1-nano)")
    p.add_argument("--facts", action="append", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                   help="reply limit; 4 is the addendum setting (MSC), longer texts need more room to reach a number")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)

    texts = distinct_texts(Path(f) for f in args.facts)
    transport = OpenAITransport(DARSConfig.OPENAI_AUX_MODEL, max_tokens=args.max_tokens,
                                max_concurrency=args.concurrency)
    records = asyncio.run(rate(texts, transport, max_tokens=args.max_tokens))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    unparsed = sum(1 for r in records if r["rating"] is None and not r["collected"])
    manifest = {"experiment": "importance_ratings", "model": DARSConfig.OPENAI_AUX_MODEL, "texts": len(texts),
                "max_tokens": args.max_tokens,
                "unparseable": unparsed, "sources": args.facts, "llm_usage": transport.ledger.as_dict(),
                "provenance": collect_provenance()}
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    print(f"rated {len(texts)} texts ({unparsed} unparseable) -> {out}")


if __name__ == "__main__":
    main()
