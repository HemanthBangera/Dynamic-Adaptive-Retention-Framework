"""
E8 — efficiency at equal token budgets, including index-time compression.

Ranking is as in E1: static protocol, original embeddings. The retrieved context
is then filled up to a token budget B with the memories' text in one of three forms:

none        the original memory text;
extractive  deterministic sentence selection, keeping half the tokens;
llmlingua2  LLMLingua-2 at rate 0.5.

Compression therefore lets more memories fit in the same budget. This is the
setting of HLLC (Masood et al., Sci. Rep. 2026). Retrieval still uses the original
vectors (shadow indexing), as DARS Layer C does; BM25 scores the original text.

Metrics per method × compression × budget:
- evidence group recall (a gold-evidence memory is shown);
- answer presence (a normalised gold answer occurs in the shown text);
- memories shown, and context tokens used;
- at the reader budget: MemoryAgentBench reader accuracy.

Latency: wall-clock time of the ranking call per query (median). The machine is
shared with other jobs, so latency is indicative only.

Compression is lazy: a memory is compressed only when a budget fill reaches it.
Results are cached on disk (``benchmark_runs/_compress_cache``), so the test run
reuses them.

Usage
-----
python -m benchmarks.dars_eval.run_efficiency --out benchmark_runs/revision/E8/dev
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from benchmarks.dars_eval.datasets import load_contexts, split_name_for
from benchmarks.dars_eval.memory_units import MemoryUnit, WindowGuard
from benchmarks.dars_eval.provenance import PROJECT_ROOT, collect_provenance
from benchmarks.dars_eval.rankers import MemoryIndex, rank
from benchmarks.dars_eval.retrieval_eval import MEMORY_HEADER_TOKENS, evidence_metrics
from benchmarks.dars_eval.run_layerc import FIXED_RATE, ENC, LinguaCompressor, answer_retained, extractive_compress
from benchmarks.dars_eval.run_static import STATIC_TIME, default_methods
from benchmarks.dars_eval.splits import question_splits
from benchmarks.dars_eval.stats import cluster_bootstrap_mean, paired_bootstrap_diff

logger = logging.getLogger(__name__)

CACHE_ROOT = PROJECT_ROOT / "benchmark_runs" / "_compress_cache"
MIN_MEMORY_TOKENS = 8
COMPRESSIONS = ("none", "extractive", "llmlingua2")


def fill_budget(ranked: Sequence[int], text_of: Callable[[int], str], budget: int,
                overhead: int = MEMORY_HEADER_TOKENS) -> Tuple[List[int], int]:
    """Longest prefix of ``ranked`` whose formatted texts fit ``budget`` (as ``cut_to_budget``)."""
    shown: List[int] = []
    used = 0
    for u in ranked:
        cost = len(ENC.encode(text_of(u))) + overhead
        if used + cost > budget:
            break
        shown.append(u)
        used += cost
    return shown, used


class CompressionCache:
    """Disk-backed map text → compressed text for one compressor configuration."""

    def __init__(self, name: str, fn: Callable[[str], str], root: Path = CACHE_ROOT):
        self.name = name
        self.fn = fn
        self.path = Path(root) / f"{name}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data: Dict[str, str] = {}
        self.misses = 0
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.data[rec["key"]] = rec["text"]

    def __call__(self, text: str) -> str:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        hit = self.data.get(key)
        if hit is not None:
            return hit
        out = self.fn(text)
        self.misses += 1
        self.data[key] = out
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": key, "text": out}, ensure_ascii=False) + "\n")
        return out


async def run(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    guard = WindowGuard()
    emb = guard.embedder
    budgets = sorted(int(b) for b in args.budgets)
    lingua_budgets = {int(b) for b in args.lingua_budgets}
    by_name = {m.name: m for m in default_methods()}
    methods = [by_name[n.strip()] for n in args.methods.split(",") if n.strip()]
    reader_methods = {n.strip() for n in args.reader_methods.split(",") if n.strip()}
    compressors: Dict[str, Callable[[str], str]] = {}
    if "extractive" in args.compressions:
        compressors["extractive"] = CompressionCache(
            f"extractive_r{FIXED_RATE}", lambda t: extractive_compress(t, emb, FIXED_RATE))
    if "llmlingua2" in args.compressions:
        lingua = LinguaCompressor()
        compressors["llmlingua2"] = CompressionCache(
            f"llmlingua2_r{FIXED_RATE}", lambda t: lingua(t, rate=FIXED_RATE))

    reader_t = None
    if reader_methods and args.reader_budget:
        from config.settings import DARSConfig
        from core.llm_transport import OpenAITransport

        reader_t = OpenAITransport(DARSConfig.OPENAI_READER_MODEL, max_concurrency=args.concurrency)

    rows: List[Dict[str, Any]] = []
    for source in args.sources:
        reader = None
        if reader_t is not None:
            from benchmarks.dars_eval.reader import MABReader

            reader = MABReader(reader_t, source, split_name_for(source))
        for ctx in load_contexts(source, guard):
            splits = question_splits(source, ctx.index, len(ctx.questions))
            selected = [q for q in range(len(ctx.questions))
                        if (args.split == "all" or splits[q] == args.split) and ctx.evidence[q]]
            units = [MemoryUnit(u.text, None, dict(u.meta)) for u in ctx.units]
            vectors = np.asarray(emb.encode_batch([u.text for u in units], batch_size=64), dtype=np.float32)
            index = MemoryIndex(units, f"e8_{source}_{ctx.index}", vectors=vectors, default_time=STATIC_TIME)
            limit = min(len(units), budgets[-1] // (MIN_MEMORY_TOKENS + MEMORY_HEADER_TOKENS) + 1)
            qvecs = emb.encode_batch([ctx.queries[q] for q in selected])
            pending = []
            for q, qv in zip(selected, qvecs):
                answers = ctx.answers[q] if isinstance(ctx.answers[q], list) else [ctx.answers[q]]
                for method in methods:
                    t0 = time.perf_counter()
                    ranked, _ = rank(index, method, ctx.queries[q], qv, limit=limit, current_time=STATIC_TIME)
                    latency = time.perf_counter() - t0
                    for comp in args.compressions:
                        text_of = (lambda u: units[u].text) if comp == "none" else \
                            (lambda u, f=compressors[comp]: f(units[u].text))
                        for b in budgets:
                            if comp == "llmlingua2" and b not in lingua_budgets:
                                continue
                            shown, used = fill_budget(ranked, text_of, b)
                            texts = [text_of(u) for u in shown]
                            ev = evidence_metrics(shown, ctx.evidence[q]) or {}
                            row = {"source": source, "context": ctx.index, "question": q, "split": splits[q],
                                   "method": method.name, "compression": comp, "budget": b,
                                   "n_shown": len(shown), "tokens_used": used,
                                   "group_recall": ev.get("group_recall"),
                                   "answer_present": float(answer_retained(" ".join(texts), answers)),
                                   "latency_s": latency}
                            rows.append(row)
                            if reader is not None and method.name in reader_methods and b == args.reader_budget:
                                pending.append((row, texts, q))
            if pending:
                async def read(job):
                    row, texts, q = job
                    res = await reader.answer(texts, ctx.formatted_queries[q], ctx.answers[q], seed=0)
                    row["reader_substring_exact_match"] = float(res["metrics"].get("substring_exact_match", 0.0))
                    row["reader_f1"] = float(res["metrics"].get("f1", 0.0))
                    row["reader_output"] = res["output"]

                await asyncio.gather(*(read(j) for j in pending))
            logger.info("E8: %s context %d done (%d questions)", source, ctx.index, len(selected))

    with (out / "per_question.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    summary = summarise(rows, args.n_boot)
    manifest = {
        "experiment": "E8_efficiency", "sources": args.sources, "split": args.split, "budgets": budgets,
        "lingua_budgets": sorted(lingua_budgets), "compressions": args.compressions, "compression_rate": FIXED_RATE,
        "methods": [m.as_dict() for m in methods], "reader_methods": sorted(reader_methods),
        "reader_budget": args.reader_budget,
        "compression_cache": {k: {"path": str(c.path), "new_compressions": c.misses} for k, c in compressors.items()},
        "llm_usage": reader_t.ledger.as_dict() if reader_t else None,
        "latency_note": "ranking wall-clock on a shared CPU (i5-12500H); indicative only",
        "provenance": collect_provenance(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    for key, v in summary.items():
        gr, ap = v.get("group_recall"), v.get("answer_present")
        em = v.get("reader_substring_exact_match")
        print(f"{key:48s} shown={v['n_shown']['mean']:.1f} recall={gr['mean']:.3f} answer={ap['mean']:.3f}"
              + (f" EM={em['mean']:.3f}" if em else ""))


def summarise(rows: List[Dict[str, Any]], n_boot: int) -> Dict[str, Any]:
    groups: Dict[Tuple[str, str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["source"], r["method"], r["compression"], r["budget"])].append(r)
    out: Dict[str, Any] = {}
    metric_keys = ("n_shown", "tokens_used", "group_recall", "answer_present",
                   "reader_substring_exact_match", "reader_f1")
    for (source, method, comp, b), rs in sorted(groups.items()):
        rs.sort(key=lambda r: (r["context"], r["question"]))
        cl = [f"{r['source']}:{r['context']}" for r in rs]
        entry: Dict[str, Any] = {"n": len(rs), "latency_median_ms": float(np.median([r["latency_s"] for r in rs]) * 1000)}
        for k in metric_keys:
            vals = [(r[k], c) for r, c in zip(rs, cl) if r.get(k) is not None]
            if vals:
                entry[k] = cluster_bootstrap_mean([v for v, _ in vals], [c for _, c in vals], n_boot=n_boot).as_dict()
        if comp != "none":
            base = {(r["context"], r["question"]): r for r in groups.get((source, method, "none", b), [])}
            entry["vs_uncompressed"] = {}
            for k in ("group_recall", "answer_present", "reader_substring_exact_match"):
                pairs = [(r[k], base[(r["context"], r["question"])][k], c) for r, c in zip(rs, cl)
                         if r.get(k) is not None and (r["context"], r["question"]) in base
                         and base[(r["context"], r["question"])].get(k) is not None]
                if pairs:
                    entry["vs_uncompressed"][k] = paired_bootstrap_diff(
                        [a for a, _, _ in pairs], [b_ for _, b_, _ in pairs], [c for _, _, c in pairs], n_boot=n_boot)
        out[f"{source}|{method}|{comp}|{b}"] = entry
    return out


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="E8 efficiency at equal token budgets with index-time compression")
    p.add_argument("--sources", nargs="+", default=["ruler_qa1_197K", "ruler_qa2_421K", "longmemeval_s*"])
    p.add_argument("--out", required=True)
    p.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    p.add_argument("--methods", default="similarity,bm25,dars_wrrf_k50_b0.5")
    p.add_argument("--reader-methods", default="similarity,dars_wrrf_k50_b0.5")
    p.add_argument("--compressions", nargs="+", default=list(COMPRESSIONS), choices=COMPRESSIONS)
    p.add_argument("--budgets", nargs="+", default=["512", "1024", "2048"])
    p.add_argument("--lingua-budgets", nargs="+", default=["1024"],
                   help="budgets at which LLMLingua-2 is evaluated (CPU cost)")
    p.add_argument("--reader-budget", type=int, default=1024, help="0 disables the reader")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for noisy in ("httpx", "core.layer_d.storage", "benchmarks.memory_agent_bench.loader"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
