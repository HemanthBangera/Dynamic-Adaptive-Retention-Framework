"""
E1 — static single-session retrieval at equal token budgets.

Every memory is ingested with one common timestamp and queried at that time, so
recency, frequency and utility are identical across memories — the control
condition the reviewers describe.  Methods are compared on evidence coverage at
fixed token budgets; selected methods can also be read by the gpt-4o-mini reader.

Reader-only baselines (used only when named in ``--methods`` / ``--reader``):
no_memory     the reader gets an empty memory list (parametric-knowledge floor);
full_context  every memory unit in document order, one ``Memory i:`` block each,
              where the whole context fits the reader window.

Usage
-----
python -m benchmarks.dars_eval.run_static --source ruler_qa1_197K --split dev \
    --out benchmark_runs/revision/E1/ruler_qa1_197K_dev \
    [--reader similarity,bm25 --reader-budget 5120 --reader-seeds 0]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tiktoken

from benchmarks.dars_eval.datasets import TASK_GOALS, Context, family_of, load_contexts, split_name_for
from benchmarks.dars_eval.labels import label_coverage
from benchmarks.dars_eval.memory_units import MemoryUnit, WindowGuard
from benchmarks.dars_eval.provenance import collect_provenance
from benchmarks.dars_eval.rankers import MemoryIndex, Method, rank
from benchmarks.dars_eval.retrieval_eval import MEMORY_HEADER_TOKENS, cut_to_budget, evidence_metrics, precedence
from benchmarks.dars_eval.splits import question_splits
from benchmarks.dars_eval.stats import cluster_bootstrap_mean

logger = logging.getLogger(__name__)

STATIC_TIME = 1_700_000_000.0          # common ingestion and query time (unix seconds)
DEFAULT_BUDGETS = (1024, 2048, 5120, 8192, 16384)
FULL_CONTEXT_MAX_TOKENS = 120_000      # gpt-4o-mini window is 128k; leave room for the prompt template
ENC = tiktoken.encoding_for_model("gpt-4o-mini")


def default_methods() -> List[Method]:
    """E1 method grid (dev).  The configuration carried to test is chosen on dev and pre-registered."""
    methods = [
        Method("similarity", rank_mode="similarity"),
        Method("bm25", kind="bm25"),
        Method("recency", kind="recency"),
        Method("random", kind="random", seed=0),
        Method("dars_rrf_k15", rank_mode="rrf", fetch_k=15),
        Method("dars_rrf_k50", rank_mode="rrf", fetch_k=50),
    ]
    for fk in (20, 50, 100):
        for bd in (0.25, 0.5):
            methods.append(Method(f"dars_wrrf_k{fk}_b{bd}", rank_mode="wrrf", fetch_k=fk, beta_dars=bd))
    for a in (0.5, 0.8):
        methods.append(Method(f"dars_blend_a{a}", rank_mode="blend", fetch_k=50, alpha=a))
    return methods


def reader_only_methods() -> List[Method]:
    """Reader baselines that do not rank memories (not part of the retrieval grid)."""
    return [Method("no_memory", kind="none"), Method("full_context", kind="full")]


def select_methods(spec: str) -> List[Method]:
    """Methods named in ``spec`` (comma-separated), in canonical order; the E1 grid if empty."""
    if not spec:
        return default_methods()
    wanted = [m.strip() for m in spec.split(",") if m.strip()]
    pool = default_methods() + reader_only_methods()
    unknown = set(wanted) - {m.name for m in pool}
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    return [m for m in pool if m.name in wanted]


def predictive_values(ctx: Context, vectors: np.ndarray, goal: str, embedder) -> Optional[np.ndarray]:
    """P per unit: 'alfworld' = as submitted (None → the vault computes it), 'none' = 0, 'task' = domain goal."""
    if goal == "alfworld":
        return None
    if goal == "none":
        return np.zeros(len(ctx.units))
    if goal == "task":
        g = np.asarray(embedder.encode(TASK_GOALS[family_of(ctx.source)]), dtype=np.float32)
        v = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        return np.clip(v @ (g / np.linalg.norm(g)), 0.0, 1.0)
    raise ValueError(f"Unknown goal variant {goal!r}")


def limit_for(unit_tokens: Sequence[int], max_budget: int) -> int:
    """How many ranked units are needed to fill the largest budget."""
    cheapest = min(unit_tokens) + MEMORY_HEADER_TOKENS
    return min(len(unit_tokens), max_budget // max(cheapest, 1) + 1)


def question_metrics(ctx: Context, q: int, ranked: Sequence[int], unit_tokens: Sequence[int],
                     budgets: Sequence[int], unit_of_serial: Dict[int, int]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    labels = ctx.extra.get("fc_labels")
    for b in budgets:
        cut = cut_to_budget(ranked, unit_tokens, b)
        m: Dict[str, Any] = dict(evidence_metrics(cut, ctx.evidence[q]) or {})
        if labels and labels[q] and labels[q]["gold_is_newest"]:
            lab = labels[q]
            m["precedence"] = precedence(
                cut, unit_of_serial[lab["gold_serial"]],
                [unit_of_serial[s] for s in lab["conflict_serials"] if s in unit_of_serial],
            )
        m["n_units"] = len(cut)
        out[str(b)] = m
    return out


async def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    guard = WindowGuard()
    contexts = load_contexts(args.source, guard)
    methods = select_methods(args.methods)
    budgets = sorted(int(b) for b in args.budgets)

    reader = None
    reader_methods = {m.strip() for m in args.reader.split(",") if m.strip()} if args.reader else set()
    if reader_methods:
        from benchmarks.dars_eval.reader import MABReader
        from config.settings import DARSConfig
        from core.llm_transport import OpenAITransport

        transport = OpenAITransport(DARSConfig.OPENAI_READER_MODEL, max_concurrency=args.concurrency)
        reader = MABReader(transport, args.source, split_name_for(args.source))

    rows_path = out_dir / "per_question.jsonl"
    rows_path.write_text("", encoding="utf-8")
    # metric key → list of (value, cluster id)
    results: Dict[str, Dict[str, List[Tuple[float, int]]]] = defaultdict(lambda: defaultdict(list))
    coverage, unit_stats = [], []
    t_start = time.time()

    for ctx in contexts:
        splits = question_splits(args.source, ctx.index, len(ctx.questions))
        selected = [q for q in range(len(ctx.questions)) if args.split == "all" or splits[q] == args.split]
        coverage.append({"context": ctx.index, **label_coverage([ctx.evidence[q] for q in selected])})

        # Static protocol: every memory shares STATIC_TIME, whatever its natural timestamp.
        static_units = [MemoryUnit(u.text, None, dict(u.meta)) for u in ctx.units]
        embedder = WindowGuard().embedder
        vectors = np.asarray(embedder.encode_batch([u.text for u in static_units], batch_size=64), dtype=np.float32)
        index = MemoryIndex(
            static_units, f"e1_{args.source}_{ctx.index}",
            vectors=vectors,
            predictive=predictive_values(ctx, vectors, args.goal, embedder),
            default_time=STATIC_TIME,
        )
        unit_tokens = [len(ENC.encode(u.text)) for u in static_units]
        unit_stats.append({"context": ctx.index, "units": len(static_units),
                           "unit_tokens_median": float(np.median(unit_tokens)),
                           "context_tokens": int(sum(unit_tokens))})
        if any(m.kind == "full" for m in methods) and sum(unit_tokens) > FULL_CONTEXT_MAX_TOKENS:
            raise ValueError(f"full_context does not fit the reader window for {args.source} context "
                             f"{ctx.index}: {sum(unit_tokens)} > {FULL_CONTEXT_MAX_TOKENS} tokens")
        limit = limit_for(unit_tokens, budgets[-1])
        unit_of_serial = {u.meta["serial"]: i for i, u in enumerate(static_units) if "serial" in u.meta}
        qvecs = embedder.encode_batch([ctx.queries[q] for q in selected])

        pending: List[Tuple[Dict[str, Any], List[str], int]] = []
        with rows_path.open("a", encoding="utf-8") as fh:
            for method in methods:
                for q, qv in zip(selected, qvecs):
                    if method.kind in ("none", "full"):
                        shown = [] if method.kind == "none" else list(range(len(static_units)))
                        record = {
                            "source": args.source, "context": ctx.index, "question": q, "split": splits[q],
                            "method": method.name, "budgets": {}, "ranked": shown,
                            "context_tokens": int(sum(unit_tokens[u] for u in shown)),
                            "reader_budget_label": method.kind,
                        }
                        if method.name in reader_methods:
                            pending.append((record, [static_units[u].text for u in shown], q))
                        else:
                            fh.write(json.dumps(record) + "\n")
                        continue
                    ranked, comps = rank(index, method, ctx.queries[q], qv, limit=limit, current_time=STATIC_TIME)
                    per_budget = question_metrics(ctx, q, ranked, unit_tokens, budgets, unit_of_serial)
                    record: Dict[str, Any] = {
                        "source": args.source, "context": ctx.index, "question": q, "split": splits[q],
                        "method": method.name, "budgets": per_budget,
                        "ranked": ranked[: per_budget[str(budgets[-1])]["n_units"]],
                    }
                    if comps and comps[0] is not None:
                        record["top_components"] = comps[:10]
                    for b, m in per_budget.items():
                        for k, v in m.items():
                            if v is not None:
                                results[method.name][f"{k}@{b}"].append((float(v), ctx.index))
                    if method.name in reader_methods:
                        cut = cut_to_budget(ranked, unit_tokens, args.reader_budget)
                        pending.append((record, [static_units[u].text for u in cut], q))
                    else:
                        fh.write(json.dumps(record) + "\n")

        if reader is not None and pending:
            async def read(job: Tuple[Dict[str, Any], List[str], int]) -> Dict[str, Any]:
                rec, texts, q = job
                budget = rec.get("reader_budget_label", args.reader_budget)
                seeds = args.reader_seeds[:1] if budget == "full" and args.full_context_seed0 else args.reader_seeds
                rec["reader"] = []
                for seed in seeds:
                    res = await reader.answer(texts, ctx.formatted_queries[q], ctx.answers[q], seed=seed)
                    rec["reader"].append({"seed": seed, "budget": budget, **res})
                return rec

            done = await asyncio.gather(*(read(job) for job in pending))
            with rows_path.open("a", encoding="utf-8") as fh:
                for rec in done:
                    fh.write(json.dumps(rec) + "\n")
                    for r in rec["reader"]:
                        for k, v in r["metrics"].items():
                            results[rec["method"]][f"reader_{k}@{r['budget']}#s{r['seed']}"].append(
                                (float(v), rec["context"]))

    summary: Dict[str, Dict[str, Any]] = {}
    for name, metrics in results.items():
        summary[name] = {}
        for key, pairs in metrics.items():
            values = [v for v, _ in pairs]
            clusters = [c for _, c in pairs]
            summary[name][key] = cluster_bootstrap_mean(values, clusters, n_boot=args.n_boot).as_dict()

    manifest = {
        "experiment": "E1_static",
        "source": args.source,
        "split": args.split,
        "budgets": budgets,
        "goal_variant": args.goal,
        "static_time": STATIC_TIME,
        "methods": [m.as_dict() for m in methods],
        "reader_methods": sorted(reader_methods),
        "reader_budget": args.reader_budget,
        "reader_seeds": args.reader_seeds,
        "full_context_seed0_only": bool(args.full_context_seed0),
        "reader": reader.settings() if reader else None,
        "llm_usage": reader.transport.ledger.as_dict() if reader else None,
        "label_coverage": coverage,
        "unit_stats": unit_stats,
        "retrieval_query": "raw question field (EventQA: instructions removed); tail kept if longer than the embedder window",
        "memory_header_tokens": MEMORY_HEADER_TOKENS,
        "fusion": "two-stage: fetch_k nearest memories reranked, further slots in similarity order",
        "elapsed_s": time.time() - t_start,
        "provenance": collect_provenance(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    return {"summary": summary, "manifest": manifest}


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="E1 static retrieval at equal token budgets")
    p.add_argument("--source", required=True)
    p.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    p.add_argument("--out", required=True)
    p.add_argument("--methods", default="", help="comma-separated subset of method names")
    p.add_argument("--budgets", nargs="+", default=[str(b) for b in DEFAULT_BUDGETS])
    p.add_argument("--goal", choices=("alfworld", "none", "task"), default="alfworld")
    p.add_argument("--reader", default="", help="comma-separated methods to also read with gpt-4o-mini")
    p.add_argument("--reader-budget", type=int, default=5120)
    p.add_argument("--reader-seeds", type=int, nargs="+", default=[0])
    p.add_argument("--full-context-seed0", action="store_true",
                   help="read the full_context baseline with the first reader seed only (TPM-bound)")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--n-boot", type=int, default=5000)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
    for noisy in ("httpx", "core.layer_d.storage", "benchmarks.memory_agent_bench.loader"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    result = asyncio.run(run(args))
    key = f"group_recall@{args.reader_budget}"
    rows = sorted(result["summary"].items(), key=lambda kv: -(kv[1].get(key, {}).get("mean") or -1.0))
    for name, metrics in rows:
        est = metrics.get(key)
        if est:
            print(f"{name:28s} {key} = {est['mean']:.3f} [{est['ci_lo']:.3f}, {est['ci_hi']:.3f}]")


if __name__ == "__main__":
    main()
