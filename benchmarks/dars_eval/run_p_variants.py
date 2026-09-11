"""
E4 — the predictive-relevance component P.

Static single-session protocol, as in E1:
- every memory shares one timestamp and no feedback has been applied;
- so R, F and U are identical across memories, and any DARS reranking comes
  from P alone.

This isolates P's effect on retrieval, which is what Reviewer 2 asked to quantify.

For each question, the ``FETCH_K`` nearest memories and their components come
from the vault itself (``MemoryVault.search_and_rerank`` with
``return_components=True``). P is then replaced by each variant, and the
candidates are re-fused with ``rankers.rerank_candidates``. That function
reproduces the vault's fusion exactly; this is checked at run time against the
vault for the as-submitted P. Slots beyond fetch_k follow similarity order, as
in the vault's two-stage retrieval.

P variants. P is the cosine between the memory vector and a goal vector,
clipped to [0, 1]:

none          P = 0 for every memory.
as_submitted  P as stored by the vault in the submitted configuration (the ALFWorld
              goal preset, unrelated to the benchmark domain).
task          Goal = a domain-matched description of the source (``datasets.TASK_GOALS``).
dynamic       Goal = mean embedding of the previous ``m`` queries on the same context.
              Uses query text only. The first question has no history, so P = 0.
oracle        Goal = mean embedding of the next ``m`` queries on the same context. This
              is foresight of the query distribution, i.e. the original specification
              P = E[sim(e_i, c_{t+k})]. The current query is excluded, and the last
              question has no future, so P = 0. Upper bound only.

For each variant and fusion mode, the script reports:
- evidence metrics at each budget (FactConsolidation: precedence);
- the fraction of the top-10 that differs from similarity;
- Kendall τ against the similarity order of the candidates;
- the spread of P.

Differences versus similarity use a paired clustered bootstrap. No LLM calls.

Usage
-----
python -m benchmarks.dars_eval.run_p_variants --out benchmark_runs/revision/E4/dev \
    --sources ruler_qa1_197K ruler_qa2_421K "longmemeval_s*" factconsolidation_sh_32k
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import kendalltau

from benchmarks.dars_eval.datasets import TASK_GOALS, family_of, load_contexts
from benchmarks.dars_eval.memory_units import MemoryUnit, WindowGuard
from benchmarks.dars_eval.provenance import collect_provenance
from benchmarks.dars_eval.rankers import MemoryIndex, Method, rank, rerank_candidates
from benchmarks.dars_eval.run_static import ENC, STATIC_TIME, default_methods, limit_for, question_metrics
from benchmarks.dars_eval.splits import question_splits
from benchmarks.dars_eval.stats import cluster_bootstrap_mean, paired_bootstrap_diff

logger = logging.getLogger(__name__)

VARIANTS = ("none", "as_submitted", "task", "dynamic", "oracle")
FUSION_METHODS = ("dars_rrf_k15", "dars_rrf_k50", "dars_wrrf_k50_b0.5", "dars_blend_a0.5", "dars_blend_a0.8")
DEFAULT_BUDGETS = (256, 512, 1024, 2048, 5120)
PAIRED_METRICS = ("group_recall", "precedence")
FETCH_K = 50
EQUIVALENCE_CHECKS_PER_CONTEXT = 2


def unit_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def goal_p(unit_vectors: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """P = clip(cos(memory, goal), 0, 1) for unit-normalised memory vectors."""
    g = np.asarray(goal, dtype=np.float64)
    return np.clip(unit_vectors @ (g / np.linalg.norm(g)), 0.0, 1.0)


def window_goal(query_units: np.ndarray, q: int, m: int, variant: str) -> Optional[np.ndarray]:
    """Mean of the previous (dynamic) or next (oracle) ``m`` query vectors; None when empty."""
    if variant == "dynamic":
        rows = query_units[max(0, q - m):q]
    elif variant == "oracle":
        rows = query_units[q + 1:q + 1 + m]
    else:
        raise ValueError(f"Not a windowed variant: {variant!r}")
    return rows.mean(axis=0) if len(rows) else None


def order_stats(order: Sequence[int], ranked: Sequence[int], sim_ranked: Sequence[int]) -> Dict[str, float]:
    """Top-10 change versus similarity and Kendall τ over the reranked candidates."""
    k = min(10, len(ranked))
    changed = 1.0 - len(set(ranked[:k]) & set(sim_ranked[:k])) / k
    f = len(order)
    if f < 2:
        return {"top10_changed": changed, "kendall_tau": 1.0}
    pos = np.empty(f)
    pos[np.asarray(order)] = np.arange(f)
    return {"top10_changed": changed, "kendall_tau": float(kendalltau(np.arange(f), pos)[0])}


def stored_predictive(index: MemoryIndex) -> np.ndarray:
    """The P value the vault holds for every unit (the as-submitted configuration)."""
    p = np.full(len(index.units), np.nan)
    for chunk, _ in index.vault.get_all_memories(limit=1024, scroll_yield=True):
        for point in chunk:
            comps = index.vault.compute_components(point.payload.to_dict(), current_time=STATIC_TIME)
            p[index.unit_of[point.point_id]] = comps["P"]
    if np.isnan(p).any():
        raise RuntimeError("Some units have no stored predictive value")
    return p


def run_source(source: str, args: argparse.Namespace, methods: List[Method], guard: WindowGuard,
               fh, results: Dict, pairs: Dict, spread: List[Dict[str, Any]]) -> None:
    budgets = sorted(int(b) for b in args.budgets)
    embedder = guard.embedder
    sim_method = Method("similarity", rank_mode="similarity", fetch_k=FETCH_K)
    task_goal = np.asarray(embedder.encode(TASK_GOALS[family_of(source)]))

    for ctx in load_contexts(source, guard):
        splits = question_splits(source, ctx.index, len(ctx.questions))
        selected = [q for q in range(len(ctx.questions)) if args.split == "all" or splits[q] == args.split]
        units = [MemoryUnit(u.text, None, dict(u.meta)) for u in ctx.units]
        vectors = np.asarray(embedder.encode_batch([u.text for u in units], batch_size=64), dtype=np.float32)
        index = MemoryIndex(units, f"e4_{source}_{ctx.index}", vectors=vectors, default_time=STATIC_TIME)
        V = unit_rows(vectors)
        fixed = {"none": np.zeros(len(units)), "as_submitted": stored_predictive(index),
                 "task": goal_p(V, task_goal)}
        q_raw = np.asarray(embedder.encode_batch([ctx.queries[q] for q in range(len(ctx.questions))]))
        Q = unit_rows(q_raw)
        unit_tokens = [len(ENC.encode(u.text)) for u in units]
        limit = limit_for(unit_tokens, budgets[-1])
        unit_of_serial = {u.meta["serial"]: i for i, u in enumerate(units) if "serial" in u.meta}
        cluster = f"{source}:{ctx.index}"
        window_p_std: Dict[str, List[float]] = defaultdict(list)
        fallbacks: Dict[str, int] = defaultdict(int)

        for j, q in enumerate(selected):
            qid = f"{cluster}:{q}"
            sim_ranked, sim_comps = rank(index, sim_method, ctx.queries[q], q_raw[q], limit=limit,
                                         current_time=STATIC_TIME)
            fk = min(FETCH_K, len(sim_ranked))
            cand = sim_ranked[:fk]
            comps = sim_comps[:fk]
            if any(c is None or "sim_rank" not in c for c in comps):
                raise RuntimeError("Vault did not return components for the fetch_k candidates")
            if [int(c["sim_rank"]) for c in comps] != list(range(1, fk + 1)):
                raise RuntimeError("Candidates are not in similarity-rank order")

            base = question_metrics(ctx, q, sim_ranked, unit_tokens, budgets, unit_of_serial)
            fh.write(json.dumps({"source": source, "context": ctx.index, "question": q, "split": splits[q],
                                 "variant": None, "method": "similarity", "budgets": base}) + "\n")
            for b, m in base.items():
                for key, v in m.items():
                    if v is not None and key != "n_units":
                        results["similarity"][f"{key}@{b}"].append((float(v), cluster))
                        pairs["similarity"][f"{key}@{b}"][qid] = float(v)

            for variant in VARIANTS:
                if variant in fixed:
                    P = fixed[variant]
                else:
                    goal = window_goal(Q, q, args.window, variant)
                    if goal is None:
                        fallbacks[variant] += 1
                        P = fixed["none"]
                    else:
                        P = goal_p(V, goal)
                    window_p_std[variant].append(float(P.std()))
                p_cand = P[cand]
                for method in methods:
                    f = min(method.fetch_k, fk)
                    sub = [{**comps[i], "P": float(p_cand[i])} for i in range(f)]
                    order = rerank_candidates(sub, method)
                    ranked = [cand[i] for i in order] + list(sim_ranked[f:])
                    if variant == "as_submitted" and j < EQUIVALENCE_CHECKS_PER_CONTEXT:
                        vault_ranked, _ = rank(index, method, ctx.queries[q], q_raw[q], limit=limit,
                                               current_time=STATIC_TIME)
                        if list(vault_ranked) != ranked[:len(vault_ranked)]:
                            raise RuntimeError(f"Offline rerank differs from the vault for {method.name} "
                                               f"({source} context {ctx.index}, question {q})")
                    per_budget = question_metrics(ctx, q, ranked, unit_tokens, budgets, unit_of_serial)
                    ost = order_stats(order, ranked, sim_ranked)
                    key = f"{variant}|{method.name}"
                    fh.write(json.dumps({"source": source, "context": ctx.index, "question": q,
                                         "split": splits[q], "variant": variant, "method": method.name,
                                         "budgets": per_budget, **ost,
                                         "p_std_candidates": float(p_cand[:f].std())}) + "\n")
                    for b, m in per_budget.items():
                        for mk, v in m.items():
                            if v is not None and mk != "n_units":
                                results[key][f"{mk}@{b}"].append((float(v), cluster))
                                pairs[key][f"{mk}@{b}"][qid] = float(v)
                    for sk, v in (*ost.items(), ("p_std_candidates", float(p_cand[:f].std()))):
                        results[key][sk].append((float(v), cluster))

        entry = {"source": source, "context": ctx.index, "units": len(units),
                 "questions": len(selected), "fallbacks": dict(fallbacks)}
        for variant, P in fixed.items():
            entry[variant] = {"store_mean": float(P.mean()), "store_std": float(P.std()),
                              "store_max": float(P.max())}
        for variant, stds in window_p_std.items():
            entry[variant] = {"store_std_mean_over_questions": float(np.mean(stds))}
        spread.append(entry)
        logger.info("E4: %s context %d done (%d questions, %d units)", source, ctx.index, len(selected), len(units))


def summarise(results: Dict, pairs: Dict, n_boot: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, metrics in results.items():
        out[name] = {"metrics": {}}
        for key, vals in metrics.items():
            out[name]["metrics"][key] = cluster_bootstrap_mean(
                [v for v, _ in vals], [c for _, c in vals], n_boot=n_boot).as_dict()
        if name == "similarity":
            continue
        out[name]["vs_similarity"] = {}
        for key, per_q in pairs[name].items():
            if key.split("@")[0] not in PAIRED_METRICS:
                continue
            base = pairs["similarity"].get(key, {})
            qids = [qid for qid in per_q if qid in base]
            if not qids:
                continue
            out[name]["vs_similarity"][key] = paired_bootstrap_diff(
                [per_q[qid] for qid in qids], [base[qid] for qid in qids],
                [qid.rsplit(":", 1)[0] for qid in qids], n_boot=n_boot)
    return out


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="E4 predictive-relevance variants (static protocol)")
    p.add_argument("--sources", nargs="+",
                   default=["ruler_qa1_197K", "ruler_qa2_421K", "longmemeval_s*", "factconsolidation_sh_32k",
                            "factconsolidation_mh_32k"])
    p.add_argument("--out", required=True)
    p.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    p.add_argument("--budgets", nargs="+", default=[str(b) for b in DEFAULT_BUDGETS])
    p.add_argument("--window", type=int, default=10, help="m queries for the dynamic / oracle goals")
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for noisy in ("httpx", "core.layer_d.storage", "benchmarks.memory_agent_bench.loader"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    methods = [m for m in default_methods() if m.name in FUSION_METHODS]
    guard = WindowGuard()
    t0 = time.time()
    summary: Dict[str, Any] = {}
    spread: List[Dict[str, Any]] = []
    with (out / "per_question.jsonl").open("w", encoding="utf-8") as fh:
        for source in args.sources:
            results: Dict = defaultdict(lambda: defaultdict(list))
            pairs: Dict = defaultdict(lambda: defaultdict(dict))
            run_source(source, args, methods, guard, fh, results, pairs, spread)
            summary[source] = summarise(results, pairs, args.n_boot)

    manifest = {
        "experiment": "E4_p_variants", "sources": args.sources, "split": args.split,
        "budgets": sorted(int(b) for b in args.budgets), "variants": VARIANTS, "window": args.window,
        "fetch_k": FETCH_K, "methods": [m.as_dict() for m in methods], "static_time": STATIC_TIME,
        "task_goals": TASK_GOALS, "p_spread": spread, "elapsed_s": time.time() - t0,
        "provenance": collect_provenance(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")

    for source, s in summary.items():
        print(f"== {source}")
        for name, v in s.items():
            if name == "similarity":
                continue
            diffs = v["vs_similarity"]
            key = next((k for k in (f"group_recall@1024", f"precedence@256") if k in diffs), None)
            d = diffs.get(key) if key else None
            ch = v["metrics"].get("top10_changed", {}).get("mean", float("nan"))
            print(f"  {name:34s} top10_changed={ch:.3f}"
                  + (f"  {key} diff={d['diff']:+.3f} [{d['ci_lo']:+.3f},{d['ci_hi']:+.3f}]" if d else ""))


if __name__ == "__main__":
    main()
