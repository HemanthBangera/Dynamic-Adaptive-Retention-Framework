"""
E2 — multi-session memory streams with the Layer B feedback loop and a virtual clock.

Protocols
---------
factconsolidation  facts arrive in serial order, ``--fact-step`` seconds apart (newer
                   serial = newer fact); questions are asked after the last fact.
longmemeval        sessions arrive at their real timestamps; each question is asked at
                   its own date, when only the sessions dated before it are stored.
eventqa            the book is stored at t0; the questions are asked in story order,
                   ``--question-step`` seconds apart.

After each question the memories shown to the reader receive one feedback signal
(Layer B, ``LearningEngine.apply_feedback``: utility, frequency, recency on the
virtual clock) from ``--feedback``:

none            no updates (ranking state never changes)
oracle          ground truth: evidence covered (LongMemEval / FactConsolidation) or the
                reader's answer correct (EventQA, substring exact match); one verdict
                for all shown memories, as in the Layer B design
oracle_noisy:E  the oracle verdict flipped with probability E (seeded)
oracle_unit     ground truth attributed per memory: each shown memory that is gold evidence for
                the question succeeds, every other shown memory fails (unlabelled: no update)
judge           the Layer B LLM judge (SuccessEvaluator, gpt-4.1-nano); NEUTRAL → no update
lexical         deterministic per-memory attribution: a memory succeeds when the token F1
                between the reader's answer and the memory is at least ``--lexical-tau``

Logged per question:
- shown memories, evidence metrics, reader output and verdict;
- the ranking dynamics:
  - component spread among the candidates;
  - overlap with the similarity ranking, and Kendall tau;
  - the per-component top-k influence, measured two ways:
    - ``neutral_changes_topk_<C>``: component C set to its candidate mean, weights
      fixed. This measures whether C's variation changes the ranking; it is 0 when
      C is constant.
    - ``loo_changes_topk_<C>``: C removed and the other weights renormalised. This is
      an ablation, not an influence measure.

Memory budget (H5, ``--memory-budget``): after every ingestion, if more than
``budget × (units in the context)`` memories are stored, the lowest-priority
memories are deleted by ``--eviction`` (dars = lowest DARS score now; lru = oldest
last access; lfu = fewest accesses; fifo = oldest creation; random).  Evidence of a
question that was deleted before the question is asked counts as a harmful deletion.
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
from scipy.stats import kendalltau

from benchmarks.dars_eval.datasets import Context, family_of, load_contexts, split_name_for
from benchmarks.dars_eval.memory_units import WindowGuard
from benchmarks.dars_eval.provenance import collect_provenance
from benchmarks.dars_eval.rankers import (
    COMPONENTS,
    MemoryIndex,
    Method,
    leave_one_out_weights,
    rank,
    rerank_candidates,
)
from benchmarks.dars_eval.retrieval_eval import MEMORY_HEADER_TOKENS, cut_to_budget, evidence_metrics, precedence
from benchmarks.dars_eval.splits import question_splits
from benchmarks.dars_eval.stats import cluster_bootstrap_mean
from third_party.memoryagentbench_eval.eval_other_utils import f1_score

logger = logging.getLogger(__name__)
ENC = tiktoken.encoding_for_model("gpt-4o-mini")
T0 = 1_700_000_000.0


def stream_methods() -> List[Method]:
    return [
        Method("similarity", rank_mode="similarity"),
        Method("bm25", kind="bm25"),
        Method("recency", kind="recency"),
        Method("random", kind="random"),
        Method("dars_rrf_k50", rank_mode="rrf", fetch_k=50),
        Method("dars_wrrf_k50_b0.5", rank_mode="wrrf", fetch_k=50, beta_dars=0.5),
        Method("dars_blend_a0.5", rank_mode="blend", fetch_k=50, alpha=0.5),
    ]


def extra_stream_methods() -> List[Method]:
    """Methods run only when named in ``--methods``: DARS over a BM25 first stage."""
    return [
        Method("bm25_dars_rrf_k50", rank_mode="rrf", fetch_k=50, first_stage="bm25"),
        Method("bm25_dars_wrrf_k50_b0.5", rank_mode="wrrf", fetch_k=50, beta_dars=0.5, first_stage="bm25"),
        Method("bm25_dars_blend_a0.5", rank_mode="blend", fetch_k=50, alpha=0.5, first_stage="bm25"),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
#  Event schedules
# ═══════════════════════════════════════════════════════════════════════════════


def schedule(ctx: Context, args: argparse.Namespace) -> Tuple[List[Tuple[float, List[int]]], List[float]]:
    """(ingestion batches as (time, unit indices)) and the time of every question."""
    fam = family_of(ctx.source)
    if fam == "longmemeval":
        by_time: Dict[float, List[int]] = defaultdict(list)
        for i, u in enumerate(ctx.units):
            by_time[float(u.timestamp)].append(i)
        batches = sorted(by_time.items())
        return batches, list(ctx.question_times)
    if fam == "factconsolidation":
        batches = [(float(u.timestamp), [i]) for i, u in enumerate(ctx.units)]
        t_end = max(t for t, _ in batches)
        return batches, [t_end + (q + 1) * args.question_step for q in range(len(ctx.questions))]
    # eventqa / ruler: everything stored at T0, questions one step apart
    return [(T0, list(range(len(ctx.units))))], [T0 + (q + 1) * args.question_step for q in range(len(ctx.questions))]


# ═══════════════════════════════════════════════════════════════════════════════
#  Feedback
# ═══════════════════════════════════════════════════════════════════════════════


def parse_feedback(spec: str) -> Tuple[str, float]:
    if spec.startswith("oracle_noisy:"):
        return "oracle_noisy", float(spec.split(":", 1)[1])
    if spec not in ("none", "oracle", "oracle_unit", "judge", "lexical"):
        raise ValueError(f"Unknown feedback source {spec!r}")
    return spec, 0.0


def oracle_verdict(fam: str, ev: Optional[Dict[str, float]], reader_metrics: Optional[Dict[str, float]]) -> Optional[bool]:
    if fam == "eventqa":
        return None if reader_metrics is None else bool(reader_metrics.get("substring_exact_match", 0.0) >= 1.0)
    if ev is None:
        return None
    return bool(ev["group_recall"] > 0.0)


SERIAL_RULE = ("Each fact in the knowledge pool is provided with a serial number at the beginning, and the newer "
               "fact has larger serial number. \n You need to solve the conflicts of facts in the knowledge pool by "
               "finding the newest fact with larger serial number. You need to answer a question based on this rule. ")


def strip_serial_rule(formatted_query: str) -> str:
    """MemoryAgentBench's FactConsolidation query without the instruction to resolve conflicts by serial number."""
    if SERIAL_RULE not in formatted_query:
        raise ValueError("the FactConsolidation serial-number rule was not found in the query template")
    return formatted_query.replace(SERIAL_RULE, "")


def lexical_verdicts(answer: str, memory_texts: Sequence[str], tau: float) -> List[bool]:
    return [f1_score(answer or "", t)[0] >= tau for t in memory_texts]


# ═══════════════════════════════════════════════════════════════════════════════
#  One context stream
# ═══════════════════════════════════════════════════════════════════════════════


def evict(index: MemoryIndex, capacity: int, policy: str, now: float, rng: np.random.Generator,
          weights: Optional[Sequence[float]] = None,
          importance: Optional[Dict[str, float]] = None) -> List[int]:
    """Delete memories until at most ``capacity`` remain; returns the evicted unit indices.

    ``dars`` eviction scores memories with ``weights`` (w_r, w_f, w_u, w_p) when given,
    otherwise with the vault's current weights (those of the retrieval method).
    """
    excess = len(index) - capacity
    if excess <= 0:
        return []
    vault = index.vault
    # Random keys are indexed by unit, not drawn in scroll order: the vault orders points
    # by their (random uuid4) ids, so per-row draws would change with every run.
    rand_keys = rng.random(len(index.units)) if policy == "random" else None
    rows = []
    ga_raw = []            # generative_agents: (recency term, importance), normalised over the store below
    for chunk, _ in vault.get_all_memories(limit=2048, scroll_yield=True):
        for p in chunk:
            pl = p.payload
            if policy == "memorybank":
                # Ebbinghaus retention exp(-t/S): t in days since the last recall, S = 1 + recalls
                days = max(now - pl.recency, 0.0) / 86400.0
                rows.append((float(np.exp(-days / (1.0 + float(pl.frequency)))), index.unit_of[p.point_id], p.point_id))
                continue
            if policy == "generative_agents":
                unit = index.unit_of[p.point_id]
                hours = max(now - pl.recency, 0.0) / 3600.0
                ga_raw.append((0.995 ** hours, float(importance[index.units[unit].text]), unit, p.point_id))
                continue
            if policy == "dars" and weights is not None:
                comps = vault.compute_components(pl.to_dict(), current_time=now)
                key = sum(w * comps[c] for w, c in zip(weights, COMPONENTS))
            elif policy == "dars":
                key = vault.compute_dars_score(pl.to_dict(), current_time=now)
            elif policy == "lru":
                key = pl.recency
            elif policy == "lfu":
                key = float(pl.frequency)
            elif policy == "fifo":
                key = pl.created_at
            elif policy == "random":
                key = float(rand_keys[index.unit_of[p.point_id]])
            else:
                raise ValueError(f"Unknown eviction policy {policy!r}")
            rows.append((key, index.unit_of[p.point_id], p.point_id))
    if ga_raw:
        rec = np.array([r[0] for r in ga_raw])
        imp = np.array([r[1] for r in ga_raw])
        norm = lambda v: np.zeros_like(v) if v.max() - v.min() <= 0 else (v - v.min()) / (v.max() - v.min())
        for key, (_, _, unit, pid) in zip(norm(rec) + norm(imp), ga_raw):
            rows.append((float(key), unit, pid))
    # Ties (e.g. memories stored at the same time) are broken by a fixed pseudo-random
    # order per unit, never by position in the context.
    from zlib import crc32

    rows.sort(key=lambda r: (r[0], crc32(f"evict:{r[1]}".encode("utf-8")), r[1]))
    victims = rows[:excess]
    vault.delete_memories_batch([pid for _, _, pid in victims])
    evicted = [u for _, u, _ in victims]
    for u in evicted:
        pid = index.point_ids.pop(u)
        index.unit_of.pop(pid, None)
    gone = set(evicted)
    index.ingested = [u for u in index.ingested if u not in gone]
    index._bm25 = None
    return evicted


def dynamics(comps: Sequence[Optional[Dict[str, float]]], method: Method, k: int) -> Dict[str, Any]:
    """Ranking dynamics among the fused candidates of one query."""
    cand = [c for c in comps if c is not None and "sim_rank" in c]
    if len(cand) < 2:
        return {}
    spread = {f"std_{c}": float(np.std([x[c] for x in cand])) for c in COMPONENTS}
    final = rerank_candidates(cand, method)
    sim_order = sorted(range(len(cand)), key=lambda i: cand[i]["sim_rank"])
    top_final, top_sim = set(final[:k]), set(sim_order[:k])
    pos = {i: r for r, i in enumerate(final)}
    tau = kendalltau([cand[i]["sim_rank"] for i in range(len(cand))], [pos[i] for i in range(len(cand))]).statistic
    influence = {}
    # Influence of a component's *variation*: set it to its candidate mean, keep the
    # weights.  A component that is constant across candidates has influence exactly 0.
    for c in COMPONENTS:
        values = [x[c] for x in cand]
        if max(values) == min(values):
            influence[f"neutral_changes_topk_{c}"] = 0.0
            continue
        mean_c = float(np.mean(values))
        alt = rerank_candidates([{**x, c: mean_c} for x in cand], method)
        influence[f"neutral_changes_topk_{c}"] = float(set(alt[:k]) != top_final)
    # Ablation: remove the component and renormalise the remaining weights.  In blend
    # mode this also rescales the DARS share against similarity, so a constant
    # component can still change the top-k; report it only as an ablation.
    for c, w in leave_one_out_weights(method.weights).items():
        alt = rerank_candidates(cand, method, weights=w)
        influence[f"loo_changes_topk_{c}"] = float(set(alt[:k]) != top_final)
    return {
        **spread,
        "topk_differs_from_similarity": float(top_final != top_sim),
        "kendall_tau_vs_similarity": float(tau) if tau == tau else 1.0,
        **influence,
    }


async def run_context(ctx: Context, method: Method, feedback: str, args: argparse.Namespace,
                      reader, judge, rows_out: List[Dict[str, Any]]) -> None:
    from core.layer_b.engine import LearningEngine

    fam = family_of(ctx.source)
    kind, eps = parse_feedback(feedback)
    splits = question_splits(ctx.source, ctx.index, len(ctx.questions))
    index = MemoryIndex(ctx.units, f"e2_{ctx.source}_{ctx.index}_{method.name}", ingest_all=False)
    engine = LearningEngine(vault=index.vault, evaluator=judge) if kind != "none" else None
    unit_tokens = [len(ENC.encode(u.text)) for u in ctx.units]
    unit_of_serial = {u.meta["serial"]: i for i, u in enumerate(ctx.units) if "serial" in u.meta}
    labels = ctx.extra.get("fc_labels")
    batches, q_times = schedule(ctx, args)
    order = sorted(range(len(ctx.questions)), key=lambda q: (q_times[q], q))
    qvecs = index.vault.embedder.encode_batch([ctx.queries[q] for q in range(len(ctx.questions))])
    rng = np.random.default_rng([args.seed, ctx.index, len(method.name)])
    b_ptr = 0
    capacity = int(np.ceil(args.memory_budget * len(ctx.units))) if args.memory_budget else None
    evicted_all: set = set()

    for step, q in enumerate(order):
        now = q_times[q]
        while b_ptr < len(batches) and batches[b_ptr][0] <= now:
            index.add(batches[b_ptr][1])
            if capacity is not None:
                evicted_all.update(evict(index, capacity, args.eviction, batches[b_ptr][0], rng,
                                         getattr(args, "eviction_weights", None),
                                         getattr(args, "importance", None)))
            b_ptr += 1
        limit = min(len(index), args.budget // (min(unit_tokens) + MEMORY_HEADER_TOKENS) + 1) if len(index) else 0
        labelled = bool(labels and labels[q] and labels[q]["gold_is_newest"])
        # Rank precedence needs the full ranking.  A deeper call returns the same prefix
        # (exact search, stable sorts), so ``shown`` is unchanged.
        ranked, comps = rank(index, method, ctx.queries[q], qvecs[q],
                             limit=len(index) if labelled else max(limit, 1), current_time=now)
        shown = cut_to_budget(ranked, unit_tokens, args.budget)
        ev = evidence_metrics(shown, ctx.evidence[q]) if ctx.evidence[q] else None
        rec: Dict[str, Any] = {
            "source": ctx.source, "context": ctx.index, "question": q, "split": splits[q], "step": step,
            "time": now, "method": method.name, "feedback": feedback, "stored": len(index),
            "shown": shown, "evidence": ev,
        }
        if capacity is not None and ctx.evidence[q]:
            groups = ctx.evidence[q]
            lost = sum(1 for g in groups if all(u in evicted_all for u in g))
            rec["evidence_groups_evicted"] = lost / len(groups)
        if labelled:
            lab = labels[q]
            gold = unit_of_serial[lab["gold_serial"]]
            rivals = [unit_of_serial[s] for s in lab["conflict_serials"] if s in unit_of_serial]
            rec["precedence"] = precedence(shown, gold, rivals)             # within the token budget
            rec["precedence_rank"] = precedence(ranked, gold, rivals)       # over the full ranking (H2)
        if method.kind == "dars":
            rec["dynamics"] = dynamics(comps, method, args.dynamics_k)

        reader_out = None
        if reader is not None:
            display = list(reversed(shown)) if getattr(args, "display_order", "best_first") == "best_last" else shown
            reader_out = await reader.answer([ctx.units[u].text for u in display], ctx.formatted_queries[q],
                                             ctx.answers[q], seed=args.seed)
            rec["reader"] = {k: reader_out[k] for k in ("output", "parsed_output", "metrics", "cached")}

        if engine is not None and shown:
            pids = [index.point_ids[u] for u in shown]
            if kind in ("oracle", "oracle_noisy"):
                verdict = oracle_verdict(fam, ev, reader_out["metrics"] if reader_out else None)
                if verdict is not None and kind == "oracle_noisy" and rng.random() < eps:
                    verdict = not verdict
                if verdict is not None:
                    await engine.apply_feedback(pids, success=verdict, current_time=now)
                rec["verdict"] = verdict
            elif kind == "oracle_unit":
                gold = {u for g in ctx.evidence[q] for u in g}
                if gold:
                    for pid, u in zip(pids, shown):
                        await engine.apply_feedback([pid], success=u in gold, current_time=now)
                    rec["verdict"] = float(np.mean([u in gold for u in shown]))
                else:
                    rec["verdict"] = None
            elif kind == "judge":
                if reader_out is None:
                    raise ValueError("--feedback judge needs --reader")
                memories = [{"id": p, "payload": {"text_content": ctx.units[u].text}} for p, u in zip(pids, shown)]
                rec["verdict"] = await engine.process_feedback_loop(
                    ctx.formatted_queries[q], reader_out["output"], memories, current_time=now)
            elif kind == "lexical":
                if reader_out is None:
                    raise ValueError("--feedback lexical needs --reader")
                verdicts = lexical_verdicts(reader_out["parsed_output"] or reader_out["output"],
                                            [ctx.units[u].text for u in shown], args.lexical_tau)
                for pid, v in zip(pids, verdicts):
                    await engine.apply_feedback([pid], success=v, current_time=now)
                rec["verdict"] = float(np.mean(verdicts))
        rows_out.append(rec)


# ═══════════════════════════════════════════════════════════════════════════════
#  Driver
# ═══════════════════════════════════════════════════════════════════════════════


def summarise(rows: List[Dict[str, Any]], n_boot: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["feedback"], r["split"])].append(r)
    for (m, fb, split), rs in sorted(groups.items()):
        key = f"{m}|{fb}|{split}"
        metrics: Dict[str, List[Tuple[float, int]]] = defaultdict(list)
        for r in rs:
            for k, v in (r.get("evidence") or {}).items():
                metrics[k].append((v, r["context"]))
            if r.get("precedence") is not None:
                metrics["precedence"].append((r["precedence"], r["context"]))
            if r.get("precedence_rank") is not None:
                metrics["precedence_rank"].append((r["precedence_rank"], r["context"]))
            if r.get("evidence_groups_evicted") is not None:
                metrics["harmful_deletion"].append((r["evidence_groups_evicted"], r["context"]))
            for k, v in (r.get("reader") or {}).get("metrics", {}).items():
                metrics[f"reader_{k}"].append((v, r["context"]))
            for k, v in (r.get("dynamics") or {}).items():
                metrics[f"dyn_{k}"].append((v, r["context"]))
        out[key] = {
            k: cluster_bootstrap_mean([v for v, _ in vs], [c for _, c in vs], n_boot=n_boot).as_dict()
            for k, vs in metrics.items()
        }
    return out


async def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    guard = WindowGuard()
    contexts = load_contexts(args.source, guard, fc_seconds_per_serial=args.fact_step,
                             fc_serial_prefix=not args.no_serial_prefix)
    if args.fc_prompt == "no_serial_rule":
        for ctx in contexts:
            ctx.formatted_queries = [strip_serial_rule(fq) for fq in ctx.formatted_queries]
    if args.contexts:
        keep = {int(c) for c in args.contexts.split(",")}
        contexts = [c for c in contexts if c.index in keep]
    methods = stream_methods()
    if args.methods:
        wanted = [m.strip() for m in args.methods.split(",") if m.strip()]
        known = {m.name: m for m in methods + extra_stream_methods()}
        unknown = [w for w in wanted if w not in known]
        if unknown:
            raise SystemExit(f"unknown methods: {unknown}")
        methods = [known[w] for w in wanted]

    from config.settings import DARSConfig
    DARSConfig.RECENCY_DECAY_LAMBDA = float(args.decay_lambda)
    args.importance = None
    if getattr(args, "importance_file", None):
        from benchmarks.dars_eval.importance import load_ratings
        args.importance = load_ratings(Path(args.importance_file))
    elif args.memory_budget and args.eviction == "generative_agents":
        raise SystemExit("--eviction generative_agents needs --importance-file")

    reader = judge = None
    transports = []
    if args.reader:
        from benchmarks.dars_eval.reader import MABReader
        from core.llm_transport import OpenAITransport

        t_reader = OpenAITransport(DARSConfig.OPENAI_READER_MODEL, max_concurrency=args.concurrency)
        reader = MABReader(t_reader, args.source, split_name_for(args.source))
        transports.append(t_reader)
    if any(parse_feedback(f)[0] == "judge" for f in args.feedback):
        from core.layer_b.evaluator import SuccessEvaluator
        from core.llm_transport import OpenAITransport

        t_judge = OpenAITransport(DARSConfig.OPENAI_AUX_MODEL, max_concurrency=args.concurrency)
        judge = SuccessEvaluator(transport=t_judge)
        transports.append(t_judge)

    rows: List[Dict[str, Any]] = []
    t_start = time.time()
    jobs = []
    for method in methods:
        # Pure similarity / BM25 / recency / random rankings ignore the learned state,
        # so feedback cannot change them: run them once.
        adaptive = method.kind == "dars" and method.rank_mode != "similarity"
        feedbacks = args.feedback if adaptive else ["none"]
        for fb in feedbacks:
            for ctx in contexts:
                jobs.append(run_context(ctx, method, fb, args, reader, judge, rows))
    sem = asyncio.Semaphore(args.parallel_streams)

    async def guarded(job):
        async with sem:
            await job

    await asyncio.gather(*(guarded(j) for j in jobs))

    # Streams process every question (state must evolve through all of them), but only
    # the reported split is written or summarised — test rows stay unseen until the
    # pre-registered test run.
    # Deterministic order: parallel streams finish in any order, and the summary's sums
    # and bootstrap cluster order must not depend on it (bit-identical replays).
    rows = sorted((r for r in rows if args.report_split == "all" or r["split"] == args.report_split),
                  key=lambda r: (r["method"], r["feedback"], r["context"], r["step"]))
    with (out_dir / "per_question.jsonl").open("w", encoding="utf-8") as fh:
        for r in sorted(rows, key=lambda r: (r["method"], r["feedback"], r["context"], r["step"])):
            fh.write(json.dumps(r) + "\n")
    summary = summarise(rows, args.n_boot)
    manifest = {
        "experiment": "E2_stream",
        "source": args.source,
        "report_split": args.report_split,
        "methods": [m.as_dict() for m in methods],
        "feedback": args.feedback,
        "budget": args.budget,
        "decay_lambda_per_hour": args.decay_lambda,
        "fact_step_s": args.fact_step,
        "question_step_s": args.question_step,
        "lexical_tau": args.lexical_tau,
        "serial_prefix": not args.no_serial_prefix,
        "fc_prompt": args.fc_prompt,
        "display_order": getattr(args, "display_order", "best_first"),
        "memory_budget_fraction": args.memory_budget,
        "eviction_policy": args.eviction if args.memory_budget else None,
        "eviction_weights": args.eviction_weights if args.memory_budget and args.eviction == "dars" else None,
        "seed": args.seed,
        "reader": reader.settings() if reader else None,
        "llm_usage": {t.model: t.ledger.as_dict() for t in transports},
        "elapsed_s": time.time() - t_start,
        "provenance": collect_provenance(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    return {"summary": summary, "manifest": manifest}


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="E2 multi-session memory streams")
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--methods", default="")
    p.add_argument("--contexts", default="", help="comma-separated context indices (default: all)")
    p.add_argument("--feedback", nargs="+", default=["none", "oracle"])
    p.add_argument("--budget", type=int, default=5120)
    p.add_argument("--decay-lambda", type=float, default=0.005, help="recency decay per hour")
    p.add_argument("--fact-step", type=float, default=3600.0, help="FactConsolidation: seconds between serials")
    p.add_argument("--question-step", type=float, default=3600.0, help="seconds between questions (EventQA/FC)")
    p.add_argument("--lexical-tau", type=float, default=0.5)
    p.add_argument("--dynamics-k", type=int, default=10)
    p.add_argument("--reader", action="store_true")
    p.add_argument("--memory-budget", type=float, default=0.0,
                   help="keep at most this fraction of the context's units (0 = unlimited)")
    p.add_argument("--eviction", choices=("dars", "lru", "lfu", "fifo", "random", "memorybank", "generative_agents"),
                   default="dars")
    p.add_argument("--importance-file", default=None,
                   help="importance ratings (benchmarks.dars_eval.importance output) for generative_agents eviction")
    p.add_argument("--eviction-weights", type=float, nargs=4, default=None,
                   help="weights (w_r w_f w_u w_p) for dars eviction (default: the retrieval method's)")
    p.add_argument("--report-split", choices=("dev", "test", "all"), default="dev",
                   help="split written and summarised (default dev; test only for the pre-registered run)")
    p.add_argument("--no-serial-prefix", action="store_true",
                   help="FactConsolidation: store bare facts, without the 'N. ' serial prefix")
    p.add_argument("--display-order", choices=("best_first", "best_last"), default="best_first",
                   help="order in which the shown memories reach the reader (best_last: highest-ranked next to the query)")
    p.add_argument("--fc-prompt", choices=("mab", "no_serial_rule"), default="mab",
                   help="FactConsolidation query: MemoryAgentBench's, or with its serial-number rule removed")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--parallel-streams", type=int, default=4)
    p.add_argument("--n-boot", type=int, default=5000)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
    for noisy in ("httpx", "core.layer_d.storage", "core.layer_b.engine", "benchmarks.memory_agent_bench.loader"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    result = asyncio.run(run(args))
    for key, metrics in result["summary"].items():
        gr = metrics.get("group_recall", {}).get("mean")
        pr = metrics.get("precedence", {}).get("mean")
        prr = metrics.get("precedence_rank", {}).get("mean")
        acc = metrics.get("reader_substring_exact_match", {}).get("mean")
        print(f"{key:45s} recall={gr} precedence={pr} precedence_rank={prr} reader_acc={acc}")


if __name__ == "__main__":
    main()
