"""
E10 — ALFWorld task stream: does learned utility rank the correct object location,
and does DARS eviction keep the memories later tasks need?  (H3, H5.)

Protocol
--------
Memories per task come from ``data/groupB/extractor.py`` (seeded): location
instances ("X is located in Y") and a location concept ("X objects are typically
found in Y receptacles") for the target object, tool instance/concept memories,
five other location facts from the task's initial state, a strategy template for
the task type and a goal memory.  Concepts and strategies are stored once
(identical text is not stored again); instances and goals are stored per task.

The ALFWorld train split is shuffled (seeded) into one interleaved task stream,
``--task-step`` seconds apart on a virtual clock.  A seeded 10 % of train tasks is
excluded from the stream and serves as the development evaluation set; the
in-distribution and out-of-distribution evaluation splits are the test sets.

For each stream task:
1. retrieval: the goal description and a location query for the target object
   ("Where can I find a <Object>?", as an agent must first find the object) each
   retrieve the top-``k`` stored memories by similarity (the same state for every
   scoring policy);
2. environment feedback on the retrieved memories — the walkthrough shows where
   the target object actually was and which tool the task needed:
     * location memory about the target object → success iff its receptacle is
       the true source receptacle;
     * tool memory for the task's action → success iff it names the tool used;
     * strategy memory → success iff it is the task's type;
   every retrieved memory also counts as accessed (frequency, recency);
3. the task's own memories are stored.

Held-out evaluation (frozen final state):
* H3 — among the location concept memories about the task's target object
  (candidates tied in form, differing only in the receptacle), where is the true
  receptacle ranked for the location query?  Candidate features (R, F, U, P,
  timestamps, similarity, count prior) are saved, so every method, weight vector
  and decay rate is scored offline (``analyze`` / ``tune``).
* H5 — for memory budgets, which policy keeps the memories the held-out task
  needs (true location concept, the needed tool concept, the strategy)?
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from benchmarks.dars_eval.provenance import collect_provenance
from benchmarks.dars_eval.rankers import COMPONENTS, Method, leave_one_out_weights, rerank_candidates
from benchmarks.dars_eval.splits import SPLIT_SEED
from benchmarks.dars_eval.stats import cluster_bootstrap_mean, paired_bootstrap_diff
from core.layer_d.schema import DARSWeights
from core.layer_d.storage import MemoryVault
from data.groupB.extractor import ProcessedTask, _parse_walkthrough, extract_task

logger = logging.getLogger(__name__)

T0 = 1_700_000_000.0
DEFAULT_WEIGHTS = (0.30, 0.20, 0.30, 0.20)
DEV_FRACTION = 0.10

LOC_INSTANCE = re.compile(r"^(\w+) is located in (\w+)$")
LOC_CONCEPT = re.compile(r"^(\w+) objects are typically found in (\w+) receptacles$")
TOOL_INSTANCE = re.compile(r"^(\w+) is used to (\w+) objects$")
TOOL_CONCEPT = re.compile(r"^(\w+) receptacles are commonly used for (\w+) operations$")


# ═══════════════════════════════════════════════════════════════════════════════
#  Data
# ═══════════════════════════════════════════════════════════════════════════════


def load_splits() -> Dict[str, List[Dict[str, Any]]]:
    from datasets import load_dataset

    return {
        name: [dict(r) for r in load_dataset("awawa-agi/alfworld-raw", split=name)]
        for name in ("train", "eval_in_distribution", "eval_out_of_distribution")
    }


def describe(text: str, mem_type: str, task: ProcessedTask) -> Dict[str, Any]:
    """Structured fields of one extracted memory (templates are fixed)."""
    info: Dict[str, Any] = {"text": text, "mem_type": mem_type, "task_type": task.task_type}
    for pattern, kind in ((LOC_CONCEPT, "location_concept"), (LOC_INSTANCE, "location_instance")):
        m = pattern.match(text)
        if m:
            info.update(kind=kind, obj=m.group(1).lower(), recep=m.group(2).lower())
            return info
    for pattern, kind in ((TOOL_CONCEPT, "tool_concept"), (TOOL_INSTANCE, "tool_instance")):
        m = pattern.match(text)
        if m:
            info.update(kind=kind, tool=m.group(1).lower(), action=m.group(2).lower())
            return info
    info["kind"] = "strategy" if mem_type == "strategy" else ("goal" if mem_type == "goal" else "other")
    return info


def task_truth(task: ProcessedTask) -> Dict[str, Optional[str]]:
    wt = _parse_walkthrough(task.walkthrough)
    return {
        "obj": (wt["target_obj"] or "").lower() or None,
        "source": (wt["source"] or "").lower() or None,
        "tool": (wt["tool"] or "").lower() or None,
        "action": (wt["action_type"] or "").lower() or None,
        "task_type": task.task_type,
    }


def location_query(obj: str) -> str:
    return f"Where can I find a {obj.capitalize()}?"


def adjudicate(mem: Dict[str, Any], truth: Dict[str, Optional[str]]) -> Optional[bool]:
    """Environment verdict for one retrieved memory, or None if the task says nothing about it."""
    kind = mem["kind"]
    if kind in ("location_concept", "location_instance") and truth["obj"] and mem["obj"] == truth["obj"]:
        return mem["recep"] == truth["source"]
    if kind in ("tool_concept", "tool_instance") and truth["action"] and mem["action"] == truth["action"]:
        return mem["tool"] == truth["tool"]
    if kind == "strategy":
        return mem["task_type"] == truth["task_type"]
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Stream
# ═══════════════════════════════════════════════════════════════════════════════


class Stream:
    def __init__(self, args: argparse.Namespace, goal_vector: np.ndarray):
        self.vault = MemoryVault(collection_name="alfworld_stream", location=":memory:",
                                 weights=DARSWeights(*DEFAULT_WEIGHTS))
        self.vault.initialize_collection(recreate=True)
        self.emb = self.vault.embedder
        self.goal = goal_vector / np.linalg.norm(goal_vector)
        self.args = args
        self.meta: Dict[str, Dict[str, Any]] = {}       # pid → structured fields + vector
        self.by_text: Dict[str, str] = {}               # deduplicated memories
        self.location_counts: Counter = Counter()       # (obj, recep) → tasks with that true source
        self.stats = Counter()

    def ingest(self, task: ProcessedTask, when: float) -> None:
        records, infos = [], []
        for m in task.memories:
            info = describe(m["text"], m["mem_type"], task)
            if info["kind"] in ("location_concept", "tool_concept", "strategy") and m["text"] in self.by_text:
                continue
            records.append(m["text"])
            infos.append(info)
        if not records:
            return
        vecs = np.asarray(self.emb.encode_batch(records), dtype=np.float32)
        p_vals = np.clip((vecs / np.linalg.norm(vecs, axis=1, keepdims=True)) @ self.goal, 0.0, 1.0)
        pids = self.vault.store_memories_batch([
            {"text": t, "vector": v.tolist(), "predictive_value": float(p), "timestamp": when,
             "tags": [f"task:{task.task_id}"]}
            for t, v, p in zip(records, vecs, p_vals)
        ])
        for pid, info, vec in zip(pids, infos, vecs):
            info["vector"] = vec
            info["created"] = when
            self.meta[pid] = info
            if info["kind"] in ("location_concept", "tool_concept", "strategy"):
                self.by_text[info["text"]] = pid

    def step(self, task: ProcessedTask, when: float) -> None:
        truth = task_truth(task)
        if self.meta:
            queries = [task.goal_description]
            if truth["obj"]:
                queries.append(location_query(truth["obj"]))
            seen: Dict[str, Any] = {}
            for query, qv in zip(queries, self.emb.encode_batch(queries)):
                for m in self.vault.search_and_rerank(query, fetch_k=self.args.k, top_n=self.args.k,
                                                      rank_mode="similarity", current_time=when, query_vector=qv):
                    seen.setdefault(m.point_id, m)
            for m in seen.values():
                verdict = adjudicate(self.meta[m.point_id], truth)
                if verdict is not None:
                    self.vault.update_utility(m.point_id, success=verdict)
                    self.stats["adjudicated"] += 1
                    self.stats["success" if verdict else "failure"] += 1
                self.vault.increment_frequency(m.point_id)
                self.vault.update_recency(m.point_id, current_time=when)
                self.stats["accesses"] += 1
        self.ingest(task, when)
        if truth["obj"] and truth["source"]:
            self.location_counts[(truth["obj"], truth["source"])] += 1
        self.stats["tasks"] += 1

    def payloads(self) -> Dict[str, Dict[str, Any]]:
        out = {}
        for chunk, _ in self.vault.get_all_memories(limit=2048, scroll_yield=True):
            for p in chunk:
                out[p.point_id] = p.payload.to_dict()
        return out


# ═══════════════════════════════════════════════════════════════════════════════
#  Held-out evaluation records
# ═══════════════════════════════════════════════════════════════════════════════


def eval_record(stream: Stream, task: ProcessedTask, payloads: Dict[str, Dict[str, Any]],
                now: float, split: str) -> Dict[str, Any]:
    truth = task_truth(task)
    query = location_query(truth["obj"]) if truth["obj"] else task.goal_description
    qv = np.asarray(stream.emb.encode(query))
    qv = qv / np.linalg.norm(qv)
    cands = []
    for pid, info in stream.meta.items():
        if info["kind"] == "location_concept" and truth["obj"] and info["obj"] == truth["obj"]:
            p = payloads[pid]
            comps = stream.vault.compute_components(p, current_time=now)
            v = info["vector"]
            cands.append({
                "pid": pid, "recep": info["recep"], "is_true": info["recep"] == truth["source"],
                "sim": float(np.dot(v, qv) / np.linalg.norm(v)),
                "count_prior": stream.location_counts[(info["obj"], info["recep"])],
                "recency": p["recency"], "created_at": p["created_at"], "now": now,
                "frequency": p["frequency"], "success": p["success_count"], "failure": p["failure_count"],
                **comps,
            })
    order = sorted(range(len(cands)), key=lambda i: (-cands[i]["sim"], i))
    for r, i in enumerate(order, 1):
        cands[i]["sim_rank"] = float(r)
    needed = []
    if truth["obj"] and truth["source"]:
        needed.append(stream.by_text.get(f"{truth['obj'].capitalize()} objects are typically found in "
                                         f"{truth['source'].capitalize()} receptacles"))
    for pid, info in stream.meta.items():
        if info["kind"] == "tool_concept" and truth["action"] and info["action"] == truth["action"] \
                and info["tool"] == truth["tool"]:
            needed.append(pid)
        if info["kind"] == "strategy" and info["task_type"] == truth["task_type"]:
            needed.append(pid)
    return {
        "task_id": task.task_id, "split": split, "task_type": task.task_type, "truth": truth,
        "query": query, "candidates": cands, "reachable": any(c["is_true"] for c in cands),
        "needed": sorted({p for p in needed if p}),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Analysis
# ═══════════════════════════════════════════════════════════════════════════════


def _mrr(order_is_true: Sequence[bool]) -> Tuple[float, float]:
    for r, t in enumerate(order_is_true, 1):
        if t:
            return 1.0 / r, float(r == 1)
    return 0.0, 0.0


def candidate_methods(weights: Sequence[float]) -> Dict[str, Any]:
    methods: Dict[str, Any] = {
        "similarity": ("sim",),
        "count_prior": ("key", "count_prior"),
        "utility": ("key", "U"),
        "frequency": ("key", "F"),
        "dars_score_only": ("dars", tuple(weights)),
        "dars_rrf": ("fused", Method("rrf", rank_mode="rrf", weights=tuple(weights))),
        "dars_wrrf_b0.5": ("fused", Method("wrrf", rank_mode="wrrf", beta_dars=0.5, weights=tuple(weights))),
        "dars_blend_a0.5": ("fused", Method("blend", rank_mode="blend", alpha=0.5, weights=tuple(weights))),
    }
    for c, w in leave_one_out_weights(tuple(weights)).items():
        methods[f"dars_rrf_minus_{c}"] = ("fused", Method(f"rrf-{c}", rank_mode="rrf", weights=w))
    return methods


def rank_candidates(cands: List[Dict[str, Any]], spec: Tuple, rng: np.random.Generator) -> List[int]:
    n = len(cands)
    tiebreak = rng.permutation(n)  # random tie-break (ties must not favour the true candidate)
    if spec[0] == "sim":
        return sorted(range(n), key=lambda i: (-cands[i]["sim"], tiebreak[i]))
    if spec[0] == "key":
        return sorted(range(n), key=lambda i: (-cands[i][spec[1]], tiebreak[i]))
    if spec[0] == "dars":
        s = [sum(w * cands[i][c] for w, c in zip(spec[1], COMPONENTS)) for i in range(n)]
        return sorted(range(n), key=lambda i: (-round(s[i], 6), tiebreak[i]))
    return rerank_candidates(cands, spec[1])


def analyze(run_dir: Path, split: str, n_boot: int, seed: int = 0) -> Dict[str, Any]:
    rows = [json.loads(l) for l in (run_dir / f"eval_{split}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    methods = candidate_methods(manifest["weights"])
    reachable = [r for r in rows if r["reachable"] and len(r["candidates"]) > 1]
    report: Dict[str, Any] = {"split": split, "tasks": len(rows), "reachable_with_alternatives": len(reachable),
                              "mean_candidates": float(np.mean([len(r["candidates"]) for r in reachable])) if reachable else 0,
                              "h3": {}, "h3_paired_vs_similarity": {}, "h5": {}}
    per_method: Dict[str, List[float]] = {}
    for name, spec in methods.items():
        rng = np.random.default_rng(seed)
        mrrs, hits = [], []
        for r in reachable:
            order = rank_candidates(r["candidates"], spec, rng)
            m, h = _mrr([r["candidates"][i]["is_true"] for i in order])
            mrrs.append(m)
            hits.append(h)
        per_method[name] = mrrs
        report["h3"][name] = {"mrr": cluster_bootstrap_mean(mrrs, n_boot=n_boot).as_dict(),
                              "hit@1": cluster_bootstrap_mean(hits, n_boot=n_boot).as_dict()}
    for name, mrrs in per_method.items():
        if name != "similarity":
            report["h3_paired_vs_similarity"][name] = paired_bootstrap_diff(mrrs, per_method["similarity"], n_boot=n_boot)

    snapshot = [json.loads(l) for l in (run_dir / "memories.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = np.random.default_rng(seed)
    scores = {
        "dars": {m["pid"]: m["S"] for m in snapshot},
        "recency_lru": {m["pid"]: m["recency"] for m in snapshot},
        "fifo": {m["pid"]: m["created_at"] for m in snapshot},
        "frequency_lfu": {m["pid"]: m["frequency"] for m in snapshot},
        "random": {m["pid"]: float(x) for m, x in zip(snapshot, rng.random(len(snapshot)))},
    }
    for keep in (0.25, 0.5, 0.75):
        report["h5"][str(keep)] = {}
        n_keep = int(math.ceil(keep * len(snapshot)))
        for name, sc in scores.items():
            kept = set(sorted(sc, key=lambda p: (-sc[p], p))[:n_keep])
            rates = [sum(1 for p in r["needed"] if p not in kept) / len(r["needed"]) for r in rows if r["needed"]]
            report["h5"][str(keep)][name] = cluster_bootstrap_mean(rates, n_boot=n_boot).as_dict()
    return report


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════


def cmd_run(args: argparse.Namespace) -> None:
    from config.settings import DARSConfig

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    DARSConfig.RECENCY_DECAY_LAMBDA = float(args.decay_lambda)
    data = load_splits()
    train = [t for t in (extract_task(r) for r in data["train"]) if t is not None]
    order = np.random.default_rng(SPLIT_SEED).permutation(len(train))
    n_dev = int(round(DEV_FRACTION * len(train)))
    dev_tasks = [train[i] for i in order[:n_dev]]
    stream_tasks = [train[i] for i in order[n_dev:]]
    if args.limit:
        stream_tasks = stream_tasks[: args.limit]

    from core.layer_d.embedding import EmbeddingEngine

    goal = np.asarray(EmbeddingEngine().encode(DARSConfig.GOAL_PRESETS["ALFWorld"]))
    stream = Stream(args, goal)
    t0 = time.time()
    for i, task in enumerate(stream_tasks):
        stream.step(task, T0 + i * args.task_step)
        if (i + 1) % 250 == 0:
            logger.info("ALFWorld stream: %d/%d tasks, %d memories (%.0fs)",
                        i + 1, len(stream_tasks), len(stream.meta), time.time() - t0)
    now = T0 + len(stream_tasks) * args.task_step
    payloads = stream.payloads()

    with (out / "memories.jsonl").open("w", encoding="utf-8") as fh:
        for pid, info in stream.meta.items():
            p = payloads[pid]
            comps = stream.vault.compute_components(p, current_time=now)
            fh.write(json.dumps({
                "pid": pid, "kind": info["kind"], "obj": info.get("obj"), "recep": info.get("recep"),
                "tool": info.get("tool"), "action": info.get("action"), "task_type": info["task_type"],
                "recency": p["recency"], "created_at": p["created_at"], "frequency": p["frequency"],
                "success": p["success_count"], "failure": p["failure_count"],
                "S": stream.vault.compute_dars_score(p, current_time=now), **comps,
            }) + "\n")

    eval_sets = {"dev": dev_tasks}
    if args.eval_test:
        eval_sets["test_in"] = [t for t in (extract_task(r) for r in data["eval_in_distribution"]) if t]
        eval_sets["test_out"] = [t for t in (extract_task(r) for r in data["eval_out_of_distribution"]) if t]
    for split, tasks in eval_sets.items():
        with (out / f"eval_{split}.jsonl").open("w", encoding="utf-8") as fh:
            for task in tasks:
                fh.write(json.dumps(eval_record(stream, task, payloads, now, split), default=float) + "\n")

    manifest = {
        "experiment": "E10_alfworld",
        "stream_tasks": len(stream_tasks),
        "dev_tasks": len(dev_tasks),
        "eval_sets": {k: len(v) for k, v in eval_sets.items()},
        "memories": len(stream.meta),
        "stream_stats": dict(stream.stats),
        "weights": list(DEFAULT_WEIGHTS),
        "decay_lambda_per_hour": args.decay_lambda,
        "task_step_s": args.task_step,
        "retrieval_k": args.k,
        "now": now,
        "location_query": "Where can I find a <Object>?",
        "goal": "ALFWorld preset",
        "elapsed_s": time.time() - t0,
        "provenance": collect_provenance(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("stream_tasks", "dev_tasks", "eval_sets", "memories", "stream_stats", "elapsed_s")}))


def _fused_order(sim: np.ndarray, S: np.ndarray, mode: str, param: float, rrf_k: int = 60) -> np.ndarray:
    """Candidate order under one fusion mode (mirrors MemoryVault.search_and_rerank)."""
    n = len(sim)
    by_sim = np.argsort(-sim, kind="stable")
    S = np.round(np.clip(S, 0.0, 1.0), 6)
    dars_order = by_sim[np.argsort(-S[by_sim], kind="stable")]
    r_sim = np.empty(n)
    r_sim[by_sim] = np.arange(1, n + 1)
    r_dars = np.empty(n)
    r_dars[dars_order] = np.arange(1, n + 1)
    if mode == "rrf":
        score = 1 / (rrf_k + r_sim) + 1 / (rrf_k + r_dars)
    elif mode == "wrrf":
        score = 1 / (rrf_k + r_sim) + param / (rrf_k + r_dars)
    elif mode == "blend":
        lo, hi = sim.min(), sim.max()
        norm = np.ones(n) if hi - lo < 0.05 else (sim - lo) / (hi - lo)
        score = param * norm + (1 - param) * S
    elif mode == "score_only":
        score = S
    else:
        raise ValueError(mode)
    return by_sim[np.argsort(-score[by_sim], kind="stable")]


def tune(run_dir: Path, split: str) -> Dict[str, Any]:
    """Weight × λ × fusion grid on the dev split: H3 (MRR) and H5 (harmful deletion at keep 50 %)."""
    from benchmarks.dars_eval.tuning import LAMBDAS, dirichlet_around, recompute_R, select, simplex_grid

    rows = [json.loads(l) for l in (run_dir / f"eval_{split}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    tasks = [r for r in rows if r["reachable"] and len(r["candidates"]) > 1]
    grid = simplex_grid(0.1)
    modes = [("rrf", 0.0), ("wrrf", 0.25), ("wrrf", 0.5), ("wrrf", 1.0), ("blend", 0.5), ("blend", 0.8), ("score_only", 0.0)]
    arrays = []
    for r in tasks:
        c = r["candidates"]
        arrays.append({
            "sim": np.array([x["sim"] for x in c]), "true": np.array([x["is_true"] for x in c]),
            "F": np.array([x["F"] for x in c]), "U": np.array([x["U"] for x in c]), "P": np.array([x["P"] for x in c]),
            "rec": np.array([x["recency"] for x in c]), "now": c[0]["now"],
        })

    def mrr_for(w, lam, mode, param) -> float:
        total = 0.0
        for a in arrays:
            S = w[0] * recompute_R(a["rec"], a["now"], lam) + w[1] * a["F"] + w[2] * a["U"] + w[3] * a["P"]
            order = _fused_order(a["sim"], S, mode, param)
            total += 1.0 / (int(np.flatnonzero(a["true"][order])[0]) + 1)
        return total / len(arrays)

    h3: Dict[Tuple, float] = {}
    for lam in LAMBDAS:
        for mode, param in modes:
            for w in grid:
                h3[(lam, mode, param, w)] = mrr_for(w, lam, mode, param)
    best, best_val = select(h3)
    default = {f"{m}:{p}": h3[(0.005, m, p, (0.3, 0.2, 0.3, 0.2))] for m, p in modes}
    robust = [mrr_for(tuple(w), best[0], best[1], best[2]) for w in dirichlet_around(best[3], draws=200)]

    snapshot = [json.loads(l) for l in (run_dir / "memories.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    now = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["now"]
    rec = np.array([m["recency"] for m in snapshot])
    F = np.array([m["F"] for m in snapshot]); U = np.array([m["U"] for m in snapshot]); P = np.array([m["P"] for m in snapshot])
    pid_index = {m["pid"]: i for i, m in enumerate(snapshot)}
    needed = [[pid_index[p] for p in r["needed"] if p in pid_index] for r in rows if r["needed"]]
    n_keep = int(math.ceil(0.5 * len(snapshot)))
    h5: Dict[Tuple, float] = {}
    for lam in LAMBDAS:
        R = recompute_R(rec, now, lam)
        for w in grid:
            S = w[0] * R + w[1] * F + w[2] * U + w[3] * P
            kept = _keep_mask(S, n_keep, 0)          # seeded random tie-break, never storage order
            h5[(lam, w)] = float(np.mean([np.mean(~kept[nd]) for nd in needed if nd]))
    best5, best5_val = select(h5, higher_is_better=False)
    return {
        "split": split, "tasks": len(tasks),
        "h3_best": {"lambda": best[0], "mode": best[1], "param": best[2], "weights": best[3], "mrr": best_val},
        "h3_default_weights_lambda_0.005": default,
        "h3_best_dirichlet_mrr": {"mean": float(np.mean(robust)), "p05": float(np.quantile(robust, 0.05)),
                                  "p95": float(np.quantile(robust, 0.95))},
        "h3_top10": sorted(([list(map(str, k)), v] for k, v in h3.items()), key=lambda kv: -kv[1])[:10],
        "h5_best": {"lambda": best5[0], "weights": best5[1], "harmful_deletion": best5_val},
        "h5_default_lambda_0.005": h5[(0.005, (0.3, 0.2, 0.3, 0.2))],
    }


def cmd_tune(args: argparse.Namespace) -> None:
    report = tune(Path(args.run), args.split)
    (Path(args.run) / f"tune_{args.split}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "h3_top10"}, indent=1))


def cmd_analyze(args: argparse.Namespace) -> None:
    report = analyze(Path(args.run), args.split, args.n_boot)
    (Path(args.run) / f"analysis_{args.split}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"split={report['split']} tasks={report['tasks']} reachable={report['reachable_with_alternatives']} "
          f"mean_candidates={report['mean_candidates']:.1f}")
    for name, v in sorted(report["h3"].items(), key=lambda kv: -kv[1]["mrr"]["mean"]):
        pv = report["h3_paired_vs_similarity"].get(name, {}).get("p_value")
        print(f"  {name:22s} MRR={v['mrr']['mean']:.3f} [{v['mrr']['ci_lo']:.3f},{v['mrr']['ci_hi']:.3f}] "
              f"hit@1={v['hit@1']['mean']:.3f}" + (f"  vs sim p={pv:.3g}" if pv is not None else ""))
    for keep, pols in report["h5"].items():
        print(f"  keep {keep}: harmful deletion " + ", ".join(f"{k}={v['mean']:.3f}" for k, v in pols.items()))


def _keep_mask(scores: np.ndarray, n_keep: int, seed: int) -> np.ndarray:
    """Top ``n_keep`` by score; ties broken by a seeded random order (never by creation order)."""
    tie = np.random.default_rng(seed).permutation(len(scores))
    order = np.lexsort((tie, -np.asarray(scores, dtype=float)))
    kept = np.zeros(len(scores), dtype=bool)
    kept[order[:n_keep]] = True
    return kept


def evaluate(run_dir: Path, split: str, h3: Tuple[float, str, float, Sequence[float]],
             h5: Tuple[float, Sequence[float]], keep: float = 0.5, n_boot: int = 10_000,
             seed: int = 0) -> Dict[str, Any]:
    """
    Pre-registered H3 and H5 tests for fixed, dev-selected configurations on one split
    (``dev``, ``test_in`` or ``test_out``).

    H3: MRR of the true location when candidates are ordered by the H3 configuration
        (``_fused_order``, as optimised by ``tune``), vs similarity (random tie-break, as
        in ``analyze``): paired bootstrap over tasks.  Count prior and the submitted
        default (rrf, default weights, λ = 0.005) are reported alongside.
    H5: harmful-deletion rate when the end-of-stream store keeps its top ``keep`` memories
        by the H5 configuration, vs LRU and FIFO (seeded random tie-breaks for every
        policy): paired bootstrap over tasks.
    """
    from benchmarks.dars_eval.tuning import recompute_R

    rows = [json.loads(l) for l in (run_dir / f"eval_{split}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    tasks = [r for r in rows if r["reachable"] and len(r["candidates"]) > 1]
    lam3, mode3, param3, w3 = h3
    per: Dict[str, List[float]] = {"selected": [], "similarity": [], "count_prior": [], "default_rrf": []}
    rng_sim, rng_cnt = np.random.default_rng(seed), np.random.default_rng(seed)
    for r in tasks:
        c = r["candidates"]
        true = [x["is_true"] for x in c]
        sim = np.array([x["sim"] for x in c])
        rec = np.array([x["recency"] for x in c])
        F, U, P = (np.array([x[k] for x in c]) for k in ("F", "U", "P"))
        now = c[0]["now"]
        S = w3[0] * recompute_R(rec, now, lam3) + w3[1] * F + w3[2] * U + w3[3] * P
        S0 = 0.3 * recompute_R(rec, now, 0.005) + 0.2 * F + 0.3 * U + 0.2 * P
        per["selected"].append(_mrr([true[i] for i in _fused_order(sim, S, mode3, param3)])[0])
        per["default_rrf"].append(_mrr([true[i] for i in _fused_order(sim, S0, "rrf", 0.0)])[0])
        per["similarity"].append(_mrr([true[i] for i in rank_candidates(c, ("sim",), rng_sim)])[0])
        per["count_prior"].append(_mrr([true[i] for i in rank_candidates(c, ("key", "count_prior"), rng_cnt)])[0])
    h3_out = {
        "config": {"lambda": lam3, "mode": mode3, "param": param3, "weights": list(w3)},
        "tasks": len(rows), "reachable_with_alternatives": len(tasks),
        "mrr": {k: cluster_bootstrap_mean(v, n_boot=n_boot).as_dict() for k, v in per.items()},
        "paired_vs_similarity": {k: paired_bootstrap_diff(v, per["similarity"], n_boot=n_boot)
                                 for k, v in per.items() if k != "similarity"},
    }

    snapshot = [json.loads(l) for l in (run_dir / "memories.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    now = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["now"]
    rec_all = np.array([m["recency"] for m in snapshot])
    F, U, P = (np.array([m[k] for m in snapshot]) for k in ("F", "U", "P"))
    pid_index = {m["pid"]: i for i, m in enumerate(snapshot)}
    needed = [nd for nd in ([pid_index[p] for p in r["needed"] if p in pid_index] for r in rows if r["needed"]) if nd]
    n_keep = int(math.ceil(keep * len(snapshot)))
    lam5, w5 = h5
    scores = {
        "selected": w5[0] * recompute_R(rec_all, now, lam5) + w5[1] * F + w5[2] * U + w5[3] * P,
        "default": 0.3 * recompute_R(rec_all, now, 0.005) + 0.2 * F + 0.3 * U + 0.2 * P,
        "recency_lru": rec_all,
        "fifo": np.array([m["created_at"] for m in snapshot]),
        "frequency_lfu": np.array([m["frequency"] for m in snapshot], dtype=float),
    }
    rates: Dict[str, List[float]] = {}
    for name, s in scores.items():
        kept = _keep_mask(s, n_keep, seed)
        rates[name] = [float(np.mean(~kept[nd])) for nd in needed]
    h5_out = {
        "config": {"lambda": lam5, "weights": list(w5)}, "keep": keep, "tasks_with_needed": len(needed),
        "harmful_deletion": {k: cluster_bootstrap_mean(v, n_boot=n_boot).as_dict() for k, v in rates.items()},
        "paired_vs_selected": {k: paired_bootstrap_diff(rates["selected"], v, n_boot=n_boot)
                               for k, v in rates.items() if k != "selected"},
    }
    return {"split": split, "h3": h3_out, "h5": h5_out}


def cmd_evaluate(args: argparse.Namespace) -> None:
    report = evaluate(Path(args.run), args.split,
                      (args.h3_lambda, args.h3_mode, args.h3_param, tuple(args.h3_weights)),
                      (args.h5_lambda, tuple(args.h5_weights)), keep=args.keep, n_boot=args.n_boot, seed=args.seed)
    report["provenance"] = collect_provenance()
    (Path(args.run) / f"evaluate_{args.split}.json").write_text(json.dumps(report, indent=1, default=str),
                                                               encoding="utf-8")
    h3, h5 = report["h3"], report["h5"]
    print(f"split={args.split} tasks={h3['tasks']} reachable={h3['reachable_with_alternatives']}")
    for k, v in h3["mrr"].items():
        d = h3["paired_vs_similarity"].get(k)
        print(f"  H3 {k:12s} MRR={v['mean']:.3f} [{v['ci_lo']:.3f},{v['ci_hi']:.3f}]"
              + (f"  vs similarity {d['diff']:+.3f} p={d['p_value']:.3g}" if d else ""))
    for k, v in h5["harmful_deletion"].items():
        d = h5["paired_vs_selected"].get(k)
        print(f"  H5 {k:14s} harmful={v['mean']:.3f}"
              + (f"  selected minus {k}: {d['diff']:+.3f} p={d['p_value']:.3g}" if d else ""))


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="E10 ALFWorld task stream")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--limit", type=int, default=0, help="cap stream tasks (smoke tests)")
    r.add_argument("--k", type=int, default=10)
    r.add_argument("--task-step", type=float, default=3600.0)
    r.add_argument("--decay-lambda", type=float, default=0.005)
    r.add_argument("--eval-test", action="store_true", help="also write held-out test evaluation (pre-registered run only)")
    r.set_defaults(func=cmd_run)
    a = sub.add_parser("analyze")
    a.add_argument("--run", required=True)
    a.add_argument("--split", default="dev")
    a.add_argument("--n-boot", type=int, default=5000)
    a.set_defaults(func=cmd_analyze)
    t = sub.add_parser("tune")
    t.add_argument("--run", required=True)
    t.add_argument("--split", default="dev")
    t.set_defaults(func=cmd_tune)
    e = sub.add_parser("evaluate", help="pre-registered H3/H5 tests for fixed, dev-selected configurations")
    e.add_argument("--run", required=True)
    e.add_argument("--split", default="test_in")
    e.add_argument("--h3-lambda", type=float, required=True)
    e.add_argument("--h3-mode", choices=("rrf", "wrrf", "blend", "score_only"), required=True)
    e.add_argument("--h3-param", type=float, default=0.0)
    e.add_argument("--h3-weights", type=float, nargs=4, required=True)
    e.add_argument("--h5-lambda", type=float, required=True)
    e.add_argument("--h5-weights", type=float, nargs=4, required=True)
    e.add_argument("--keep", type=float, default=0.5)
    e.add_argument("--n-boot", type=int, default=10_000)
    e.add_argument("--seed", type=int, default=0)
    e.set_defaults(func=cmd_evaluate)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for noisy in ("httpx", "core.layer_d.storage"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args.func(args)


if __name__ == "__main__":
    main()
