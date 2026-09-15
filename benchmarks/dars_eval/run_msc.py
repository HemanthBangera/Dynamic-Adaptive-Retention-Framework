"""
E9 — Multi-Session Chat (MSC): do DARS retention scores predict which memories
will be needed later?  (Hypothesis H4; H5 for budgeted eviction.)

Protocol, per dialogue (sessions on a virtual clock, ``--session-gap`` apart)
-----------------------------------------------------------------------------
session 0   the initial persona facts of both speakers are stored.
session 1,2 every dialogue turn retrieves the top-``k`` stored facts by semantic
            similarity (the same retrieval for every scoring policy, so all
            policies are compared on an identical memory state).  Each retrieved
            fact receives Layer B feedback — success if it is restated in that
            session's persona summary (lexical match), else failure — updating
            utility, frequency and recency on the virtual clock.  After the
            conversation, the session's new persona facts are stored; sentences
            that restate an existing fact are not stored again.
end of s2   every stored fact is scored by each retention policy (DARS, its
            leave-one-out ablations, recency/LRU, FIFO, frequency/LFU, utility,
            mention count, random).
label       whether the fact is restated in the speaker's session-3 persona
            summary (token F1 ≥ τ; τ = 0.5 primary, 0.6 / 0.7 and an embedding
            match at cosine ≥ 0.80 as robustness).  Session 3 is never used before
            scoring.

Matching uses lexical token F1 (SQuAD normalisation), never the retrieval
embedder, so labels are independent of the ranking model.  No LLM calls.

Usage
-----
python -m benchmarks.dars_eval.run_msc run     --out benchmark_runs/revision/E9/dev --split dev
python -m benchmarks.dars_eval.run_msc analyze --run benchmark_runs/revision/E9/dev
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from benchmarks.dars_eval.provenance import collect_provenance
from benchmarks.dars_eval.rankers import COMPONENTS, leave_one_out_weights
from benchmarks.dars_eval.splits import SPLIT_SEED
from benchmarks.dars_eval.stats import auroc_delong, auroc_delong_paired, cluster_bootstrap_mean
from core.layer_d.schema import DARSWeights
from core.layer_d.storage import MemoryVault
from third_party.memoryagentbench_eval.eval_other_utils import f1_score

logger = logging.getLogger(__name__)

T0 = 1_700_000_000.0
DEV_FRACTION = 0.30
DEFAULT_WEIGHTS = (0.30, 0.20, 0.30, 0.20)
LABEL_TAUS = (0.5, 0.6, 0.7)
EMBED_LABEL_COS = 0.80


# ═══════════════════════════════════════════════════════════════════════════════
#  Data
# ═══════════════════════════════════════════════════════════════════════════════


def load_msc_dialogues(hf_split: str = "train", label_session: int = 3) -> Dict[int, Dict[int, Dict[str, Any]]]:
    """Dialogues of one Hugging Face split that have every session up to ``label_session``.

    The pre-registered E9 study uses ``train`` with sessions 0-3. The ``validation`` and ``test``
    splits (five sessions each) were never used by it and serve the confirmatory addendum.
    """
    from datasets import load_dataset

    ds = load_dataset("nayohan/multi_session_chat", split=hf_split)
    by: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for row in ds:
        by[int(row["dialoug_id"])][int(row["session_id"])] = row
    needed = set(range(label_session + 1))
    return {d: s for d, s in by.items() if needed <= set(s)}


def dialogue_splits(dialogue_ids: Sequence[int], seed: int = SPLIT_SEED,
                    dev_fraction: float = DEV_FRACTION) -> Dict[int, str]:
    ids = sorted(dialogue_ids)
    order = np.random.default_rng(seed).permutation(len(ids))
    n_dev = int(round(dev_fraction * len(ids)))
    dev = {ids[i] for i in order[:n_dev]}
    return {d: ("dev" if d in dev else "test") for d in ids}


def token_f1(a: str, b: str) -> float:
    return float(f1_score(a, b)[0])


def best_match(sentence: str, candidates: Sequence[str]) -> Tuple[int, float]:
    best_i, best = -1, 0.0
    for i, c in enumerate(candidates):
        s = token_f1(sentence, c)
        if s > best:
            best_i, best = i, s
    return best_i, best


def persona_sentences(session: Dict[str, Any]) -> List[Tuple[int, str]]:
    out = [(1, s.strip()) for s in session.get("persona1", []) if s and s.strip()]
    out += [(2, s.strip()) for s in session.get("persona2", []) if s and s.strip()]
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  One dialogue
# ═══════════════════════════════════════════════════════════════════════════════


def run_dialogue(did: int, sessions: Dict[int, Dict[str, Any]], args: argparse.Namespace,
                 goal_vector: Optional[np.ndarray]) -> List[Dict[str, Any]]:
    """Stream sessions 1 .. label_session-1, then label facts by session ``label_session``.

    With ``args.record_writes`` each fact also carries its write history: the sessions whose
    persona summary stated it (creation and every later restatement), the time of the last
    such write, and how many later summaries it could have been restated in. These do not
    depend on what retrieval returned, unlike recency, frequency and utility in the vault.
    """
    label_session = int(getattr(args, "label_session", 3))
    record_writes = bool(getattr(args, "record_writes", False))
    vault = MemoryVault(collection_name=f"msc_{did}", location=":memory:",
                        weights=DARSWeights(*DEFAULT_WEIGHTS))
    vault.initialize_collection(recreate=True)
    emb = vault.embedder
    facts: List[Dict[str, Any]] = []          # text, speaker, session, pid, mentions
    gap = args.session_gap
    turn_step = args.turn_step

    def store(new: List[Tuple[int, str]], when: float, session_idx: int) -> None:
        for speaker, sentence in new:
            same = [f for f in facts if f["speaker"] == speaker]
            j, score = best_match(sentence, [f["text"] for f in same])
            if j >= 0 and score >= args.dedup_tau:
                same[j]["mentions"] += 1
                same[j]["mention_sessions"].append(session_idx)
                same[j]["last_write_time"] = when
                continue
            vec = emb.encode(sentence)
            if goal_vector is not None:
                p = float(np.clip(np.dot(vec, goal_vector) / (np.linalg.norm(vec) * np.linalg.norm(goal_vector)), 0, 1))
            else:
                p = 0.0
            pid = vault.store_memory(sentence, predictive_value=p, vector_override=vec,
                                     tags=[f"speaker:{speaker}"], sim_timestamp=when)
            facts.append({"text": sentence, "speaker": speaker, "session": session_idx,
                          "pid": pid, "mentions": 1, "mention_sessions": [session_idx],
                          "last_write_time": when})

    store(persona_sentences(sessions[0]), T0, 0)

    for s in range(1, label_session):
        t_s = T0 + s * gap
        summary = persona_sentences(sessions[s])
        summary_by_speaker = {1: [t for sp, t in summary if sp == 1], 2: [t for sp, t in summary if sp == 2]}
        turns = [t for t in sessions[s].get("dialogue", []) if t and t.strip()]
        turn_vecs = emb.encode_batch(turns) if turns else []
        pid_to_fact = {f["pid"]: f for f in facts}
        for ti, (turn, qv) in enumerate(zip(turns, turn_vecs)):
            now = t_s + (ti + 1) * turn_step
            hits = vault.search_and_rerank(turn, fetch_k=args.k, top_n=args.k, rank_mode="similarity",
                                           current_time=now, query_vector=qv)
            for m in hits:
                fact = pid_to_fact[m.point_id]
                _, score = best_match(fact["text"], summary_by_speaker[fact["speaker"]])
                success = score >= args.feedback_tau
                vault.update_utility(m.point_id, success=success)
                vault.increment_frequency(m.point_id)
                vault.update_recency(m.point_id, current_time=now)
        store(summary, t_s + (len(turns) + 1) * turn_step, s)

    t_end = T0 + (label_session - 1) * gap + args.turn_step * 200
    s3 = persona_sentences(sessions[label_session])
    s3_by_speaker = {1: [t for sp, t in s3 if sp == 1], 2: [t for sp, t in s3 if sp == 2]}
    s3_vecs = {sp: (np.asarray(emb.encode_batch(v)) if v else None) for sp, v in s3_by_speaker.items()}

    rows = []
    for f in facts:
        mem = vault.get_memory(f["pid"])
        payload = mem.payload.to_dict()
        comps = vault.compute_components(payload, current_time=t_end)
        labels = {}
        for tau in LABEL_TAUS:
            _, score = best_match(f["text"], s3_by_speaker[f["speaker"]])
            labels[f"lex_{tau}"] = bool(score >= tau)
        vecs = s3_vecs[f["speaker"]]
        if vecs is not None and len(vecs):
            v = np.asarray(mem.vector)
            cos = vecs @ v / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(v))
            labels["embed_0.8"] = bool(cos.max() >= EMBED_LABEL_COS)
        else:
            labels["embed_0.8"] = False
        row = {
            "dialogue": did, "speaker": f["speaker"], "text": f["text"], "created_session": f["session"],
            "mentions": f["mentions"], "frequency": payload["frequency"],
            "success": payload["success_count"], "failure": payload["failure_count"],
            "recency": payload["recency"], "created_at": payload["created_at"],
            "components": comps, "t_end": t_end, "labels": labels,
        }
        if record_writes:
            row["mention_sessions"] = list(f["mention_sessions"])
            row["last_write_time"] = f["last_write_time"]
            row["opportunities"] = (label_session - 1) - f["session"]
            row["label_session"] = label_session
        rows.append(row)
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
#  Scoring policies and analysis
# ═══════════════════════════════════════════════════════════════════════════════


def dars_score(comps: Dict[str, float], weights: Sequence[float]) -> float:
    return round(min(max(sum(w * comps[c] for w, c in zip(weights, COMPONENTS)), 0.0), 1.0), 6)


def policy_scores(rows: Sequence[Dict[str, Any]], weights: Sequence[float], seed: int = 0) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = {
        "dars": np.array([dars_score(r["components"], weights) for r in rows]),
        "recency_lru": np.array([r["recency"] for r in rows]),
        "fifo": np.array([r["created_at"] for r in rows]),
        "frequency_lfu": np.array([r["frequency"] for r in rows], dtype=float),
        "utility": np.array([r["components"]["U"] for r in rows]),
        "predictive": np.array([r["components"]["P"] for r in rows]),
        "mention_count": np.array([r["mentions"] for r in rows], dtype=float),
        "random": rng.random(len(rows)),
    }
    for c, w in leave_one_out_weights(tuple(weights)).items():
        out[f"dars_minus_{c}"] = np.array([dars_score(r["components"], w) for r in rows])
    return out


def eviction(rows: Sequence[Dict[str, Any]], scores: np.ndarray, label: str, keep_fraction: float,
             seed: int = 0) -> Dict[str, List[float]]:
    """Per dialogue: keep the top ``keep_fraction`` facts; harmful deletions = needed facts evicted.

    Score ties (e.g. facts stored in the same session) are broken by a seeded random
    order, identical for every policy, never by position in the dialogue.
    """
    tie = np.random.default_rng(seed).random(len(rows))
    by_d: Dict[int, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_d[r["dialogue"]].append(i)
    harmful, recall_kept, dialogues = [], [], []
    for d, idx in by_d.items():
        n_keep = max(1, int(math.ceil(keep_fraction * len(idx))))
        order = sorted(idx, key=lambda i: (-scores[i], tie[i]))
        kept = set(order[:n_keep])
        needed = [i for i in idx if rows[i]["labels"][label]]
        if not needed:
            continue
        evicted_needed = sum(1 for i in needed if i not in kept)
        harmful.append(evicted_needed / len(needed))
        recall_kept.append(1 - evicted_needed / len(needed))
        dialogues.append(d)
    return {"harmful_deletion_rate": harmful, "needed_retained": recall_kept, "dialogues": dialogues}


def analyze(run_dir: Path, split: str, n_boot: int) -> Dict[str, Any]:
    rows = [json.loads(l) for l in (run_dir / "facts.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if split == "all" or r["split"] == split]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    weights = manifest["weights"]
    scores = policy_scores(rows, weights)
    report: Dict[str, Any] = {"split": split, "facts": len(rows),
                              "dialogues": len({r["dialogue"] for r in rows}), "labels": {}}
    for label in [f"lex_{t}" for t in LABEL_TAUS] + ["embed_0.8"]:
        y = np.array([r["labels"][label] for r in rows])
        entry: Dict[str, Any] = {"base_rate": float(y.mean()), "auroc": {}, "paired_vs_dars": {}, "eviction": {}}
        if y.all() or not y.any():
            report["labels"][label] = entry
            continue
        for name, s in scores.items():
            entry["auroc"][name] = auroc_delong(s, y)
            if name != "dars":
                entry["paired_vs_dars"][name] = auroc_delong_paired(scores["dars"], s, y)
        for m in (0.25, 0.5, 0.75):
            entry["eviction"][str(m)] = {}
            for name in ("dars", "recency_lru", "fifo", "frequency_lfu", "mention_count", "random"):
                ev = eviction(rows, scores[name], label, m)
                entry["eviction"][str(m)][name] = cluster_bootstrap_mean(
                    ev["harmful_deletion_rate"], ev["dialogues"], n_boot=n_boot).as_dict()
        report["labels"][label] = entry
    return report


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════


def cmd_run(args: argparse.Namespace) -> None:
    from config.settings import DARSConfig

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    DARSConfig.RECENCY_DECAY_LAMBDA = float(args.decay_lambda)
    hf_split = getattr(args, "hf_split", "train")
    label_session = int(getattr(args, "label_session", 3))
    dialogues = load_msc_dialogues(hf_split, label_session)
    if hf_split == "train":
        splits = dialogue_splits(list(dialogues))
    else:                       # an untouched split is used whole, never divided into dev/test
        if args.split != "all":
            raise SystemExit(f"--hf-split {hf_split} is used whole; pass --split all")
        splits = {d: hf_split for d in dialogues}
    chosen = [d for d in sorted(dialogues) if args.split == "all" or splits[d] == args.split]
    if args.limit:
        chosen = chosen[: args.limit]

    goal_vector = None
    if args.goal != "none":
        from core.layer_d.embedding import EmbeddingEngine

        desc = DARSConfig.GOAL_PRESETS["MSC" if args.goal == "msc" else "ALFWorld"]
        goal_vector = np.asarray(EmbeddingEngine().encode(desc))

    t0 = time.time()
    n_facts = 0
    with (out / "facts.jsonl").open("w", encoding="utf-8") as fh:
        for i, did in enumerate(chosen):
            for r in run_dialogue(did, dialogues[did], args, goal_vector):
                r["split"] = splits[did]
                if hf_split != "train":     # dialogue ids restart in every split
                    r["dialogue"] = f"{hf_split}:{did}"
                fh.write(json.dumps(r) + "\n")
                n_facts += 1
            if (i + 1) % 50 == 0:
                logger.info("MSC: %d/%d dialogues (%.0fs)", i + 1, len(chosen), time.time() - t0)
    manifest = {
        "experiment": "E9_msc",
        "split": args.split,
        "dialogues": len(chosen),
        "facts": n_facts,
        "weights": list(DEFAULT_WEIGHTS),
        "decay_lambda_per_hour": args.decay_lambda,
        "session_gap_s": args.session_gap,
        "turn_step_s": args.turn_step,
        "retrieval_k": args.k,
        "feedback_tau": args.feedback_tau,
        "dedup_tau": args.dedup_tau,
        "label_taus": list(LABEL_TAUS),
        "embed_label_cos": EMBED_LABEL_COS,
        "goal": args.goal,
        "hf_split": hf_split,
        "label_session": label_session,
        "record_writes": bool(getattr(args, "record_writes", False)),
        "elapsed_s": time.time() - t0,
        "provenance": collect_provenance(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    print(f"wrote {n_facts} facts from {len(chosen)} dialogues to {out}")


def tune(run_dir: Path, split: str, label: str = "lex_0.5") -> Dict[str, Any]:
    """Weight × λ grid on the dev split: H4 (AUROC) and H5 (harmful deletion at keep 50 %)."""
    from benchmarks.dars_eval.tuning import (
        DEFAULT_WEIGHTS as DW, LAMBDAS, auc_rows, dirichlet_around, recompute_R, select, simplex_grid,
    )

    rows = [json.loads(l) for l in (run_dir / "facts.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if split == "all" or r["split"] == split]
    y = np.array([r["labels"][label] for r in rows])
    rec = np.array([r["recency"] for r in rows])
    t_end = np.array([r["t_end"] for r in rows])
    F = np.array([r["components"]["F"] for r in rows])
    U = np.array([r["components"]["U"] for r in rows])
    P = np.array([r["components"]["P"] for r in rows])
    grid = np.array(simplex_grid(0.1))
    by_d: Dict[int, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_d[r["dialogue"]].append(i)
    groups = [np.array(ix) for ix in by_d.values() if y[ix].any()]

    tie = np.random.default_rng(0).random(len(rows))      # the same seeded tie-break as eviction()

    def harmful(S: np.ndarray) -> float:
        rates = []
        for ix in groups:
            n_keep = max(1, int(math.ceil(0.5 * len(ix))))
            kept = set(ix[np.lexsort((tie[ix], -S[ix]))[:n_keep]].tolist())
            need = ix[y[ix]]
            rates.append(sum(1 for i in need if i not in kept) / len(need))
        return float(np.mean(rates))

    h4: Dict[Tuple, float] = {}
    h5: Dict[Tuple, float] = {}
    for lam in LAMBDAS:
        comps = np.vstack([recompute_R(rec, t_end, lam), F, U, P])          # 4 × N
        S = grid @ comps                                                    # configs × N
        aucs = auc_rows(S, y)
        for w, a, s in zip(map(tuple, grid), aucs, S):
            h4[(lam, w)] = float(a)
            h5[(lam, w)] = harmful(s)
    best4, val4 = select(h4)
    best5, val5 = select(h5, higher_is_better=False)
    key_default = (0.005, tuple(DW))

    def robust(best: Tuple) -> Dict[str, float]:
        comps = np.vstack([recompute_R(rec, t_end, best[0]), F, U, P])
        draws = dirichlet_around(best[1], draws=1000)
        a = auc_rows(draws @ comps, y)
        return {"mean": float(a.mean()), "p05": float(np.quantile(a, 0.05)), "p95": float(np.quantile(a, 0.95))}

    return {
        "split": split, "label": label, "facts": len(rows),
        "h4_best": {"lambda": best4[0], "weights": best4[1], "auroc": val4},
        "h4_default": h4[key_default],
        "h4_best_dirichlet": robust(best4),
        "h4_default_dirichlet": robust(key_default),
        "h4_single_component": {c: h4[(best4[0], tuple(1.0 if j == i else 0.0 for j in range(4)))]
                                for i, c in enumerate(COMPONENTS)},
        "h5_best": {"lambda": best5[0], "weights": best5[1], "harmful_deletion": val5},
        "h5_default": h5[key_default],
        "h4_top10": sorted(([list(map(str, k)), v] for k, v in h4.items()), key=lambda kv: -kv[1])[:10],
    }


def cmd_tune(args: argparse.Namespace) -> None:
    report = tune(Path(args.run), args.split)
    (Path(args.run) / f"tune_{args.split}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "h4_top10"}, indent=1))


def cmd_analyze(args: argparse.Namespace) -> None:
    report = analyze(Path(args.run), args.split, args.n_boot)
    (Path(args.run) / f"analysis_{args.split}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    lab = report["labels"].get("lex_0.5", {})
    print(f"split={report['split']} facts={report['facts']} dialogues={report['dialogues']} "
          f"base_rate(lex_0.5)={lab.get('base_rate')}")
    for name, a in sorted(lab.get("auroc", {}).items(), key=lambda kv: -kv[1]["auc"]):
        pv = lab["paired_vs_dars"].get(name, {}).get("p_value")
        print(f"  {name:22s} AUROC={a['auc']:.3f} [{a['ci_lo']:.3f},{a['ci_hi']:.3f}]"
              + (f"  vs DARS p={pv:.3g}" if pv is not None else ""))
    for m, pols in lab.get("eviction", {}).items():
        print(f"  keep {m}: " + ", ".join(f"{k}={v['mean']:.3f}" for k, v in pols.items()))


def evaluate(run_dir: Path, split: str, h4: Tuple[float, Sequence[float]], h5: Tuple[float, Sequence[float]],
             label: str = "lex_0.5", keep: float = 0.5, n_boot: int = 10_000) -> Dict[str, Any]:
    """
    Pre-registered H4 and H5 tests for fixed, dev-selected configurations on one split.

    H4: AUROC of the DARS score (the H4 λ and weights; R recomputed from each fact's
        last access) for restatement, vs recency-only (LRU) and frequency-only (LFU):
        paired DeLong.  FIFO, utility-only and mention count are reported as well.
    H5: harmful-deletion rate when each dialogue keeps its top ``keep`` facts by the
        H5 configuration, vs LRU and FIFO: paired bootstrap over dialogues.
    The submitted default (weights (0.3, 0.2, 0.3, 0.2), λ = 0.005) is reported alongside.
    """
    from benchmarks.dars_eval.stats import paired_bootstrap_diff

    rows = [json.loads(l) for l in (run_dir / "facts.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if split == "all" or r["split"] == split]
    if not rows:
        raise ValueError(f"No facts for split {split!r} in {run_dir}")
    y = np.array([r["labels"][label] for r in rows])

    def dars_at(lam: float, weights: Sequence[float]) -> np.ndarray:
        return np.array([dars_score({**r["components"],
                                     "R": math.exp(-lam * max(r["t_end"] - r["recency"], 0.0) / 3600.0)}, weights)
                         for r in rows])

    baselines = {
        "recency_lru": np.array([r["recency"] for r in rows]),
        "frequency_lfu": np.array([r["frequency"] for r in rows], dtype=float),
        "fifo": np.array([r["created_at"] for r in rows]),
        "utility": np.array([r["components"]["U"] for r in rows]),
        "mention_count": np.array([r["mentions"] for r in rows], dtype=float),
    }
    s4 = dars_at(h4[0], h4[1])
    default = dars_at(0.005, DEFAULT_WEIGHTS)
    h4_out = {
        "config": {"lambda": h4[0], "weights": list(h4[1])},
        "primary_comparators": ["recency_lru", "frequency_lfu"],
        "auroc_selected": auroc_delong(s4, y),
        "auroc_default": auroc_delong(default, y),
        "comparators": {k: auroc_delong(v, y) for k, v in baselines.items()},
        "paired": {k: auroc_delong_paired(s4, v, y) for k, v in baselines.items()},
        "paired_default": auroc_delong_paired(s4, default, y),
    }

    s5 = dars_at(h5[0], h5[1])
    sel = eviction(rows, s5, label, keep)
    h5_out: Dict[str, Any] = {
        "config": {"lambda": h5[0], "weights": list(h5[1])}, "keep": keep,
        "primary_comparators": ["recency_lru", "fifo"],
        "selected": cluster_bootstrap_mean(sel["harmful_deletion_rate"], sel["dialogues"], n_boot=n_boot).as_dict(),
        "comparators": {}, "paired": {},
    }
    for name, scores in (("default", default), ("recency_lru", baselines["recency_lru"]), ("fifo", baselines["fifo"]),
                         ("frequency_lfu", baselines["frequency_lfu"]), ("mention_count", baselines["mention_count"])):
        ev = eviction(rows, scores, label, keep)
        if ev["dialogues"] != sel["dialogues"]:
            raise RuntimeError("Eviction comparison is not paired over the same dialogues")
        h5_out["comparators"][name] = cluster_bootstrap_mean(
            ev["harmful_deletion_rate"], ev["dialogues"], n_boot=n_boot).as_dict()
        h5_out["paired"][name] = paired_bootstrap_diff(
            sel["harmful_deletion_rate"], ev["harmful_deletion_rate"], sel["dialogues"], n_boot=n_boot)
    return {"split": split, "label": label, "facts": len(rows), "dialogues": len({r["dialogue"] for r in rows}),
            "base_rate": float(y.mean()), "h4": h4_out, "h5": h5_out}


def cmd_evaluate(args: argparse.Namespace) -> None:
    report = evaluate(Path(args.run), args.split, (args.h4_lambda, tuple(args.h4_weights)),
                      (args.h5_lambda, tuple(args.h5_weights)), label=args.label, keep=args.keep, n_boot=args.n_boot)
    report["provenance"] = collect_provenance()
    (Path(args.run) / f"evaluate_{args.split}.json").write_text(json.dumps(report, indent=1, default=str),
                                                               encoding="utf-8")
    h4, h5 = report["h4"], report["h5"]
    print(f"split={args.split} facts={report['facts']} dialogues={report['dialogues']} base_rate={report['base_rate']:.3f}")
    print(f"  H4 AUROC selected={h4['auroc_selected']['auc']:.3f} default={h4['auroc_default']['auc']:.3f}")
    for k, v in h4["paired"].items():
        print(f"    vs {k:14s} {v['auc_b']:.3f}  diff={v['diff']:+.3f} p={v['p_value']:.3g}")
    print(f"  H5 harmful deletion (keep {h5['keep']}) selected={h5['selected']['mean']:.3f}")
    for k, v in h5["paired"].items():
        print(f"    vs {k:14s} {h5['comparators'][k]['mean']:.3f}  diff={v['diff']:+.3f} p={v['p_value']:.3g}")


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="E9 MSC retention prediction")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--k", type=int, default=3)
    r.add_argument("--session-gap", type=float, default=24 * 3600.0)
    r.add_argument("--turn-step", type=float, default=60.0)
    r.add_argument("--decay-lambda", type=float, default=0.005)
    r.add_argument("--feedback-tau", type=float, default=0.5)
    r.add_argument("--dedup-tau", type=float, default=0.5)
    r.add_argument("--goal", choices=("msc", "alfworld", "none"), default="msc")
    r.add_argument("--hf-split", choices=("train", "validation", "test"), default="train",
                   help="Hugging Face split; validation/test are the untouched confirmatory data")
    r.add_argument("--label-session", type=int, default=3,
                   help="session whose persona summary labels the facts; earlier sessions are streamed")
    r.add_argument("--record-writes", action="store_true",
                   help="also record each fact's write history (mention sessions, last write time)")
    r.set_defaults(func=cmd_run)
    a = sub.add_parser("analyze")
    a.add_argument("--run", required=True)
    a.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    a.add_argument("--n-boot", type=int, default=5000)
    a.set_defaults(func=cmd_analyze)
    t = sub.add_parser("tune")
    t.add_argument("--run", required=True)
    t.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    t.set_defaults(func=cmd_tune)
    e = sub.add_parser("evaluate", help="pre-registered H4/H5 tests for fixed, dev-selected configurations")
    e.add_argument("--run", required=True)
    e.add_argument("--split", choices=("dev", "test", "all"), default="test")
    e.add_argument("--label", default="lex_0.5")
    e.add_argument("--h4-lambda", type=float, required=True)
    e.add_argument("--h4-weights", type=float, nargs=4, required=True)
    e.add_argument("--h5-lambda", type=float, required=True)
    e.add_argument("--h5-weights", type=float, nargs=4, required=True)
    e.add_argument("--keep", type=float, default=0.5)
    e.add_argument("--n-boot", type=int, default=10_000)
    e.set_defaults(func=cmd_evaluate)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
    for noisy in ("httpx", "core.layer_d.storage"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args.func(args)


if __name__ == "__main__":
    main()
