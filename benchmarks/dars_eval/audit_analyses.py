"""
Secondary analyses announced by the fourth audit (deviations log, 2026-09-12), run on existing
test-phase artifacts. None of them changes a pre-registered result; each reports what a reader
needs in order to interpret one. Every command first reproduces a published test number from the
same artifact (a known-answer check) before computing anything new.

    python -m benchmarks.dars_eval.audit_analyses h1             # L1, L2: by-construction rows, tight budgets
    python -m benchmarks.dars_eval.audit_analyses alfworld-rank  # L7, L8: tie-break sensitivity, count prior
    python -m benchmarks.dars_eval.audit_analyses alfworld-evict # L9: needed-memory composition, eviction baselines
    python -m benchmarks.dars_eval.audit_analyses msc            # L11, L12, L14, L15: strongest baselines, labels, clustering
    python -m benchmarks.dars_eval.audit_analyses transfer       # P1: cross-domain weight transfer (offline parts)
    python -m benchmarks.dars_eval.audit_analyses robustness     # P2: test-split Dirichlet and decay-rate sensitivity
    python -m benchmarks.dars_eval.audit_analyses persistence    # L20: does a memory needed once get needed again?
    python -m benchmarks.dars_eval.audit_analyses lme-age        # L17: age of LongMemEval evidence at question time

Outputs go to ``benchmark_runs/revision/test/E13_audit/<command>.json``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from benchmarks.dars_eval.provenance import collect_provenance
from benchmarks.dars_eval.stats import (
    auroc_cluster_bootstrap, auroc_cluster_bootstrap_paired, auroc_delong, cluster_bootstrap_mean,
    paired_bootstrap_diff,
)

TEST = Path("benchmark_runs/revision/test")
OUT = TEST / "E13_audit"
N_BOOT = 10_000

E1_SOURCES = {"ruler_qa1_197K": "ruler_qa1_197K", "ruler_qa2_421K": "ruler_qa2_421K",
              "longmemeval_s": "longmemeval_s*", "factconsolidation_sh_32k": "factconsolidation_sh_32k",
              "factconsolidation_mh_32k": "factconsolidation_mh_32k"}

SELECTED = {                              # pre-registration section 11
    "msc_h4": (0.05, (0.2, 0.0, 0.8, 0.0)),
    "msc_h5": (0.0005, (0.3, 0.0, 0.7, 0.0)),
    "alfworld_h3": (0.0005, (0.0, 1.0, 0.0, 0.0)),
    "alfworld_h5": (0.0005, (0.0, 0.0, 0.0, 1.0)),
    "default": (0.005, (0.3, 0.2, 0.3, 0.2)),
}


def _jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def _check(name: str, got: float, expected: float, tol: float = 5e-4) -> Dict[str, float]:
    if abs(got - expected) > tol:
        raise AssertionError(f"known-answer check failed for {name}: got {got:.6f}, published {expected:.6f}")
    return {"check": name, "reproduced": got, "published": expected}


def _write(name: str, report: Dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report["provenance"] = collect_provenance()
    (OUT / f"{name}.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"wrote {OUT / (name + '.json')}")


# ═══════════════════════════════════════════════════════════════════════════════
#  L1, L2 — H1 evidence recall: identical by construction, and what happens when it is not
# ═══════════════════════════════════════════════════════════════════════════════


def h1(n_boot: int = N_BOOT) -> Dict[str, Any]:
    report: Dict[str, Any] = {"note": "shown set at budget B = ranked[:n_units@B]; fetch_k of dars_rrf_k15 is 15",
                              "sources": {}, "checks": []}
    for folder, source in E1_SOURCES.items():
        rows = [r for r in _jsonl(TEST / "E1" / folder / "per_question.jsonl") if r["split"] == "test"]
        by = defaultdict(dict)
        for r in rows:
            by[(r["context"], r["question"])][r["method"]] = r
        keys = [k for k, v in by.items() if "similarity" in v and "dars_rrf_k15" in v
                and "group_recall" in next(iter(v["similarity"]["budgets"].values()))]   # unlabelled questions carry no metrics
        summary = json.loads((TEST / "E1" / folder / "summary.json").read_text(encoding="utf-8"))
        budgets = sorted(int(b) for b in by[keys[0]]["similarity"]["budgets"])
        entry: Dict[str, Any] = {"questions": len(keys), "budgets": {}}
        for method in ("dars_rrf_k15", "dars_rrf_k50"):
            if method not in by[keys[0]]:
                continue
            entry["budgets"][method] = {}
            for b in budgets:
                same, d_recall, d_mrr, d_ndcg, clusters, below_fetch = [], [], [], [], [], []
                for k in keys:
                    s, d = by[k]["similarity"], by[k][method]
                    ns, nd = s["budgets"][str(b)]["n_units"], d["budgets"][str(b)]["n_units"]
                    same.append(set(s["ranked"][:ns]) == set(d["ranked"][:nd]))
                    below_fetch.append(min(ns, nd) < (15 if method.endswith("k15") else 50))
                    for arr, m in ((d_recall, "group_recall"), (d_mrr, "mrr"), (d_ndcg, "ndcg")):
                        arr.append(float(d["budgets"][str(b)][m]) - float(s["budgets"][str(b)][m]))
                    clusters.append(k[0])
                identical = float(np.mean(same))
                if not any(below_fetch):          # known answer: every shown set must equal similarity's
                    assert identical == 1.0, f"{source} {method} B={b}: sets differ although n_units >= fetch_k"
                entry["budgets"][method][str(b)] = {
                    "share_identical_shown_set": identical,
                    "share_questions_with_fewer_units_than_fetch_k": float(np.mean(below_fetch)),
                    "recall_diff": paired_bootstrap_diff(d_recall, [0.0] * len(d_recall), clusters, n_boot=n_boot),
                    "mrr_diff": paired_bootstrap_diff(d_mrr, [0.0] * len(d_mrr), clusters, n_boot=n_boot),
                    "ndcg_diff": paired_bootstrap_diff(d_ndcg, [0.0] * len(d_ndcg), clusters, n_boot=n_boot),
                    "by_construction": identical == 1.0 and not any(below_fetch),
                }
        sim5120 = float(np.mean([by[k]["similarity"]["budgets"]["5120"]["group_recall"] for k in keys]))
        pub = summary["similarity"]["group_recall@5120"]
        report["checks"].append(_check(f"{source} similarity recall@5120", sim5120,
                                       pub["mean"] if isinstance(pub, dict) else pub))
        report["sources"][source] = entry
    return report


# ═══════════════════════════════════════════════════════════════════════════════
#  L7, L8 — ALFWorld location ranking: tie-break sensitivity and the count prior
# ═══════════════════════════════════════════════════════════════════════════════


def _mrr(is_true_in_order: Sequence[bool]) -> float:
    for r, t in enumerate(is_true_in_order, 1):
        if t:
            return 1.0 / r
    return 0.0


def _order(keys: np.ndarray, sim: np.ndarray, tie: str, rng: np.random.Generator) -> List[int]:
    n = len(keys)
    if tie == "similarity":
        second = -sim
    else:
        second = rng.permutation(n).astype(float)
    return list(np.lexsort((second, -np.round(keys, 6))))


def alfworld_rank(n_boot: int = N_BOOT, seed: int = 0) -> Dict[str, Any]:
    from benchmarks.dars_eval.tuning import recompute_R

    report: Dict[str, Any] = {"splits": {}, "checks": []}
    published = {"test_in": {"selected": 0.5099860853432283, "count_prior": 0.626326530612245},
                 "test_out": {"selected": 0.4845331725102718, "count_prior": 0.6495425905731249}}
    lam, w = SELECTED["alfworld_h3"]
    for split in ("test_in", "test_out"):
        tasks = [r for r in _jsonl(TEST / "E10" / f"eval_{split}.jsonl") if r["reachable"] and len(r["candidates"]) > 1]
        rankers = ("similarity", "count_prior", "frequency_F", "utility_U", "selected_score", "success_count")
        per = {(name, tie): [] for name in rankers for tie in ("similarity", "random")}
        rngs = {tie: np.random.default_rng(seed) for tie in ("similarity", "random")}
        for t in tasks:
            c = t["candidates"]
            true = [x["is_true"] for x in c]
            sim = np.array([x["sim"] for x in c])
            rec = np.array([x["recency"] for x in c])
            F, U, P = (np.array([x[k] for x in c]) for k in ("F", "U", "P"))
            keys = {
                "similarity": sim,
                "count_prior": np.array([x["count_prior"] for x in c], dtype=float),
                "frequency_F": F,
                "utility_U": U,
                "selected_score": w[0] * recompute_R(rec, c[0]["now"], lam) + w[1] * F + w[2] * U + w[3] * P,
                "success_count": np.array([x["success"] for x in c], dtype=float),
            }
            for tie in ("similarity", "random"):
                for name in rankers:
                    per[(name, tie)].append(_mrr([true[i] for i in _order(keys[name], sim, tie, rngs[tie])]))
        entry: Dict[str, Any] = {"tasks": len(tasks), "mrr": {}, "paired_vs_similarity": {}}
        for (name, tie), v in per.items():
            entry["mrr"][f"{name}|{tie}"] = cluster_bootstrap_mean(v, n_boot=n_boot).as_dict()
            if name != "similarity":
                entry["paired_vs_similarity"][f"{name}|{tie}"] = paired_bootstrap_diff(
                    v, per[("similarity", "random")], n_boot=n_boot)
        entry["paired_count_prior_vs_selected"] = paired_bootstrap_diff(
            per[("count_prior", "similarity")], per[("selected_score", "similarity")], n_boot=n_boot)
        report["checks"].append(_check(f"{split} selected MRR (similarity tie-break)",
                                       float(np.mean(per[("selected_score", "similarity")])), published[split]["selected"]))
        report["splits"][split] = entry
    return report


# ═══════════════════════════════════════════════════════════════════════════════
#  L9 — ALFWorld eviction: what the needed memories are, and strong simple baselines
# ═══════════════════════════════════════════════════════════════════════════════


def alfworld_evict(n_boot: int = N_BOOT, seed: int = 0) -> Dict[str, Any]:
    from benchmarks.dars_eval.run_alfworld import _keep_mask
    from benchmarks.dars_eval.tuning import recompute_R

    snap = _jsonl(TEST / "E10" / "memories.jsonl")
    now = json.loads((TEST / "E10" / "manifest.json").read_text(encoding="utf-8"))["now"]
    idx = {m["pid"]: i for i, m in enumerate(snap)}
    rec = np.array([m["recency"] for m in snap])
    F, U, P = (np.array([m[k] for m in snap]) for k in ("F", "U", "P"))
    scores = {
        "selected_P_only": P,
        "default": 0.3 * recompute_R(rec, now, 0.005) + 0.2 * F + 0.3 * U + 0.2 * P,
        "recency_lru": rec,
        "fifo": np.array([m["created_at"] for m in snap]),
        "frequency_lfu": np.array([m["frequency"] for m in snap], dtype=float),
        "utility_U": U,
        "success_count": np.array([m["success"] for m in snap], dtype=float),
    }
    report: Dict[str, Any] = {"memories": len(snap), "splits": {}, "checks": []}
    published = {"test_in": 0.09642857142857142, "test_out": 0.13805970149253732}
    kinds = sorted({m["kind"] for m in snap})
    for split in ("test_in", "test_out"):
        rows = _jsonl(TEST / "E10" / f"eval_{split}.jsonl")
        needed = [nd for nd in ([idx[p] for p in r["needed"] if p in idx] for r in rows if r["needed"]) if nd]
        needed_set = {i for nd in needed for i in nd}
        age_h = (now - np.array([m["created_at"] for m in snap])) / 3600.0
        comp = {}
        for label, members in (("needed", sorted(needed_set)), ("not_needed", [i for i in range(len(snap)) if i not in needed_set])):
            members = np.array(members)
            comp[label] = {
                "count": int(len(members)),
                "kind_share": {k: float(np.mean([snap[i]["kind"] == k for i in members])) for k in kinds},
                "median_age_hours": float(np.median(age_h[members])),
                "median_hours_since_last_access": float(np.median((now - rec[members]) / 3600.0)),
                "mean_frequency": float(np.mean(scores["frequency_lfu"][members])),
                "mean_P": float(np.mean(P[members])),
            }
        entry: Dict[str, Any] = {"tasks_with_needed": len(needed), "composition": comp, "harmful_deletion": {}}
        for keep in (0.25, 0.5, 0.75):
            n_keep = int(math.ceil(keep * len(snap)))
            rates = {}
            for name, s in scores.items():
                kept = _keep_mask(s, n_keep, seed)
                rates[name] = [float(np.mean(~kept[nd])) for nd in needed]
            random_rates = []
            for rs in range(5):
                kept = _keep_mask(np.zeros(len(snap)), n_keep, 1000 + rs)
                random_rates.append([float(np.mean(~kept[nd])) for nd in needed])
            rates["random_mean_of_5"] = list(np.mean(np.array(random_rates), axis=0))
            entry["harmful_deletion"][str(keep)] = {
                name: {**cluster_bootstrap_mean(v, n_boot=n_boot).as_dict(),
                       "selected_minus_this": paired_bootstrap_diff(rates["selected_P_only"], v, n_boot=n_boot)}
                for name, v in rates.items()}
            if keep == 0.5:
                report["checks"].append(_check(f"{split} selected harmful deletion", float(np.mean(rates["selected_P_only"])),
                                               published[split]))
        report["splits"][split] = entry
    return report


# ═══════════════════════════════════════════════════════════════════════════════
#  L11, L12, L14, L15 — MSC: strongest baselines, all labels, clustered intervals
# ═══════════════════════════════════════════════════════════════════════════════


def _msc_rows() -> List[Dict[str, Any]]:
    return [r for r in _jsonl(TEST / "E9" / "facts.jsonl") if r["split"] == "test"]


def _msc_scores(rows: Sequence[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    from benchmarks.dars_eval.retention_signals import dars_read_score

    sel = json.loads(Path("benchmark_runs/revision/addendum/selection_dev.json").read_text(encoding="utf-8"))
    mm = sel["metadata_model"]
    feats = np.array([[2 - r["created_session"], r["mentions"]] for r in rows], dtype=float)
    return {
        "read_h4": dars_read_score(rows, *SELECTED["msc_h4"]),
        "read_h5": dars_read_score(rows, *SELECTED["msc_h5"]),
        "default": dars_read_score(rows, *SELECTED["default"]),
        "recency_lru": np.array([r["recency"] for r in rows]),
        "fifo": np.array([r["created_at"] for r in rows]),
        "frequency_lfu": np.array([r["frequency"] for r in rows], dtype=float),
        "utility": np.array([r["components"]["U"] for r in rows]),
        "mention_count": np.array([r["mentions"] for r in rows], dtype=float),
        "metadata_model_dev_fit": 1 / (1 + np.exp(-(mm["intercept"] + feats @ np.array(mm["coef"])))),
    }


def msc(n_boot_auc: int = 2000, n_boot: int = N_BOOT) -> Dict[str, Any]:
    from benchmarks.dars_eval.run_msc import eviction

    rows = _msc_rows()
    scores = _msc_scores(rows)
    clusters = [r["dialogue"] for r in rows]
    report: Dict[str, Any] = {"facts": len(rows), "dialogues": len(set(clusters)), "labels": {}, "checks": []}
    for label in ("lex_0.5", "lex_0.6", "lex_0.7", "embed_0.8"):
        y = np.array([r["labels"][label] for r in rows])
        entry: Dict[str, Any] = {"base_rate": float(y.mean()), "auroc": {}, "vs_read_h4": {}, "harmful_deletion": {}}
        for name, s in scores.items():
            entry["auroc"][name] = {"delong": auroc_delong(s, y),
                                    "clustered": auroc_cluster_bootstrap(s, y, clusters, n_boot=n_boot_auc)}
            if name != "read_h4":
                entry["vs_read_h4"][name] = auroc_cluster_bootstrap_paired(scores["read_h4"], s, y, clusters,
                                                                           n_boot=n_boot_auc)
        for keep in (0.25, 0.5, 0.75):
            ev_sel = eviction(rows, scores["read_h5"], label, keep)
            entry["harmful_deletion"][str(keep)] = {}
            for name, s in scores.items():
                ev = eviction(rows, s, label, keep)
                entry["harmful_deletion"][str(keep)][name] = {
                    **cluster_bootstrap_mean(ev["harmful_deletion_rate"], ev["dialogues"], n_boot=n_boot).as_dict(),
                    "read_h5_minus_this": paired_bootstrap_diff(ev_sel["harmful_deletion_rate"],
                                                                ev["harmful_deletion_rate"], ev["dialogues"], n_boot=n_boot)}
        if label == "lex_0.5":
            report["checks"].append(_check("MSC read_h4 AUROC", entry["auroc"]["read_h4"]["delong"]["auc"], 0.890635766021679))
            report["checks"].append(_check("MSC read_h5 harmful deletion",
                                           entry["harmful_deletion"]["0.5"]["read_h5"]["mean"], 0.0944414655719934))
        report["labels"][label] = entry
    return report


# ═══════════════════════════════════════════════════════════════════════════════
#  P1 — cross-domain weight transfer (offline datasets)
# ═══════════════════════════════════════════════════════════════════════════════


def transfer(n_boot: int = 5000, seed: int = 0) -> Dict[str, Any]:
    from benchmarks.dars_eval.retention_signals import dars_read_score
    from benchmarks.dars_eval.run_alfworld import _fused_order, _keep_mask
    from benchmarks.dars_eval.run_msc import eviction
    from benchmarks.dars_eval.tuning import recompute_R

    report: Dict[str, Any] = {"configs": {k: {"lambda": v[0], "weights": list(v[1])} for k, v in SELECTED.items()},
                              "msc": {}, "alfworld": {},
                              "note": "LongMemEval eviction needs stream re-runs per weight vector; reported separately"}
    rows = _msc_rows()
    y = np.array([r["labels"]["lex_0.5"] for r in rows])
    for name, (lam, w) in SELECTED.items():
        s = dars_read_score(rows, lam, w)
        ev = eviction(rows, s, "lex_0.5", 0.5)
        report["msc"][name] = {"auroc": auroc_delong(s, y)["auc"],
                               "harmful_deletion": cluster_bootstrap_mean(ev["harmful_deletion_rate"], ev["dialogues"],
                                                                          n_boot=n_boot).as_dict()}
    snap = _jsonl(TEST / "E10" / "memories.jsonl")
    now = json.loads((TEST / "E10" / "manifest.json").read_text(encoding="utf-8"))["now"]
    idx = {m["pid"]: i for i, m in enumerate(snap)}
    rec = np.array([m["recency"] for m in snap])
    F, U, P = (np.array([m[k] for m in snap]) for k in ("F", "U", "P"))
    for split in ("test_in", "test_out"):
        erows = _jsonl(TEST / "E10" / f"eval_{split}.jsonl")
        tasks = [r for r in erows if r["reachable"] and len(r["candidates"]) > 1]
        needed = [nd for nd in ([idx[p] for p in r["needed"] if p in idx] for r in erows if r["needed"]) if nd]
        n_keep = int(math.ceil(0.5 * len(snap)))
        report["alfworld"][split] = {}
        for name, (lam, w) in SELECTED.items():
            mrr = []
            for t in tasks:
                c = t["candidates"]
                sim = np.array([x["sim"] for x in c])
                S = (w[0] * recompute_R(np.array([x["recency"] for x in c]), c[0]["now"], lam)
                     + w[1] * np.array([x["F"] for x in c]) + w[2] * np.array([x["U"] for x in c])
                     + w[3] * np.array([x["P"] for x in c]))
                mrr.append(_mrr([c[i]["is_true"] for i in _fused_order(sim, S, "score_only", 0.0)]))
            S_all = w[0] * recompute_R(rec, now, lam) + w[1] * F + w[2] * U + w[3] * P
            kept = _keep_mask(S_all, n_keep, seed)
            harm = [float(np.mean(~kept[nd])) for nd in needed]
            report["alfworld"][split][name] = {"location_mrr_score_only": cluster_bootstrap_mean(mrr, n_boot=n_boot).as_dict(),
                                               "harmful_deletion": cluster_bootstrap_mean(harm, n_boot=n_boot).as_dict()}
    return report


# ═══════════════════════════════════════════════════════════════════════════════
#  P2 — test-split robustness to weights (Dirichlet) and to the decay rate
# ═══════════════════════════════════════════════════════════════════════════════


def robustness(draws_msc: int = 1000, draws_alf: int = 200, seed: int = 0) -> Dict[str, Any]:
    from benchmarks.dars_eval.run_alfworld import _fused_order, _keep_mask
    from benchmarks.dars_eval.tuning import LAMBDAS, auc_rows, dirichlet_around, recompute_R

    report: Dict[str, Any] = {"msc": {}, "alfworld": {}}
    rows = _msc_rows()
    y = np.array([r["labels"]["lex_0.5"] for r in rows])
    rec = np.array([r["recency"] for r in rows])
    t_end = np.array([r["t_end"] for r in rows])
    Fm, Um, Pm = (np.array([r["components"][k] for r in rows]) for k in ("F", "U", "P"))
    base = {"recency_lru": auroc_delong(rec, y)["auc"],
            "fifo": auroc_delong(np.array([r["created_at"] for r in rows]), y)["auc"],
            "utility": auroc_delong(Um, y)["auc"]}
    for name in ("msc_h4", "default"):
        lam, w = SELECTED[name]
        comps = np.vstack([recompute_R(rec, t_end, lam), Fm, Um, Pm])
        d = dirichlet_around(w, draws=draws_msc, seed=seed)
        a = auc_rows(d @ comps, y)
        report["msc"][name] = {
            "dirichlet_auroc": {"mean": float(a.mean()), "p05": float(np.quantile(a, 0.05)), "p95": float(np.quantile(a, 0.95))},
            "rank_reversal_rate": {k: float(np.mean(a < v)) for k, v in base.items()},
            "lambda_sweep_auroc": {str(l): float(auc_rows(np.atleast_2d(np.asarray(w) @ np.vstack([recompute_R(rec, t_end, l), Fm, Um, Pm])), y)[0])
                                   for l in LAMBDAS},
        }
    report["msc"]["baselines_auroc"] = base

    snap = _jsonl(TEST / "E10" / "memories.jsonl")
    now = json.loads((TEST / "E10" / "manifest.json").read_text(encoding="utf-8"))["now"]
    idx = {m["pid"]: i for i, m in enumerate(snap)}
    recs = np.array([m["recency"] for m in snap])
    F, U, P = (np.array([m[k] for m in snap]) for k in ("F", "U", "P"))
    for split in ("test_in", "test_out"):
        erows = _jsonl(TEST / "E10" / f"eval_{split}.jsonl")
        tasks = [r for r in erows if r["reachable"] and len(r["candidates"]) > 1]
        needed = [nd for nd in ([idx[p] for p in r["needed"] if p in idx] for r in erows if r["needed"]) if nd]
        sim_mrr = float(np.mean([_mrr([c["is_true"] for c in sorted(t["candidates"], key=lambda x: -x["sim"])]) for t in tasks]))
        entry: Dict[str, Any] = {"similarity_mrr": sim_mrr}
        for name in ("alfworld_h3", "default"):
            lam, w = SELECTED[name]
            vals = []
            for wd in dirichlet_around(w, draws=draws_alf, seed=seed):
                m = []
                for t in tasks:
                    c = t["candidates"]
                    sim = np.array([x["sim"] for x in c])
                    S = (wd[0] * recompute_R(np.array([x["recency"] for x in c]), c[0]["now"], lam)
                         + wd[1] * np.array([x["F"] for x in c]) + wd[2] * np.array([x["U"] for x in c])
                         + wd[3] * np.array([x["P"] for x in c]))
                    m.append(_mrr([c[i]["is_true"] for i in _fused_order(sim, S, "score_only", 0.0)]))
                vals.append(float(np.mean(m)))
            v = np.array(vals)
            entry[f"h3_dirichlet_{name}"] = {"mean": float(v.mean()), "p05": float(np.quantile(v, 0.05)),
                                             "p95": float(np.quantile(v, 0.95)),
                                             "share_below_similarity": float(np.mean(v < sim_mrr))}
        n_keep = int(math.ceil(0.5 * len(snap)))
        lfu = float(np.mean([np.mean(~_keep_mask(np.array([m["frequency"] for m in snap], dtype=float), n_keep, 0)[nd]) for nd in needed]))
        for name in ("alfworld_h5", "default"):
            lam, w = SELECTED[name]
            harms = []
            for wd in dirichlet_around(w, draws=draws_alf, seed=seed):
                S = wd[0] * recompute_R(recs, now, lam) + wd[1] * F + wd[2] * U + wd[3] * P
                kept = _keep_mask(S, n_keep, 0)
                harms.append(float(np.mean([np.mean(~kept[nd]) for nd in needed])))
            h = np.array(harms)
            entry[f"h5_dirichlet_{name}"] = {"mean": float(h.mean()), "p05": float(np.quantile(h, 0.05)),
                                             "p95": float(np.quantile(h, 0.95)), "share_worse_than_lfu": float(np.mean(h > lfu)),
                                             "lfu": lfu}
        report["alfworld"][split] = entry
    return report


# ═══════════════════════════════════════════════════════════════════════════════
#  L20 — need persistence
# ═══════════════════════════════════════════════════════════════════════════════


def _mab_persistence(source: str) -> Dict[str, Any]:
    from benchmarks.dars_eval.datasets import load_contexts
    from benchmarks.dars_eval.memory_units import WindowGuard

    contexts = load_contexts(source, WindowGuard())
    later_given_earlier, n_pairs, evidence_units, reused = [], 0, 0, 0
    for ctx in contexts:
        order = list(range(len(ctx.questions)))
        if ctx.question_times:
            order.sort(key=lambda q: ctx.question_times[q])
        seen: set = set()
        for q in order:
            units = {u for g in ctx.evidence[q] for u in g}
            if not units:
                continue
            evidence_units += len(units)
            reused += len(units & seen)
            seen |= units
        # P(a unit that was evidence for some question is evidence for another, later one)
        count = defaultdict(int)
        for q in order:
            for u in {u for g in ctx.evidence[q] for u in g}:
                count[u] += 1
        if count:
            later_given_earlier.append(float(np.mean([c > 1 for c in count.values()])))
            n_pairs += len(count)
    return {"evidence_units_distinct": n_pairs,
            "share_of_evidence_units_needed_by_more_than_one_question": float(np.mean(later_given_earlier)) if later_given_earlier else None,
            "share_of_question_evidence_already_needed_earlier": reused / evidence_units if evidence_units else None}


def persistence() -> Dict[str, Any]:
    report: Dict[str, Any] = {"mab": {}, "msc": {}, "alfworld": {}}
    for source in ("longmemeval_s*", "ruler_qa1_197K", "ruler_qa2_421K", "factconsolidation_sh_32k", "factconsolidation_mh_32k"):
        report["mab"][source] = _mab_persistence(source)
    # MSC, from the dev rows with write histories (dev data): restated in session 2 -> restated in session 3
    dev = _jsonl(Path("benchmark_runs/revision/addendum/msc_dev_writes/facts.jsonl"))
    before = [r for r in dev if r["created_session"] < 2]
    restated2 = np.array([2 in r["mention_sessions"] for r in before])
    y = np.array([r["labels"]["lex_0.5"] for r in before])
    report["msc"] = {"split": "dev", "facts_created_before_session_2": len(before),
                     "p_needed_given_restated_in_s2": float(y[restated2].mean()),
                     "p_needed_given_not_restated_in_s2": float(y[~restated2].mean()),
                     "base_rate": float(y.mean())}
    # ALFWorld: is a memory that succeeded during the training stream needed by a held-out task?
    snap = _jsonl(TEST / "E10" / "memories.jsonl")
    idx = {m["pid"]: i for i, m in enumerate(snap)}
    for split in ("test_in", "test_out"):
        needed = {idx[p] for r in _jsonl(TEST / "E10" / f"eval_{split}.jsonl") for p in (r["needed"] or []) if p in idx}
        succ = np.array([m["success"] > 0 for m in snap])
        need = np.array([i in needed for i in range(len(snap))])
        report["alfworld"][split] = {"p_needed_given_training_success": float(need[succ].mean()) if succ.any() else None,
                                     "p_needed_given_no_training_success": float(need[~succ].mean()),
                                     "base_rate": float(need.mean())}
    return report


# ═══════════════════════════════════════════════════════════════════════════════
#  L17 — LongMemEval: how old is the evidence when the question is asked?
# ═══════════════════════════════════════════════════════════════════════════════


def lme_age(n_boot: int = N_BOOT) -> Dict[str, Any]:
    from benchmarks.dars_eval.datasets import load_contexts
    from benchmarks.dars_eval.memory_units import WindowGuard
    from benchmarks.dars_eval.splits import question_splits

    contexts = load_contexts("longmemeval_s*", WindowGuard())
    pct_evidence, newest_half, clusters, oldest_evidence_pct = [], [], [], []
    for ctx in contexts:
        splits = question_splits("longmemeval_s*", ctx.index, len(ctx.questions))
        ts = np.array([u.timestamp for u in ctx.units], dtype=float)
        for q, sp in enumerate(splits):
            units = sorted({u for g in ctx.evidence[q] for u in g})
            if sp != "test" or not units or not ctx.question_times:
                continue
            qt = ctx.question_times[q]
            stored = np.flatnonzero(ts <= qt)
            if len(stored) < 2:
                continue
            order = stored[np.argsort(ts[stored], kind="stable")]       # oldest first
            rank = {u: i / (len(order) - 1) for i, u in enumerate(order)}
            ev = [rank[u] for u in units if u in rank]
            if not ev:
                continue
            pct_evidence.append(float(np.mean(ev)))                    # 0 = oldest stored, 1 = newest
            oldest_evidence_pct.append(float(min(ev)))
            newest_half.append(float(np.mean([p >= 0.5 for p in ev])))
            clusters.append(ctx.index)
    return {"questions": len(pct_evidence),
            "mean_age_percentile_of_evidence (0 oldest, 1 newest)": cluster_bootstrap_mean(pct_evidence, clusters, n_boot=n_boot).as_dict(),
            "share_of_evidence_in_newest_half": cluster_bootstrap_mean(newest_half, clusters, n_boot=n_boot).as_dict(),
            "oldest_evidence_percentile": cluster_bootstrap_mean(oldest_evidence_pct, clusters, n_boot=n_boot).as_dict(),
            "interpretation_rule": "FIFO keeps the newest half; if evidence sits mostly in the newest half, FIFO is near-optimal"}


# ═══════════════════════════════════════════════════════════════════════════════
#  K3 — prior-art retention baselines on ALFWorld eviction (Generative Agents, MemoryBank)
# ═══════════════════════════════════════════════════════════════════════════════


def alfworld_prior_art(n_boot: int = N_BOOT, seed: int = 0) -> Dict[str, Any]:
    """Eviction by Generative Agents and MemoryBank scores on a re-run snapshot that carries memory text.

    The re-run (``E13_audit/alfworld_text``) reproduces the published P-only harmful deletion exactly; its
    stream differs from the pre-registered one by 4 of 37,822 adjudications (embedding-cache precision).
    """
    from benchmarks.dars_eval.importance import load_ratings
    from benchmarks.dars_eval.run_alfworld import _keep_mask
    from benchmarks.dars_eval.tuning import recompute_R

    run = OUT / "alfworld_text"
    snap = _jsonl(run / "memories.jsonl")
    now = json.loads((run / "manifest.json").read_text(encoding="utf-8"))["now"]
    ratings = load_ratings(OUT / "importance_lme_alfworld.jsonl")
    idx = {m["pid"]: i for i, m in enumerate(snap)}
    rec = np.array([m["recency"] for m in snap])
    F, U, P = (np.array([m[k] for m in snap]) for k in ("F", "U", "P"))
    hours = np.maximum(now - rec, 0.0) / 3600.0
    imp = np.array([float(ratings[m["text"]]) for m in snap])
    mm = lambda v: np.zeros_like(v) if v.max() - v.min() <= 0 else (v - v.min()) / (v.max() - v.min())
    scores = {
        "selected_P_only": P,
        "default": 0.3 * recompute_R(rec, now, 0.005) + 0.2 * F + 0.3 * U + 0.2 * P,
        "generative_agents": mm(0.995 ** hours) + mm(imp),
        "importance_only": imp,
        "memorybank": np.exp(-(hours / 24.0) / (1.0 + np.array([m["frequency"] for m in snap], dtype=float))),
        "frequency_lfu": np.array([m["frequency"] for m in snap], dtype=float),
        "fifo": np.array([m["created_at"] for m in snap]),
    }
    report: Dict[str, Any] = {"snapshot": str(run), "memories": len(snap), "splits": {}, "checks": []}
    published = {"test_in": 0.09642857142857142, "test_out": 0.13805970149253732}
    for split in ("test_in", "test_out"):
        rows = _jsonl(run / f"eval_{split}.jsonl")
        needed = [nd for nd in ([idx[p] for p in r["needed"] if p in idx] for r in rows if r["needed"]) if nd]
        entry: Dict[str, Any] = {}
        for keep in (0.25, 0.5, 0.75):
            n_keep = int(math.ceil(keep * len(snap)))
            rates = {name: [float(np.mean(~_keep_mask(s, n_keep, seed)[nd])) for nd in needed] for name, s in scores.items()}
            entry[str(keep)] = {name: {**cluster_bootstrap_mean(v, n_boot=n_boot).as_dict(),
                                       "selected_minus_this": paired_bootstrap_diff(rates["selected_P_only"], v, n_boot=n_boot)}
                                for name, v in rates.items()}
            if keep == 0.5:
                report["checks"].append(_check(f"{split} P-only harmful deletion on the re-run snapshot",
                                               float(np.mean(rates["selected_P_only"])), published[split]))
        report["splits"][split] = entry
    return report


def msc_writes(n_boot_auc: int = 2000, n_boot: int = N_BOOT) -> Dict[str, Any]:
    """K5 — write-side vs read-side scores on the original MSC test dialogues (exploratory: already seen).

    Uses the addendum's frozen configurations and scoring code (``confirm_msc.all_scores``) on
    ``E13_audit/msc_test_writes``, a re-run of the E9 test stream with write histories recorded (its rows
    equal the pre-registered E9 rows apart from the new fields).
    """
    from benchmarks.dars_eval.confirm_msc import FAMILY, all_scores, compare_auroc, compare_harm
    from benchmarks.dars_eval.importance import load_ratings

    rows = _jsonl(OUT / "msc_test_writes" / "facts.jsonl")
    cfg = json.loads(Path("experiments/addendum_config.json").read_text(encoding="utf-8"))
    importance = load_ratings(Path("benchmark_runs/revision/addendum/importance.jsonl"))
    scores = all_scores(rows, cfg, importance)
    clusters = [r["dialogue"] for r in rows]
    report: Dict[str, Any] = {"facts": len(rows), "dialogues": len(set(clusters)), "labels": {}, "checks": []}
    for label in ("lex_0.5", "lex_0.6", "lex_0.7", "embed_0.8"):
        y = np.array([r["labels"][label] for r in rows])
        entry: Dict[str, Any] = {"auroc": {k: auroc_delong(s, y)["auc"] for k, s in scores.items()}, "comparisons": {}}
        for cid, endpoint, a, b, sign in FAMILY:
            entry["comparisons"][cid] = (compare_auroc(scores[a], scores[b], y, clusters, n_boot_auc) if endpoint == "auroc"
                                         else compare_harm(rows, scores[a], scores[b], label, 0.5, n_boot))
            entry["comparisons"][cid].update({"a": a, "b": b, "predicted_sign": sign})
        report["labels"][label] = entry
    report["checks"].append(_check("MSC read_h4 AUROC on the write-history re-run",
                                   report["labels"]["lex_0.5"]["auroc"]["read_h4"], 0.890635766021679))
    return report


# ═══════════════════════════════════════════════════════════════════════════════
#  L4 — robustness to a stronger embedder (same memory units and labels)
# ═══════════════════════════════════════════════════════════════════════════════

STRONG_EMBEDDER = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def embedder_robustness(n_boot: int = N_BOOT) -> Dict[str, Any]:
    """Do the static-retrieval and precedence conclusions depend on MiniLM?

    The memory units and evidence labels are exactly those of the study (built for MiniLM's window);
    only the vectors change. BM25 does not depend on the embedder. For FactConsolidation the
    lifecycle signal is added the way DARS adds recency: reciprocal-rank fusion of the similarity rank
    and the recency rank over the 50 nearest facts, similarity order beyond.
    """
    import tiktoken
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer

    from benchmarks.dars_eval.datasets import load_contexts
    from benchmarks.dars_eval.memory_units import WindowGuard
    from benchmarks.dars_eval.rankers import bm25_tokens
    from benchmarks.dars_eval.retrieval_eval import MEMORY_HEADER_TOKENS, cut_to_budget, evidence_metrics, precedence
    from benchmarks.dars_eval.splits import question_splits

    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    guard = WindowGuard()
    minilm = guard.embedder
    bge = SentenceTransformer(STRONG_EMBEDDER, device="cpu")
    report: Dict[str, Any] = {"strong_embedder": STRONG_EMBEDDER, "sources": {}}

    def order_by(scores: np.ndarray) -> List[int]:
        return list(np.argsort(-scores, kind="stable"))

    def fused_recency(sim: np.ndarray, ts: np.ndarray, k: int = 50, rrf_k: int = 60) -> List[int]:
        by_sim = order_by(sim)
        top = by_sim[:k]
        rec_rank = {u: r for r, u in enumerate(sorted(top, key=lambda u: (-ts[u], u)), 1)}
        sim_rank = {u: r for r, u in enumerate(top, 1)}
        fused = sorted(top, key=lambda u: -(1 / (rrf_k + sim_rank[u]) + 1 / (rrf_k + rec_rank[u])))
        return fused + by_sim[k:]

    for source in ("ruler_qa1_197K", "ruler_qa2_421K", "longmemeval_s*", "factconsolidation_sh_32k", "factconsolidation_mh_32k"):
        contexts = load_contexts(source, guard)
        per: Dict[str, List[float]] = defaultdict(list)
        clusters: Dict[str, List[int]] = defaultdict(list)
        for ctx in contexts:
            texts = [u.text for u in ctx.units]
            tokens = [len(enc.encode(t)) for t in texts]
            V_m = np.asarray(minilm.encode_batch(texts, batch_size=64), dtype=np.float32)
            V_m /= np.linalg.norm(V_m, axis=1, keepdims=True)
            V_b = np.asarray(bge.encode(texts, batch_size=64, normalize_embeddings=True), dtype=np.float32)
            bm = BM25Okapi([bm25_tokens(t) or ["_"] for t in texts])
            ts = np.array([u.timestamp if u.timestamp is not None else 0.0 for u in ctx.units], dtype=float)
            splits = question_splits(source, ctx.index, len(ctx.questions))
            labels = ctx.extra.get("fc_labels")
            unit_of_serial = {u.meta["serial"]: i for i, u in enumerate(ctx.units) if "serial" in u.meta}
            q_test = [q for q, sp in enumerate(splits) if sp == "test" and ctx.evidence[q]]
            if not q_test:
                continue
            Q_m = np.asarray(minilm.encode_batch([ctx.queries[q] for q in q_test]), dtype=np.float32)
            Q_m /= np.linalg.norm(Q_m, axis=1, keepdims=True)
            Q_b = np.asarray(bge.encode([BGE_QUERY_PREFIX + ctx.queries[q] for q in q_test], normalize_embeddings=True))
            for j, q in enumerate(q_test):
                rankings = {
                    "similarity_minilm": order_by(V_m @ Q_m[j]),
                    "similarity_bge": order_by(V_b @ Q_b[j]),
                    "bm25": order_by(np.asarray(bm.get_scores(bm25_tokens(ctx.queries[q])))),
                }
                if labels is not None:
                    rankings["recency_fused_minilm"] = fused_recency(V_m @ Q_m[j], ts)
                    rankings["recency_fused_bge"] = fused_recency(V_b @ Q_b[j], ts)
                for name, ranked in rankings.items():
                    budgets = (256,) if labels is not None else (1024, 2048, 5120)
                    for b in budgets:
                        ev = evidence_metrics(cut_to_budget(ranked, tokens, b), ctx.evidence[q])
                        per[f"{name}|recall@{b}"].append(ev["group_recall"])
                        clusters[f"{name}|recall@{b}"].append(ctx.index)
                    if labels is None:
                        ev = evidence_metrics(cut_to_budget(ranked, tokens, 5120), ctx.evidence[q])
                        per[f"{name}|mrr@5120"].append(ev["mrr"])
                        clusters[f"{name}|mrr@5120"].append(ctx.index)
                    elif labels[q] and labels[q]["gold_is_newest"]:
                        lab = labels[q]
                        p = precedence(ranked, unit_of_serial[lab["gold_serial"]],
                                       [unit_of_serial[s] for s in lab["conflict_serials"] if s in unit_of_serial])
                        if p is not None:
                            per[f"{name}|precedence_rank"].append(p)
                            clusters[f"{name}|precedence_rank"].append(ctx.index)
        entry: Dict[str, Any] = {"means": {k: cluster_bootstrap_mean(v, clusters[k], n_boot=n_boot).as_dict() for k, v in per.items()},
                                 "contrasts": {}}
        metrics = sorted({k.split("|", 1)[1] for k in per})
        for metric in metrics:
            pairs = [("similarity_bge", "similarity_minilm"), ("bm25", "similarity_bge"), ("bm25", "similarity_minilm"),
                     ("recency_fused_bge", "similarity_bge"), ("recency_fused_bge", "bm25")]
            for a, b in pairs:
                ka, kb = f"{a}|{metric}", f"{b}|{metric}"
                if ka in per and kb in per and len(per[ka]) == len(per[kb]):
                    entry["contrasts"][f"{a} - {b} | {metric}"] = paired_bootstrap_diff(per[ka], per[kb], clusters[ka], n_boot=n_boot)
        report["sources"][source] = entry
    return report


COMMANDS = {"h1": h1, "alfworld-rank": alfworld_rank, "alfworld-evict": alfworld_evict, "msc": msc,
            "embedder-robustness": embedder_robustness,
            "alfworld-prior-art": alfworld_prior_art, "msc-writes": msc_writes,
            "transfer": transfer, "robustness": robustness, "persistence": persistence, "lme-age": lme_age}


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Fourth-audit secondary analyses on test-phase artifacts")
    p.add_argument("command", choices=sorted(COMMANDS))
    args = p.parse_args(argv)
    report = COMMANDS[args.command]()
    _write(args.command.replace("-", "_"), report)
    for c in report.get("checks", []):
        print(f"  known-answer ok: {c['check']} = {c['reproduced']:.4f}")


if __name__ == "__main__":
    main()
