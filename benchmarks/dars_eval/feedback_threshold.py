"""How accurate must feedback be before a learned utility signal is worth having?

E5 measures the deployed judge against ground truth: Cohen's kappa 0.263, accuracy 0.634 on the
test split. E9 and E10 show that utility is the component that makes the retention score work.
This joins the two questions: it degrades the feedback that produced each memory's success and
failure counts, and reports the error rate at which the retention advantage disappears.

Why this can be done offline and exactly. In E9 and E10 retrieval is similarity for *every*
policy (pre-registration section 3), so a wrong verdict cannot change which memories were
retrieved - it only changes the verdict recorded against them. Each verdict is independent, and
utility depends on the counts alone, so flipping a fraction ``eps`` of verdicts is exactly a
binomial resample of the stored counts:

    s' = (s - Binomial(s, eps)) + Binomial(f, eps),    f' = (s + f) - s'

No stream has to be re-run and no API call is made. Recency, frequency, FIFO and LRU do not use
verdicts at all, so they are flat reference lines: where the DARS curve crosses them is the
answer.

    python -m benchmarks.dars_eval.feedback_threshold --run-msc benchmark_runs/revision/test/E9 \
        --run-alfworld benchmark_runs/revision/test/E10 --out benchmark_runs/revision/test/E12
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from benchmarks.dars_eval.run_alfworld import _keep_mask
from benchmarks.dars_eval.run_msc import dars_score, eviction
from benchmarks.dars_eval.stats import auroc_delong

EPS_GRID = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
SEEDS = (0, 1, 2, 3, 4)


def resample_counts(success: np.ndarray, failure: np.ndarray, eps: float,
                    rng: np.random.Generator) -> np.ndarray:
    """Successes left after flipping each verdict independently with probability ``eps``."""
    success = np.asarray(success, dtype=np.int64)
    failure = np.asarray(failure, dtype=np.int64)
    if eps <= 0:
        return success.copy()
    kept = success - rng.binomial(success, eps)
    gained = rng.binomial(failure, eps)
    return kept + gained


def utility(success: np.ndarray, failure: np.ndarray) -> np.ndarray:
    """Laplace-smoothed success rate, as the vault stores it."""
    return (success + 1.0) / (success + failure + 2.0)


def _rows(path: Path, split: Optional[str] = None) -> List[Dict[str, Any]]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if split in (None, "all") or r.get("split") == split]


def msc_curve(run_dir: Path, split: str, h4: Sequence[float], lam4: float,
              h5: Sequence[float], lam5: float, label: str = "lex_0.5", keep: float = 0.5,
              eps_grid: Sequence[float] = EPS_GRID, seeds: Sequence[int] = SEEDS) -> Dict[str, Any]:
    """H4 AUROC and H5 harmful deletion as feedback degrades (MSC)."""
    rows = _rows(run_dir / "facts.jsonl", split)
    y = np.array([r["labels"][label] for r in rows])
    success = np.array([r["success"] for r in rows])
    failure = np.array([r["failure"] for r in rows])
    total = success + failure
    comp = [r["components"] for r in rows]
    r4 = np.array([math.exp(-lam4 * max(r["t_end"] - r["recency"], 0.0) / 3600.0) for r in rows])
    r5 = np.array([math.exp(-lam5 * max(r["t_end"] - r["recency"], 0.0) / 3600.0) for r in rows])

    def score(u: np.ndarray, weights: Sequence[float], R: np.ndarray) -> np.ndarray:
        return np.array([dars_score({**c, "R": R[i], "U": u[i]}, weights) for i, c in enumerate(comp)])

    baselines = {"recency_lru": np.array([r["recency"] for r in rows]),
                 "fifo": np.array([r["created_at"] for r in rows]),
                 "frequency_lfu": np.array([r["frequency"] for r in rows], dtype=float)}
    out: Dict[str, Any] = {
        "dataset": "MSC", "split": split, "facts": len(rows), "label": label, "keep": keep,
        "verdicts_per_memory": float(total.mean()),
        "baselines": {k: {"auroc": auroc_delong(v, y)["auc"],
                          "harmful_deletion": float(np.mean(eviction(rows, v, label, keep)["harmful_deletion_rate"]))}
                      for k, v in baselines.items()},
        "curve": [],
    }
    for eps in eps_grid:
        aurocs, harms = [], []
        for seed in seeds:
            rng = np.random.default_rng(1000 + seed)
            s2 = resample_counts(success, failure, eps, rng)
            u = utility(s2, total - s2)
            aurocs.append(auroc_delong(score(u, h4, r4), y)["auc"])
            harms.append(float(np.mean(eviction(rows, score(u, h5, r5), label, keep)["harmful_deletion_rate"])))
            if eps == 0:            # deterministic: one seed is enough
                break
        out["curve"].append({"eps": eps, "auroc": float(np.mean(aurocs)), "auroc_sd": float(np.std(aurocs)),
                             "harmful_deletion": float(np.mean(harms)), "harmful_sd": float(np.std(harms))})
    return out


def alfworld_curve(run_dir: Path, split: str, h5: Sequence[float], lam5: float, keep: float = 0.5,
                   eps_grid: Sequence[float] = EPS_GRID,
                   seeds: Sequence[int] = SEEDS) -> Dict[str, Any]:
    """H5 harmful deletion as feedback degrades (ALFWorld)."""
    snapshot = _rows(run_dir / "memories.jsonl")
    tasks = _rows(run_dir / f"eval_{split}.jsonl")
    now = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["now"]
    index = {m["pid"]: i for i, m in enumerate(snapshot)}
    needed = [nd for nd in ([index[p] for p in t["needed"] if p in index] for t in tasks if t.get("needed")) if nd]
    success = np.array([m["success"] for m in snapshot])
    failure = np.array([m["failure"] for m in snapshot])
    total = success + failure
    F = np.array([m["F"] for m in snapshot])
    P = np.array([m["P"] for m in snapshot])
    R = np.exp(-lam5 * np.maximum(now - np.array([m["recency"] for m in snapshot]), 0.0) / 3600.0)
    n_keep = int(math.ceil(keep * len(snapshot)))

    def harmful(scores: np.ndarray, seed: int = 0) -> float:
        kept = _keep_mask(scores, n_keep, seed)
        return float(np.mean([float(np.mean(~kept[nd])) for nd in needed]))

    baselines = {"recency_lru": np.array([m["recency"] for m in snapshot]),
                 "fifo": np.array([m["created_at"] for m in snapshot]),
                 "frequency_lfu": np.array([m["frequency"] for m in snapshot], dtype=float)}
    out: Dict[str, Any] = {
        "dataset": "ALFWorld", "split": split, "memories": len(snapshot),
        "tasks_with_needed": len(needed), "keep": keep,
        "verdicts_per_memory": float(total.mean()),
        "baselines": {k: {"harmful_deletion": harmful(v)} for k, v in baselines.items()},
        "curve": [],
    }
    w_r, w_f, w_u, w_p = h5
    for eps in eps_grid:
        harms = []
        for seed in seeds:
            rng = np.random.default_rng(2000 + seed)
            s2 = resample_counts(success, failure, eps, rng)
            u = utility(s2, total - s2)
            scores = np.clip(w_r * R + w_f * F + w_u * u + w_p * P, 0.0, 1.0)
            harms.append(harmful(scores, seed))
            if eps == 0:
                break
        out["curve"].append({"eps": eps, "harmful_deletion": float(np.mean(harms)),
                             "harmful_sd": float(np.std(harms))})
    return out


def crossing(curve: Sequence[Dict[str, float]], key: str, baseline: float,
             better: str = "higher") -> Optional[float]:
    """First eps at which the curve stops beating ``baseline`` (linear interpolation)."""
    prev = None
    for point in curve:
        value = point[key]
        beats = value > baseline if better == "higher" else value < baseline
        if not beats:
            if prev is None:
                return point["eps"]
            v0, e0 = prev
            span = (v0 - value) or 1.0
            return float(e0 + (point["eps"] - e0) * (v0 - baseline) / span)
        prev = (value, point["eps"])
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  How much feedback, and how accurate: a reliability law (fourth audit, K4)
# ═══════════════════════════════════════════════════════════════════════════════

Q_GRID = (1.0, 0.5, 0.25, 0.125)


def subsample_counts(success: np.ndarray, failure: np.ndarray, q: float,
                     rng: np.random.Generator) -> tuple:
    """Keep each verdict independently with probability ``q``: fewer verdicts per memory."""
    success = np.asarray(success, dtype=np.int64)
    failure = np.asarray(failure, dtype=np.int64)
    if q >= 1.0:
        return success.copy(), failure.copy()
    return rng.binomial(success, q), rng.binomial(failure, q)


def asymmetric_resample(success: np.ndarray, failure: np.ndarray, fnr: float, fpr: float,
                        rng: np.random.Generator) -> np.ndarray:
    """Successes after a rater that misses a true success with probability ``fnr`` and confirms a
    true failure with probability ``fpr`` (the symmetric case is fnr = fpr = eps)."""
    success = np.asarray(success, dtype=np.int64)
    failure = np.asarray(failure, dtype=np.int64)
    return (success - rng.binomial(success, fnr)) + rng.binomial(failure, fpr)


def judge_flip_rates(precision: float, recall: float, base_rate: float) -> Dict[str, float]:
    """Per-verdict error rates implied by precision and recall at a given positive rate."""
    tp = base_rate * recall
    fp = tp * (1.0 / precision - 1.0)
    return {"fnr": 1.0 - recall, "fpr": fp / (1.0 - base_rate)}


def _msc_arrays(run_dir: Path, split: str, label: str, lam4: float, lam5: float) -> Dict[str, Any]:
    rows = _rows(run_dir / "facts.jsonl", split)
    age_h = np.array([max(r["t_end"] - r["recency"], 0.0) / 3600.0 for r in rows])
    return {
        "rows": rows,
        "y": np.array([r["labels"][label] for r in rows]),
        "success": np.array([r["success"] for r in rows]),
        "failure": np.array([r["failure"] for r in rows]),
        "F": np.array([r["components"]["F"] for r in rows]),
        "P": np.array([r["components"]["P"] for r in rows]),
        "R4": np.exp(-lam4 * age_h),
        "R5": np.exp(-lam5 * age_h),
    }


def _score(a: Dict[str, Any], u: np.ndarray, weights: Sequence[float], R: np.ndarray) -> np.ndarray:
    w_r, w_f, w_u, w_p = weights
    return np.round(np.clip(w_r * R + w_f * a["F"] + w_u * u + w_p * a["P"], 0.0, 1.0), 6)


def msc_law(run_dir: Path, split: str, h4: Sequence[float], lam4: float, h5: Sequence[float], lam5: float,
            label: str = "lex_0.5", keep: float = 0.5, q_grid: Sequence[float] = Q_GRID,
            eps_grid: Sequence[float] = EPS_GRID, seeds: Sequence[int] = SEEDS,
            judge: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Crossing points eps*(n) as verdicts per memory shrink, and the judge-calibrated asymmetric point.

    Model: a utility estimate carries signal proportional to (1 - 2 eps) and noise that shrinks with
    the number of verdicts n, so the advantage over a verdict-free baseline vanishes when
    (1 - 2 eps*)^2 * n is a constant c. If the law holds, c estimated at each subsampling level is
    roughly constant, and eps*(n) = (1 - sqrt(c / n)) / 2, fitted at q = 1 only, predicts the
    crossings at the other levels.
    """
    a = _msc_arrays(run_dir, split, label, lam4, lam5)
    rows, y = a["rows"], a["y"]
    total = a["success"] + a["failure"]
    fifo = np.array([r["created_at"] for r in rows])
    recency = np.array([r["recency"] for r in rows])
    base_auc = auroc_delong(recency, y)["auc"]
    base_harm = float(np.mean(eviction(rows, fifo, label, keep)["harmful_deletion_rate"]))
    out: Dict[str, Any] = {"dataset": "MSC", "split": split, "label": label, "keep": keep,
                           "baselines": {"recency_auroc": base_auc, "fifo_harmful_deletion": base_harm},
                           "levels": []}
    for q in q_grid:
        curve = []
        n_bar = []
        for eps in eps_grid:
            aucs, harms = [], []
            for seed in seeds:
                rng = np.random.default_rng(3000 + seed)
                s_q, f_q = subsample_counts(a["success"], a["failure"], q, rng)
                n_q = s_q + f_q
                n_bar.append(float(n_q[total > 0].mean()))
                s2 = resample_counts(s_q, f_q, eps, rng)
                u = utility(s2, n_q - s2)
                aucs.append(auroc_delong(_score(a, u, h4, a["R4"]), y)["auc"])
                harms.append(float(np.mean(eviction(rows, _score(a, u, h5, a["R5"]), label, keep)["harmful_deletion_rate"])))
            curve.append({"eps": eps, "auroc": float(np.mean(aucs)), "harmful_deletion": float(np.mean(harms))})
        level = {"q": q, "mean_verdicts_per_memory_with_feedback": float(np.mean(n_bar)), "curve": curve,
                 "auroc_crosses_recency_at": crossing(curve, "auroc", base_auc, "higher"),
                 "harmful_crosses_fifo_at": crossing(curve, "harmful_deletion", base_harm, "lower")}
        n = level["mean_verdicts_per_memory_with_feedback"]
        for key, ckey in (("auroc_crosses_recency_at", "auroc_c_recency"), ("harmful_crosses_fifo_at", "harmful_c_fifo")):
            e = level[key]
            level[ckey] = None if e is None else (1 - 2 * e) ** 2 * n
        out["levels"].append(level)
    for key, ckey in (("auroc_crosses_recency_at", "auroc_c_recency"), ("harmful_crosses_fifo_at", "harmful_c_fifo")):
        levels = [lv for lv in out["levels"] if lv.get(ckey) is not None]
        if len(levels) >= 2 and levels[0]["q"] == 1.0:
            c_fit = levels[0][ckey]
            preds = []
            for lv in levels[1:]:
                n = lv["mean_verdicts_per_memory_with_feedback"]
                preds.append({"q": lv["q"], "n": n, "observed_eps_star": lv[key],
                              "predicted_eps_star": (1 - math.sqrt(min(c_fit / n, 1.0))) / 2})
            cs = [lv[ckey] for lv in levels]
            out[f"law_{ckey}"] = {"c_by_level": cs, "c_cv": float(np.std(cs) / np.mean(cs)),
                                  "c_fitted_at_q1": c_fit, "held_out_predictions": preds}
    if judge is not None:
        rates = judge_flip_rates(judge["precision"], judge["recall"], judge["base_rate"])
        aucs, harms = [], []
        for seed in seeds:
            rng = np.random.default_rng(4000 + seed)
            s2 = asymmetric_resample(a["success"], a["failure"], rates["fnr"], rates["fpr"], rng)
            u = utility(s2, total - s2)
            aucs.append(auroc_delong(_score(a, u, h4, a["R4"]), y)["auc"])
            harms.append(float(np.mean(eviction(rows, _score(a, u, h5, a["R5"]), label, keep)["harmful_deletion_rate"])))
        out["judge_calibrated"] = {"source": judge, **rates, "auroc": float(np.mean(aucs)), "auroc_sd": float(np.std(aucs)),
                                   "harmful_deletion": float(np.mean(harms)), "harmful_sd": float(np.std(harms)),
                                   "beats_recency_auroc": bool(np.mean(aucs) > base_auc),
                                   "beats_fifo_eviction": bool(np.mean(harms) < base_harm)}
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Feedback-reliability threshold for the retention score")
    p.add_argument("--run-msc", required=True)
    p.add_argument("--run-alfworld")
    p.add_argument("--split", default="test")
    p.add_argument("--alfworld-split", default="test_in")
    p.add_argument("--out", required=True)
    p.add_argument("--h4-weights", type=float, nargs=4, default=[0.2, 0.0, 0.8, 0.0])
    p.add_argument("--h4-lambda", type=float, default=0.05)
    p.add_argument("--h5-weights", type=float, nargs=4, default=[0.3, 0.0, 0.7, 0.0])
    p.add_argument("--h5-lambda", type=float, default=0.0005)
    p.add_argument("--alfworld-h5-weights", type=float, nargs=4, default=[0.0, 0.0, 0.0, 1.0])
    p.add_argument("--alfworld-h5-lambda", type=float, default=0.0005)
    p.add_argument("--judge-error", type=float, default=0.366,
                   help="measured error rate of the deployed judge (E5: accuracy 0.634)")
    p.add_argument("--law", action="store_true",
                   help="derive eps*(n) by verdict subsampling, and the judge-calibrated asymmetric point")
    p.add_argument("--judge-precision", type=float, default=0.6188235294117647)
    p.add_argument("--judge-recall", type=float, default=0.578021978021978)
    p.add_argument("--judge-base-rate", type=float, default=0.4700413223140496)
    args = p.parse_args(argv)

    if args.law:
        law = msc_law(Path(args.run_msc), args.split, args.h4_weights, args.h4_lambda, args.h5_weights, args.h5_lambda,
                      judge={"precision": args.judge_precision, "recall": args.judge_recall,
                             "base_rate": args.judge_base_rate})
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "feedback_law.json").write_text(json.dumps(law, indent=1), encoding="utf-8")
        for lv in law["levels"]:
            print(f"q={lv['q']:.3f} n={lv['mean_verdicts_per_memory_with_feedback']:.2f} "
                  f"eps*(AUROC vs recency)={lv['auroc_crosses_recency_at']} "
                  f"eps*(eviction vs FIFO)={lv['harmful_crosses_fifo_at']}")
        for key in ("law_auroc_c_recency", "law_harmful_c_fifo"):
            if key in law:
                print(key, json.dumps(law[key], indent=1))
        print("judge-calibrated:", json.dumps(law.get("judge_calibrated"), indent=1))
        print(f"wrote {out / 'feedback_law.json'}")
        return 0

    report: Dict[str, Any] = {"judge_error_rate": args.judge_error, "datasets": []}
    msc = msc_curve(Path(args.run_msc), args.split, args.h4_weights, args.h4_lambda,
                    args.h5_weights, args.h5_lambda)
    report["datasets"].append(msc)
    print(f"=== MSC ({msc['facts']} facts, {msc['verdicts_per_memory']:.1f} verdicts per memory)")
    print(f"    baselines: recency AUROC {msc['baselines']['recency_lru']['auroc']:.3f}, "
          f"harmful {msc['baselines']['recency_lru']['harmful_deletion']:.3f}; "
          f"FIFO harmful {msc['baselines']['fifo']['harmful_deletion']:.3f}")
    for point in msc["curve"]:
        print(f"    eps={point['eps']:.2f}  H4 AUROC={point['auroc']:.3f}  "
              f"H5 harmful={point['harmful_deletion']:.3f}")
    msc["auroc_crosses_recency_at"] = crossing(msc["curve"], "auroc",
                                               msc["baselines"]["recency_lru"]["auroc"], "higher")
    msc["harmful_crosses_fifo_at"] = crossing(msc["curve"], "harmful_deletion",
                                              msc["baselines"]["fifo"]["harmful_deletion"], "lower")
    print(f"    H4 stops beating recency at eps ~ {msc['auroc_crosses_recency_at']}")
    print(f"    H5 stops beating FIFO at eps ~ {msc['harmful_crosses_fifo_at']}")

    if args.run_alfworld:
        alf = alfworld_curve(Path(args.run_alfworld), args.alfworld_split,
                             args.alfworld_h5_weights, args.alfworld_h5_lambda)
        report["datasets"].append(alf)
        print(f"=== ALFWorld ({alf['memories']} memories, {alf['verdicts_per_memory']:.1f} verdicts each)")
        for point in alf["curve"]:
            print(f"    eps={point['eps']:.2f}  harmful={point['harmful_deletion']:.3f}")
        alf["harmful_crosses_fifo_at"] = crossing(alf["curve"], "harmful_deletion",
                                                  alf["baselines"]["fifo"]["harmful_deletion"], "lower")
        print(f"    stops beating FIFO at eps ~ {alf['harmful_crosses_fifo_at']}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "feedback_threshold.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {out/'feedback_threshold.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
