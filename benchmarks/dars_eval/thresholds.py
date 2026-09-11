"""
E3 — retention thresholds: the closed-form lifecycle, and stability under threshold shifts.

Layer C classifies a memory by its DARS score S (``MemoryVault.classify_memory``):

    RETAIN    S > 0.70
    COMPRESS  0.30 < S <= 0.70
    DELETE    S <= 0.30

**1. Closed form.**
- With R = e^{-λt} (t = hours since the last access) and the other components
  fixed, S(t) = w_r e^{-λt} + w_f F + w_u U + w_p P.
- The time a memory spends in each tier therefore follows analytically
  (``tuning.time_to_threshold``).
- ``lifecycle_table`` evaluates this in two cases:
  - memories never accessed: F = 0, and U = 0.5 (the Laplace prior);
  - memories retrieved k times, all successfully: F = log(1+k)/log(51), U = (k+1)/(k+2).
- ``min_successes_to_retain`` gives the smallest such k for which a
  just-accessed memory (R = 1) is RETAINed.

**2. Stability.**
- ``shift_stability`` re-classifies an empirical score snapshot under threshold
  shifts of ±0.05 and ±0.10. Example snapshots: the end-of-stream memories of
  E10 (``memories.jsonl``) or E9.
- It reports the fraction of memories that change tier, and Cohen's κ between
  the tier assignments.

Usage
-----
python -m benchmarks.dars_eval.thresholds --out benchmark_runs/revision/E3/thresholds \
    [--snapshot E10=benchmark_runs/revision/E10/dev_default/memories.jsonl] \
    [--weights 0.3 0.2 0.3 0.2] [--decay-lambda 0.005]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from benchmarks.dars_eval.tuning import DEFAULT_WEIGHTS, LAMBDAS, time_to_threshold

RETAIN = 0.70
COMPRESS = 0.30
FREQUENCY_CAP = 50
TIER_NAMES = ("delete", "compress", "retain")
P_VALUES = (0.0, 0.03, 0.2, 0.5)         # 0.03 ≈ mean P of the submitted configuration on MAB
ACCESS_COUNTS = (0, 1, 3, 10, 50)
SHIFTS = (-0.10, -0.05, 0.05, 0.10)


def frequency_component(accesses: int) -> float:
    return min(math.log1p(accesses) / math.log1p(FREQUENCY_CAP), 1.0)


def utility_component(successes: int, failures: int = 0) -> float:
    return (successes + 1) / (successes + failures + 2)


def tier(scores, retain: float = RETAIN, compress: float = COMPRESS) -> np.ndarray:
    """0 = delete, 1 = compress, 2 = retain (boundaries as in the code: S > retain, S > compress)."""
    s = np.asarray(scores, dtype=float)
    return np.where(s > retain, 2, np.where(s > compress, 1, 0))


def score(weights: Sequence[float], R: float, F: float, U: float, P: float) -> float:
    w_r, w_f, w_u, w_p = weights
    return w_r * R + w_f * F + w_u * U + w_p * P


def min_successes_to_retain(weights: Sequence[float], P: float, max_k: int = 1000) -> Optional[int]:
    """Smallest number of all-successful retrievals after which a just-accessed memory is RETAINed."""
    for k in range(max_k + 1):
        if score(weights, 1.0, frequency_component(k), utility_component(k), P) > RETAIN:
            return k
    return None


def lifecycle_table(weights: Sequence[float] = DEFAULT_WEIGHTS, lambdas: Sequence[float] = LAMBDAS,
                    p_values: Sequence[float] = P_VALUES,
                    access_counts: Sequence[int] = ACCESS_COUNTS) -> List[Dict[str, Any]]:
    rows = []
    for lam in lambdas:
        for P in p_values:
            for k in access_counts:
                F, U = frequency_component(k), utility_component(k)
                s0 = score(weights, 1.0, F, U, P)
                floor = score(weights, 0.0, F, U, P)
                rows.append({
                    "lambda_per_hour": lam, "P": P, "successful_accesses": k, "F": F, "U": U,
                    "S_at_access": s0, "tier_at_access": TIER_NAMES[int(tier(s0))],
                    "S_floor": floor, "tier_floor": TIER_NAMES[int(tier(floor))],
                    "hours_until_leaves_retain": time_to_threshold(RETAIN, weights, P, lam, U=U, F=F),
                    "hours_until_delete": time_to_threshold(COMPRESS, weights, P, lam, U=U, F=F),
                })
    return rows


def cohen_kappa_multiclass(a: Sequence[int], b: Sequence[int]) -> float:
    x, y = np.asarray(a), np.asarray(b)
    labels = np.union1d(x, y)
    po = float(np.mean(x == y))
    pe = float(sum(np.mean(x == c) * np.mean(y == c) for c in labels))
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)


def shift_stability(scores: Sequence[float], shifts: Sequence[float] = SHIFTS) -> Dict[str, Any]:
    """Tier changes when the retain / compress thresholds move (together and one at a time)."""
    s = np.asarray(scores, dtype=float)
    base = tier(s)
    out: Dict[str, Any] = {
        "n": int(len(s)),
        "base_shares": {TIER_NAMES[t]: float(np.mean(base == t)) for t in range(3)},
        "shifts": [],
    }
    for d in shifts:
        for which, (r, c) in (("both", (RETAIN + d, COMPRESS + d)),
                              ("retain", (RETAIN + d, COMPRESS)),
                              ("compress", (RETAIN, COMPRESS + d))):
            alt = tier(s, r, c)
            out["shifts"].append({
                "shift": d, "threshold": which, "retain": r, "compress": c,
                "changed": float(np.mean(alt != base)),
                "kappa": cohen_kappa_multiclass(base, alt),
                "shares": {TIER_NAMES[t]: float(np.mean(alt == t)) for t in range(3)},
            })
    return out


def snapshot_scores(path: Path, weights: Optional[Sequence[float]], lam: Optional[float] = None,
                    now: Optional[float] = None) -> np.ndarray:
    """S of every memory in a jsonl snapshot.

    Snapshots come in two layouts:
    - E10 ``memories.jsonl``: top-level R, F, U, P and S fields;
    - E9 ``facts.jsonl``: R, F, U, P nested under ``components``, and no S.

    The stored ``S`` is used when present and neither ``weights`` nor ``lam`` is given.
    Otherwise S is recomputed from R, F, U, P, using the default weights when ``weights``
    is None. With ``lam``, R is recomputed as exp(-lam · hours since ``recency``) at the
    snapshot time: the row's ``t_end`` if present (E9), else ``now`` (E10 manifest).
    """
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    w = tuple(weights) if weights is not None else DEFAULT_WEIGHTS
    out = []
    for r in rows:
        if weights is None and lam is None and "S" in r:
            out.append(float(r["S"]))
            continue
        c = r.get("components", r)
        R = c["R"]
        if lam is not None:
            t = r.get("t_end", now)
            if t is None:
                raise ValueError("--now is required to recompute R for snapshots without t_end")
            R = math.exp(-lam * max(float(t) - float(r["recency"]), 0.0) / 3600.0)
        out.append(min(max(score(w, R, c["F"], c["U"], c["P"]), 0.0), 1.0))
    return np.array(out, dtype=float)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="E3 retention thresholds: closed form and stability")
    p.add_argument("--out", required=True)
    p.add_argument("--weights", type=float, nargs=4, default=None,
                   help="recompute snapshot scores with these weights (default: stored S)")
    p.add_argument("--decay-lambda", type=float, default=None,
                   help="recompute R with this decay rate per hour (default: stored R)")
    p.add_argument("--now", type=float, default=None,
                   help="snapshot time for R when rows have no t_end (E10: manifest 'now')")
    p.add_argument("--snapshot", action="append", default=[], help="NAME=path/to/memories.jsonl")
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    weights = tuple(args.weights) if args.weights else DEFAULT_WEIGHTS
    report: Dict[str, Any] = {
        "weights": weights, "thresholds": {"retain_gt": RETAIN, "compress_gt": COMPRESS},
        "lifecycle": lifecycle_table(weights),
        "min_successes_to_retain": {str(P): min_successes_to_retain(weights, P) for P in P_VALUES},
        "stability": {},
    }
    for spec in args.snapshot:
        name, path = spec.split("=", 1)
        s = snapshot_scores(Path(path), args.weights, args.decay_lambda, args.now)
        report["stability"][name] = shift_stability(s)
    (out / "thresholds.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")

    print(f"weights={weights}  min successes to RETAIN by P: {report['min_successes_to_retain']}")
    for r in report["lifecycle"]:
        if r["lambda_per_hour"] == 0.005 and r["successful_accesses"] in (0, 10):
            print(f"  λ=0.005 P={r['P']:.2f} k={r['successful_accesses']:2d}: S={r['S_at_access']:.3f} "
                  f"({r['tier_at_access']}) → delete after {r['hours_until_delete']:.1f} h")
    for name, st in report["stability"].items():
        worst = max(st["shifts"], key=lambda x: x["changed"])
        print(f"  {name}: base {st['base_shares']}; worst shift {worst['shift']:+.2f} ({worst['threshold']}) "
              f"changes {worst['changed']:.1%} of tiers, κ={worst['kappa']:.3f}")


if __name__ == "__main__":
    main()
