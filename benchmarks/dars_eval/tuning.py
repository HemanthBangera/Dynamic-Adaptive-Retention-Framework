"""
E3 — weight and decay-rate selection and sensitivity (offline, from saved components).

* ``simplex_grid``         all weight vectors (w_r, w_f, w_u, w_p) on a 0.1 grid (286).
* ``recompute_R``          recency for any decay rate λ from stored timestamps.
* ``auc_rows``             AUROC of many score vectors at once (Mann–Whitney, tie-aware).
* ``dirichlet_around``     weight perturbations with a given coefficient of variation.
* ``time_to_threshold``    closed-form time until a never-accessed memory crosses a
                           retention threshold, S(t) = w_r e^{-λt} + w_f F + w_u U + w_p P.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata

LAMBDAS = (0.0005, 0.001, 0.005, 0.01, 0.05)       # per hour
DEFAULT_WEIGHTS = (0.30, 0.20, 0.30, 0.20)


def simplex_grid(step: float = 0.1) -> List[Tuple[float, float, float, float]]:
    n = int(round(1 / step))
    out = []
    for a in range(n + 1):
        for b in range(n + 1 - a):
            for c in range(n + 1 - a - b):
                out.append((a / n, b / n, c / n, (n - a - b - c) / n))
    return out


def recompute_R(recency: np.ndarray, now, lam: float) -> np.ndarray:
    return np.exp(-lam * np.maximum(np.asarray(now) - np.asarray(recency), 0.0) / 3600.0)


def auc_rows(scores: np.ndarray, labels: Sequence[bool]) -> np.ndarray:
    """AUROC of each row of ``scores`` (configs × items) against ``labels``."""
    y = np.asarray(labels, dtype=bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        raise ValueError("AUROC needs both classes")
    ranks = np.apply_along_axis(rankdata, 1, np.atleast_2d(scores))
    return (ranks[:, y].sum(axis=1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def dirichlet_around(weights: Sequence[float], cv: float = 0.2, draws: int = 1000,
                     seed: int = 0) -> np.ndarray:
    """Dirichlet draws with mean ``weights`` and mean coefficient of variation ≈ ``cv``."""
    w = np.clip(np.asarray(weights, dtype=float), 1e-3, None)
    w = w / w.sum()
    alpha0 = max(float(np.mean((1 - w) / w)) / cv ** 2 - 1, 1.0)
    return np.random.default_rng(seed).dirichlet(alpha0 * w, size=draws)


def closest_to_default(candidates: Sequence[Tuple[float, ...]],
                       default: Sequence[float] = DEFAULT_WEIGHTS) -> Tuple[float, ...]:
    d = np.asarray(default)
    return min(candidates, key=lambda w: (float(np.abs(np.asarray(w) - d).sum()), w))


def time_to_threshold(threshold: float, weights: Sequence[float], P: float, lam: float,
                      U: float = 0.5, F: float = 0.0) -> float:
    """Hours until S(t) = w_r e^{-λt} + w_f F + w_u U + w_p P falls to ``threshold``.

    0 if the memory is already at or below it; inf if it never gets there.
    """
    w_r, w_f, w_u, w_p = weights
    rest = w_f * F + w_u * U + w_p * P
    if rest + w_r <= threshold:
        return 0.0
    if rest >= threshold or w_r <= 0 or lam <= 0:
        return math.inf
    return -math.log((threshold - rest) / w_r) / lam


def select(results: Dict[Tuple, float], higher_is_better: bool = True,
           default_key: Optional[Tuple] = None) -> Tuple[Tuple, float]:
    """Best configuration; ties broken by closeness of the weights to the submitted defaults."""
    best_val = max(results.values()) if higher_is_better else min(results.values())
    tied = [k for k, v in results.items() if abs(v - best_val) < 1e-12]

    def distance(key: Tuple) -> float:
        w = next(x for x in key if isinstance(x, tuple) and len(x) == 4)
        return float(np.abs(np.asarray(w) - np.asarray(DEFAULT_WEIGHTS)).sum())

    best = min(tied, key=lambda k: (distance(k), str(k)))
    return best, best_val
