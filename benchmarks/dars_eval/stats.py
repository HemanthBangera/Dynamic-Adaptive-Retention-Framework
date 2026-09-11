"""
Statistics for the revision experiments.

* Two-stage clustered bootstrap (resample clusters, then items within each
  resampled cluster) for means and paired differences — questions are nested in
  contexts/dialogues, so i.i.d. resampling would understate uncertainty.
* Exact McNemar test for paired binary outcomes.
* Holm–Bonferroni step-down correction.
* Cohen's kappa for agreement between binary raters.
* AUROC with DeLong variance (single and paired), via the fast algorithm of
  Sun & Xu (2014).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, sqrt
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sps


@dataclass
class Estimate:
    mean: float
    lo: float
    hi: float
    n: int

    def as_dict(self) -> Dict[str, float]:
        return {"mean": self.mean, "ci_lo": self.lo, "ci_hi": self.hi, "n": self.n}


def _groups(clusters: Sequence) -> List[np.ndarray]:
    clusters = np.asarray(clusters)
    return [np.flatnonzero(clusters == c) for c in dict.fromkeys(clusters.tolist())]


def _two_stage_means(v: np.ndarray, groups: List[np.ndarray], n_boot: int,
                     rng: np.random.Generator, chunk: int = 250) -> np.ndarray:
    """
    Bootstrap means under two-stage resampling, vectorised: each replicate draws
    ``G`` clusters with replacement, then, independently for every drawn cluster,
    its items with replacement; the replicate statistic is the mean over all drawn
    items.  (Singleton clusters reduce this to the ordinary i.i.d. bootstrap.)
    """
    sizes = np.array([len(g) for g in groups])
    flat = np.concatenate(groups)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    n_groups = len(groups)
    out = np.empty(n_boot)
    for lo in range(0, n_boot, chunk):
        b = min(chunk, n_boot - lo)
        picks = rng.integers(0, n_groups, size=(b, n_groups)).ravel()
        pick_sizes = sizes[picks]
        rep = np.repeat(picks, pick_sizes)
        local = (rng.random(rep.size) * sizes[rep]).astype(np.int64)
        vals = v[flat[starts[rep] + local]]
        per_replicate_items = pick_sizes.reshape(b, n_groups).sum(axis=1)
        bounds = np.concatenate([[0], np.cumsum(per_replicate_items)[:-1]])
        out[lo:lo + b] = np.add.reduceat(vals, bounds) / per_replicate_items
    return out


def cluster_bootstrap_mean(values: Sequence[float], clusters: Optional[Sequence] = None,
                           n_boot: int = 10_000, seed: int = 0, level: float = 0.95) -> Estimate:
    """Mean with a percentile CI from the two-stage clustered bootstrap."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0)
    groups = _groups(clusters if clusters is not None else np.arange(len(v)))
    boots = _two_stage_means(v, groups, n_boot, np.random.default_rng(seed))
    a = (1 - level) / 2
    return Estimate(float(v.mean()), float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a)), len(v))


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float], clusters: Optional[Sequence] = None,
                          n_boot: int = 10_000, seed: int = 0, level: float = 0.95) -> Dict[str, float]:
    """Mean of (a − b) with a clustered-bootstrap CI and a two-sided bootstrap p-value."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    groups = _groups(clusters if clusters is not None else np.arange(len(d)))
    boots = _two_stage_means(d, groups, n_boot, np.random.default_rng(seed))
    alpha = (1 - level) / 2
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {
        "diff": float(d.mean()),
        "ci_lo": float(np.quantile(boots, alpha)),
        "ci_hi": float(np.quantile(boots, 1 - alpha)),
        "p_value": float(min(1.0, p)),
        "n": int(len(d)),
    }


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> Dict[str, float]:
    """Exact two-sided McNemar test on paired binary outcomes."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    n01 = int(np.sum(a & ~b))
    n10 = int(np.sum(~a & b))
    n = n01 + n10
    if n == 0:
        return {"a_only": n01, "b_only": n10, "p_value": 1.0}
    k = min(n01, n10)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"a_only": n01, "b_only": n10, "p_value": float(min(1.0, p))}


def holm(p_values: Sequence[float]) -> List[float]:
    """Holm–Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted.tolist()


def cohen_kappa(r1: Sequence[bool], r2: Sequence[bool]) -> float:
    """Cohen's kappa for two binary raters (1.0 = perfect, 0 = chance agreement)."""
    x = np.asarray(r1, dtype=bool)
    y = np.asarray(r2, dtype=bool)
    po = float(np.mean(x == y))
    pe = float(x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean()))
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)


# ── AUROC with DeLong variance ──────────────────────────────────────────────


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    z = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def _delong_components(scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fast DeLong (Sun & Xu, 2014) for k score vectors sharing the same labels."""
    pos = labels.astype(bool)
    x, y = scores[:, pos], scores[:, ~pos]
    m, n = x.shape[1], y.shape[1]
    k = scores.shape[0]
    tx = np.array([_midrank(x[r]) for r in range(k)])
    ty = np.array([_midrank(y[r]) for r in range(k)])
    tz = np.array([_midrank(np.concatenate([x[r], y[r]])) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n, np.array([m, n])


def auroc_delong(scores: Sequence[float], labels: Sequence[bool], level: float = 0.95) -> Dict[str, float]:
    """AUROC with a DeLong normal-approximation CI."""
    s = np.asarray(scores, dtype=float)[None, :]
    lab = np.asarray(labels, dtype=bool)
    if lab.all() or not lab.any():
        raise ValueError("AUROC needs both positive and negative labels")
    aucs, cov, _ = _delong_components(s, lab)
    se = sqrt(max(cov[0, 0], 0.0))
    z = sps.norm.ppf(1 - (1 - level) / 2)
    return {"auc": float(aucs[0]), "ci_lo": float(aucs[0] - z * se), "ci_hi": float(aucs[0] + z * se),
            "se": se, "n_pos": int(lab.sum()), "n_neg": int((~lab).sum())}


def auroc_delong_paired(scores_a: Sequence[float], scores_b: Sequence[float],
                        labels: Sequence[bool]) -> Dict[str, float]:
    """DeLong test for the difference of two correlated AUROCs on the same items."""
    s = np.vstack([np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)])
    lab = np.asarray(labels, dtype=bool)
    aucs, cov, _ = _delong_components(s, lab)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = float(aucs[0] - aucs[1])
    if var <= 0:
        return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff, "p_value": 1.0 if diff == 0 else 0.0}
    zval = diff / sqrt(var)
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff,
            "z": float(zval), "p_value": float(2 * sps.norm.sf(abs(zval)))}
