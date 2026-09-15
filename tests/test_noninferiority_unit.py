"""The pre-registered H1 non-inferiority test (one-sided clustered bootstrap, margin 0.02)."""

import numpy as np
import pytest

from benchmarks.dars_eval.stats import holm, paired_bootstrap_noninferiority

MARGIN = 0.02


def test_identical_methods_are_non_inferior():
    x = np.random.default_rng(0).integers(0, 2, 200).astype(float)
    r = paired_bootstrap_noninferiority(x, x, MARGIN, n_boot=2000)
    assert r["diff"] == 0.0 and r["p_one_sided"] == 0.0 and r["ci_lo"] == 0.0


def test_clearly_worse_method_is_not_non_inferior():
    b = np.ones(100)
    a = np.concatenate([np.zeros(20), np.ones(80)])        # 20 points worse than b
    r = paired_bootstrap_noninferiority(a, b, MARGIN, n_boot=2000)
    assert r["diff"] == pytest.approx(-0.2) and r["p_one_sided"] > 0.99


def test_p_value_agrees_with_the_confidence_bound():
    rng = np.random.default_rng(3)
    b = rng.integers(0, 2, 300).astype(float)
    a = b.copy()
    flip = rng.choice(300, 12, replace=False)
    a[flip] = 1 - a[flip]                                  # small symmetric noise
    clusters = np.repeat(np.arange(5), 60)
    r = paired_bootstrap_noninferiority(a, b, MARGIN, clusters, n_boot=4000, seed=1)
    assert (r["p_one_sided"] < 0.025) == (r["ci_lo"] > -MARGIN)


def test_doubled_one_sided_p_enters_holm_with_two_sided_p():
    adjusted = holm([2 * 0.004, 0.03, 0.2])
    assert adjusted == pytest.approx([0.024, 0.06, 0.2])
