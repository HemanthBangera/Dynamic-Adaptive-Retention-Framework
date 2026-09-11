"""Unit tests for the E3 threshold analysis (closed form and stability; pure python)."""

import math

import numpy as np
import pytest

from benchmarks.dars_eval.thresholds import (
    cohen_kappa_multiclass,
    lifecycle_table,
    min_successes_to_retain,
    shift_stability,
    tier,
)
from benchmarks.dars_eval.tuning import DEFAULT_WEIGHTS


def test_tier_boundaries_match_the_code():
    # DELETE at S <= 0.30, COMPRESS at 0.30 < S <= 0.70, RETAIN at S > 0.70
    assert tier([0.30, 0.300001, 0.70, 0.700001]).tolist() == [0, 1, 1, 2]


def test_never_accessed_memory_starts_in_compress_and_is_deleted_in_closed_form():
    row = next(r for r in lifecycle_table(DEFAULT_WEIGHTS, lambdas=(0.005,), p_values=(0.0,),
                                          access_counts=(0,)))
    assert row["S_at_access"] == pytest.approx(0.45)
    assert row["tier_at_access"] == "compress"
    assert row["hours_until_leaves_retain"] == 0.0
    assert row["hours_until_delete"] == pytest.approx(math.log(2) / 0.005)   # ≈ 138.6 h
    assert row["tier_floor"] == "delete"


def test_retain_needs_eleven_consecutive_successes_with_default_weights():
    assert min_successes_to_retain(DEFAULT_WEIGHTS, P=0.0) == 11
    assert min_successes_to_retain((0.0, 0.0, 0.0, 1.0), P=0.5) is None


def test_multiclass_kappa():
    assert cohen_kappa_multiclass([0, 1, 2, 2], [0, 1, 2, 2]) == 1.0
    # po = 0.5; pe = p(0)·p(0) + p(1)·p(1) + p(2)·p(2) = 0.5*0.25 + 0.5*0.5 + 0*0.25 = 0.375
    # kappa = (0.5 - 0.375) / (1 - 0.375) = 0.2
    assert cohen_kappa_multiclass([0, 0, 1, 1], [0, 1, 1, 2]) == pytest.approx(0.2)


def test_shift_stability():
    scores = np.array([0.1, 0.5, 0.5, 0.9])
    st = shift_stability(scores, shifts=(0.25,))
    assert st["base_shares"] == {"delete": 0.25, "compress": 0.5, "retain": 0.25}
    compress_up = next(s for s in st["shifts"] if s["threshold"] == "compress")
    assert compress_up["changed"] == pytest.approx(0.5)       # both 0.5s fall to delete (0.5 <= 0.55)
    both = next(s for s in st["shifts"] if s["threshold"] == "both")
    assert both["shares"]["retain"] == 0.0                     # 0.9 <= 0.95
