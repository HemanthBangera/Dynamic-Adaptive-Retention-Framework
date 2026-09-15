"""Unit tests for the E4 predictive-relevance helpers (pure numpy; no model or vault)."""

import numpy as np
import pytest

from benchmarks.dars_eval.run_p_variants import goal_p, order_stats, unit_rows, window_goal


def test_window_goal_uses_only_past_or_only_future_queries():
    Q = np.eye(4)
    assert window_goal(Q, 0, 3, "dynamic") is None                 # no history for the first query
    np.testing.assert_allclose(window_goal(Q, 2, 3, "dynamic"), Q[:2].mean(axis=0))
    np.testing.assert_allclose(window_goal(Q, 3, 2, "dynamic"), Q[1:3].mean(axis=0))
    np.testing.assert_allclose(window_goal(Q, 0, 2, "oracle"), Q[1:3].mean(axis=0))   # excludes query 0
    assert window_goal(Q, 3, 2, "oracle") is None                  # no future for the last query
    with pytest.raises(ValueError):
        window_goal(Q, 1, 2, "task")


def test_goal_p_is_clipped_cosine():
    V = unit_rows(np.array([[1.0, 0.0], [0.0, 2.0], [-1.0, 0.0]]))
    np.testing.assert_allclose(goal_p(V, np.array([3.0, 0.0])), [1.0, 0.0, 0.0])


def test_order_stats():
    sim = list(range(20))
    same = order_stats([0, 1, 2], sim[:3], sim[:3])
    assert same == {"top10_changed": 0.0, "kendall_tau": 1.0}
    rev = order_stats([2, 1, 0], [2, 1, 0], sim[:3])
    assert rev["top10_changed"] == 0.0 and rev["kendall_tau"] == pytest.approx(-1.0)

    cand = sim[:15]
    order = [12] + [i for i in range(15) if i != 12]                # candidate 12 promoted to rank 1
    ranked = [cand[i] for i in order] + sim[15:]
    moved = order_stats(order, ranked, sim)
    assert moved["top10_changed"] == pytest.approx(0.1)
    assert 0.0 < moved["kendall_tau"] < 1.0
