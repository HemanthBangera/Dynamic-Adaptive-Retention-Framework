"""Decision logic of the pre-registered primary analysis (make_tables.adjust)."""

import pytest

from benchmarks.dars_eval.make_tables import adjust, layout


def test_adjust_is_holm_based_and_direction_aware():
    rows = [
        {"hypothesis": "H1", "direction": "non-inferior", "effect": 0.0, "p": 0.0},
        {"hypothesis": "H2", "direction": "greater", "effect": 0.4, "p": 0.001},
        {"hypothesis": "H5", "direction": "less", "effect": 0.1, "p": 0.001},      # significant, wrong way
        {"hypothesis": "H4", "direction": "greater", "effect": 0.02, "p": 0.3},
        {"hypothesis": "H3", "comparison": "absent", "missing": "no artifact"},
    ]
    adjust(rows)
    assert rows[0]["decision"] == "non-inferior"
    assert rows[1]["decision"] == "supported"
    assert rows[2]["decision"] == "significant, opposite direction"
    assert rows[3]["decision"] == "not significant" and rows[3]["p_holm"] == pytest.approx(0.3)
    assert rows[1]["p_holm"] == pytest.approx(0.003)          # 4 live comparisons: 3 × 0.001, then max
    assert "p_holm" not in rows[4]


def test_test_layout_has_the_23_preregistered_comparisons():
    lay = layout("test")
    n_h1 = 5 + 4                                              # recall sources + EM sources
    n_h2 = len(lay["h2"])
    n_h3 = len(lay["e10"])
    n_h4 = 2
    n_h5 = 2 * (1 + 1 + len(lay["e10"]))                      # LRU, FIFO × (LongMemEval, MSC, ALFWorld splits)
    assert n_h1 + n_h2 + n_h3 + n_h4 + n_h5 == 23
