"""
The ``evaluate`` subcommands of E9 (run_msc) and E10 (run_alfworld): pre-registered tests
of fixed, dev-selected configurations.  Synthetic runs where the right answers are known
by construction (utility separates needed from unneeded memories).
"""

import json

import pytest

from benchmarks.dars_eval import run_alfworld, run_msc

UTILITY_ONLY = (0.0, 0.0, 1.0, 0.0)


def _msc_run(tmp_path):
    rows = []
    for d in range(4):
        for i in range(4):
            needed = i < 2
            rows.append({
                "dialogue": d, "speaker": 1, "text": f"fact {d}-{i}", "created_session": i % 3,
                "mentions": 1 + i, "frequency": 4 - i, "success": 0, "failure": 0,
                "recency": 1000.0 + i, "created_at": 900.0 + i, "t_end": 5000.0,
                "components": {"R": 0.5, "F": 0.1, "U": 0.9 if needed else 0.2, "P": 0.0},
                "labels": {"lex_0.5": needed}, "split": "test",
            })
    (tmp_path / "facts.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return tmp_path


def test_msc_evaluate_fixed_configuration(tmp_path):
    rep = run_msc.evaluate(_msc_run(tmp_path), "test", (0.005, UTILITY_ONLY), (0.005, UTILITY_ONLY), n_boot=200)
    assert rep["facts"] == 16 and rep["dialogues"] == 4 and rep["base_rate"] == 0.5
    assert rep["h4"]["auroc_selected"]["auc"] == pytest.approx(1.0)
    assert rep["h4"]["comparators"]["recency_lru"]["auc"] == pytest.approx(0.0)   # newest facts are unneeded
    assert rep["h5"]["selected"]["mean"] == pytest.approx(0.0)
    assert rep["h5"]["comparators"]["recency_lru"]["mean"] == pytest.approx(1.0)
    assert rep["h5"]["paired"]["fifo"]["diff"] == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        run_msc.evaluate(tmp_path, "dev", (0.005, UTILITY_ONLY), (0.005, UTILITY_ONLY), n_boot=50)


def _alf_run(tmp_path):
    rows = []
    for t in range(3):
        cands = [{
            "pid": f"c{t}{j}", "recep": f"r{j}", "is_true": j == 1, "sim": 0.9 - 0.1 * j,
            "count_prior": 5 if j == 1 else 1, "recency": 1000.0, "created_at": 900.0, "now": 5000.0,
            "frequency": 1, "success": 0, "failure": 0, "R": 0.5, "F": 0.1, "U": 0.9 if j == 1 else 0.3, "P": 0.0,
        } for j in range(3)]
        rows.append({"task_id": f"t{t}", "split": "test_in", "task_type": "x", "truth": {}, "query": "q",
                     "candidates": cands, "reachable": True, "needed": [f"m{t}"]})
    (tmp_path / "eval_test_in.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    mems = [{"pid": f"m{t}", "recency": 1000.0 + t, "created_at": 900.0 + t, "frequency": 1,
             "R": 0.5, "F": 0.1, "U": 0.9, "P": 0.0, "S": 0.5} for t in range(3)]
    mems += [{"pid": f"x{k}", "recency": 2000.0 + k, "created_at": 1900.0 + k, "frequency": 1,
              "R": 0.5, "F": 0.1, "U": 0.2, "P": 0.0, "S": 0.4} for k in range(3)]
    (tmp_path / "memories.jsonl").write_text("\n".join(json.dumps(m) for m in mems), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({"now": 5000.0}), encoding="utf-8")
    return tmp_path


def test_alfworld_evaluate_fixed_configuration(tmp_path):
    rep = run_alfworld.evaluate(_alf_run(tmp_path), "test_in", (0.005, "score_only", 0.0, UTILITY_ONLY),
                                (0.005, UTILITY_ONLY), n_boot=200)
    h3, h5 = rep["h3"], rep["h5"]
    assert h3["reachable_with_alternatives"] == 3
    assert h3["mrr"]["selected"]["mean"] == pytest.approx(1.0)
    assert h3["mrr"]["similarity"]["mean"] == pytest.approx(0.5)
    assert h3["mrr"]["count_prior"]["mean"] == pytest.approx(1.0)
    assert h3["paired_vs_similarity"]["selected"]["diff"] == pytest.approx(0.5)
    assert h5["harmful_deletion"]["selected"]["mean"] == pytest.approx(0.0)
    assert h5["harmful_deletion"]["recency_lru"]["mean"] == pytest.approx(1.0)
    assert h5["paired_vs_selected"]["fifo"]["diff"] == pytest.approx(-1.0)


def test_keep_mask_breaks_ties_randomly_not_by_position():
    import numpy as np

    scores = np.zeros(10)
    kept = run_alfworld._keep_mask(scores, 5, seed=0)
    assert kept.sum() == 5 and not kept[:5].all()        # not simply the first five
