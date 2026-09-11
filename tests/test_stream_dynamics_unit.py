"""Unit tests for the E2 ranking-dynamics measures (pure python; no model or vault)."""

from benchmarks.dars_eval.rankers import Method
from benchmarks.dars_eval.run_stream import dynamics


def _candidates():
    # Similarity prefers c0; recency prefers c3.  F, U, P are constant across candidates.
    sims = [0.9, 0.8, 0.7, 0.6]
    recency = [0.1, 0.2, 0.9, 1.0]
    return [
        {"sim": s, "sim_rank": float(i + 1), "R": r, "F": 0.0, "U": 0.5, "P": 0.0}
        for i, (s, r) in enumerate(zip(sims, recency))
    ]


def test_constant_component_has_zero_neutral_influence():
    rrf = Method("dars_rrf", rank_mode="rrf", fetch_k=4)
    d = dynamics(_candidates(), rrf, k=2)
    assert d["neutral_changes_topk_F"] == 0.0
    assert d["neutral_changes_topk_U"] == 0.0
    assert d["neutral_changes_topk_P"] == 0.0
    assert d["neutral_changes_topk_R"] == 1.0          # recency decides the second slot
    assert d["topk_differs_from_similarity"] == 1.0
    assert d["std_F"] == 0.0


def test_blend_ablation_can_move_ranking_without_component_variation():
    blend = Method("dars_blend", rank_mode="blend", fetch_k=4, alpha=0.5)
    d = dynamics(_candidates(), blend, k=2)
    assert d["neutral_changes_topk_F"] == 0.0           # variation-based influence stays 0
    assert set(d) >= {"loo_changes_topk_F", "neutral_changes_topk_R"}
