"""E5: the agreement report separates the two halves of the reference label and splits by source."""

from benchmarks.dars_eval.run_judge import agreement_report


def _item(source, context, correct, has_evidence, verdict):
    return {"source": source, "context": context, "question": 0, "method": "m", "seed": 0,
            "answer": "a", "parsed": "a", "memories": ["m0"], "memory_is_evidence": [has_evidence],
            "reference": bool(correct and has_evidence), "correct": correct, "has_evidence": has_evidence,
            "judge": {"j|layer_b|s0": verdict}}


def test_rater_tracking_correctness_only_is_visible_in_the_secondary_kappas():
    # The judge says YES exactly when the answer is correct, but half of those
    # answers were produced without the gold evidence, so the reference differs.
    items = []
    for i in range(20):
        correct = i % 2 == 0
        has_evidence = i % 4 == 0
        items.append(_item("src_a", i % 5, correct, has_evidence, "YES" if correct else "NO"))
    rep = agreement_report(items, taus=[], n_boot=50)
    r = rep["raters"]["j|layer_b|s0"]
    assert r["vs_correct_kappa"] == 1.0                  # perfectly tracks correctness
    assert r["kappa"] < r["vs_correct_kappa"]            # but not the AND-reference
    assert r["vs_has_evidence_kappa"] < r["vs_correct_kappa"]


def test_by_source_breakdown_counts_every_source():
    items = [_item("src_a", 0, True, True, "YES"), _item("src_a", 1, False, True, "NO"),
             _item("src_b", 0, True, True, "NO"), _item("src_b", 1, False, False, "YES")]
    rep = agreement_report(items, taus=[], n_boot=50)
    by = rep["raters"]["j|layer_b|s0"]["by_source"]
    assert set(by) == {"src_a", "src_b"}
    assert by["src_a"]["n"] == 2 and by["src_b"]["n"] == 2
    assert by["src_a"]["accuracy"] == 1.0                # both items agree with the reference
    assert by["src_b"]["accuracy"] == 0.0                # both disagree


def test_constant_ratings_give_no_kappa_instead_of_crashing():
    items = [_item("src_a", i, True, True, "YES") for i in range(4)]
    r = agreement_report(items, taus=[], n_boot=50)["raters"]["j|layer_b|s0"]
    assert r["kappa"] is None and r["vs_correct_kappa"] is None
    assert r["accuracy"]["mean"] == 1.0
