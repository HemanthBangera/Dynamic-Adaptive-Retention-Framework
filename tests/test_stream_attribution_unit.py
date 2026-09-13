"""Per-memory oracle credit, and FactConsolidation without serial numbers (local vault, no API)."""

import argparse
import asyncio

import pytest

from benchmarks.dars_eval.datasets import Context
from benchmarks.dars_eval.memory_units import fact_units
from benchmarks.dars_eval.rankers import Method
from benchmarks.dars_eval.run_stream import parse_feedback, run_context, strip_serial_rule

CONTEXT = "Here is a list of facts:\n0. Alpha lives in Paris.\n1. Beta lives in Rome.\n2. Gamma lives in Oslo.\n"


def test_oracle_unit_is_a_known_feedback_source():
    assert parse_feedback("oracle_unit") == ("oracle_unit", 0.0)
    with pytest.raises(ValueError):
        parse_feedback("oracle_units")


def test_fact_units_without_prefix_keep_serial_time_and_metadata():
    with_prefix = fact_units(CONTEXT, t0=10.0, seconds_per_serial=5.0)
    bare = fact_units(CONTEXT, t0=10.0, seconds_per_serial=5.0, serial_prefix=False)
    assert [u.text for u in with_prefix] == ["0. Alpha lives in Paris.", "1. Beta lives in Rome.", "2. Gamma lives in Oslo."]
    assert [u.text for u in bare] == ["Alpha lives in Paris.", "Beta lives in Rome.", "Gamma lives in Oslo."]
    assert [(u.timestamp, u.meta) for u in bare] == [(u.timestamp, u.meta) for u in with_prefix]


def test_serial_rule_is_removed_from_the_real_template_and_nothing_else():
    from benchmarks.memory_agent_bench.qa_builder import MAB_RAG_TEMPLATE_KEY, build_qa_pairs

    row = {"questions": ["Where does Alpha live?"], "answers": ["Paris"], "context": CONTEXT}
    fq = build_qa_pairs(row, "factconsolidation_sh_6k", agent_key=MAB_RAG_TEMPLATE_KEY)[0][0]
    stripped = strip_serial_rule(fq)
    assert "serial number" not in stripped and "Where does Alpha live?" in stripped
    assert len(fq) - len(stripped) > 150
    with pytest.raises(ValueError):
        strip_serial_rule(stripped)


def _run(feedback, evidence):
    units = fact_units(CONTEXT, t0=1_000.0, seconds_per_serial=3600.0)
    ctx = Context(source="factconsolidation_sh_6k", index=0, units=units, questions=["q"],
                  queries=["Where does Alpha live?"], formatted_queries=["fq"], answers=["Paris"],
                  evidence=[evidence], extra={})
    args = argparse.Namespace(question_step=3600.0, budget=2048, memory_budget=0.0, eviction="dars",
                              eviction_weights=None, seed=0, dynamics_k=10, lexical_tau=0.5)
    rows = []
    method = Method("dars_rrf_k50", rank_mode="rrf", fetch_k=50)
    captured = {}

    import benchmarks.dars_eval.run_stream as rs
    original = rs.MemoryIndex

    class Capturing(original):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            captured["index"] = self

    rs.MemoryIndex = Capturing
    try:
        asyncio.run(run_context(ctx, method, feedback, args, None, None, rows))
    finally:
        rs.MemoryIndex = original
    idx = captured["index"]
    counts = {}
    for u, pid in idx.point_ids.items():
        pl = idx.vault.get_memory(pid).payload
        counts[u] = (pl.success_count, pl.failure_count)
    return rows[0], counts


def test_oracle_unit_credits_only_the_gold_memory():
    rec, counts = _run("oracle_unit", [[0]])
    assert set(rec["shown"]) == {0, 1, 2}
    assert counts[0] == (1, 0) and counts[1] == (0, 1) and counts[2] == (0, 1)
    assert rec["verdict"] == pytest.approx(1 / 3)


def test_set_level_oracle_credits_every_shown_memory():
    rec, counts = _run("oracle", [[0]])
    assert all(c == (1, 0) for c in counts.values()) and rec["verdict"] is True


def test_the_two_oracles_agree_when_every_shown_memory_is_evidence():
    _, unit = _run("oracle_unit", [[0], [1], [2]])
    _, whole = _run("oracle", [[0], [1], [2]])
    assert unit == whole


def test_unlabelled_questions_get_no_per_memory_update():
    rec, counts = _run("oracle_unit", [])
    assert rec["verdict"] is None and all(c == (0, 0) for c in counts.values())
