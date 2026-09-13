"""run_judge.build_items: method/seed filters, reference labels and reader-only baselines (no LLM calls)."""

import json

import pytest

from benchmarks.dars_eval.datasets import load_contexts
from benchmarks.dars_eval.memory_units import WindowGuard
from benchmarks.dars_eval.run_judge import build_items

SOURCE = "ruler_qa1_197K"


@pytest.fixture(scope="module")
def ctx():
    return load_contexts(SOURCE, WindowGuard())[0]


def _reader(seed, em):
    return {"seed": seed, "budget": 5120, "output": "x", "parsed_output": "x",
            "metrics": {"substring_exact_match": em}}


def _run_dir(tmp_path, ctx):
    q = next(i for i, ev in enumerate(ctx.evidence) if ev)
    gold = sorted({u for g in ctx.evidence[q] for u in g})
    other = [u for u in range(len(ctx.units)) if u not in gold][:3]
    records = [
        {"source": SOURCE, "context": ctx.index, "question": q, "method": "similarity",
         "ranked": gold + other, "reader": [_reader(0, 1.0), _reader(1, 0.0)]},
        {"source": SOURCE, "context": ctx.index, "question": q, "method": "bm25",
         "ranked": other, "reader": [_reader(0, 1.0)]},
        {"source": SOURCE, "context": ctx.index, "question": q, "method": "no_memory",
         "ranked": [], "reader_budget_label": "none", "reader": [_reader(0, 1.0)]},
    ]
    (tmp_path / "manifest.json").write_text(json.dumps({"source": SOURCE, "reader_budget": 5120}), encoding="utf-8")
    (tmp_path / "per_question.jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return tmp_path


def test_method_and_seed_filters(tmp_path, ctx):
    d = _run_dir(tmp_path, ctx)
    guard = WindowGuard()
    assert len(build_items(d, guard)) == 4
    assert len(build_items(d, guard, methods=["similarity"])) == 2
    assert len(build_items(d, guard, seeds=[0])) == 3
    assert len(build_items(d, guard, methods=["similarity", "bm25"], seeds=[0])) == 2


def test_reference_needs_correct_answer_and_gold_evidence(tmp_path, ctx):
    d = _run_dir(tmp_path, ctx)
    guard = WindowGuard()
    s0 = build_items(d, guard, methods=["similarity"], seeds=[0])[0]
    s1 = build_items(d, guard, methods=["similarity"], seeds=[1])[0]
    assert s0["has_evidence"] and s0["correct"] and s0["reference"]
    assert s1["has_evidence"] and not s1["correct"] and not s1["reference"]
    bm25 = build_items(d, guard, methods=["bm25"])[0]
    assert bm25["correct"] and not bm25["has_evidence"] and not bm25["reference"]


def test_reader_only_baseline_uses_what_the_reader_saw(tmp_path, ctx):
    d = _run_dir(tmp_path, ctx)
    item = build_items(d, WindowGuard(), methods=["no_memory"])[0]
    assert item["memories"] == [] and not item["has_evidence"] and not item["reference"]
