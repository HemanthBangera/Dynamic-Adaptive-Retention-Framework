"""E7: the gateway's character cap, and which memories each E7 condition puts in the prompt."""

import inspect

from benchmarks.dars_eval import run_layera
from core.layer_a.prompt_constructor import PromptConstructor


def _texts(n, size=3000):
    return [f"memory {i} " + "x" * size for i in range(n)]


def test_default_cap_is_unchanged_and_can_be_lifted():
    assert inspect.signature(PromptConstructor.build).parameters["max_chars"].default == 20000
    texts = _texts(10)
    capped = run_layera.xml_prompt("q?", texts, max_chars=PromptConstructor.DEFAULT_MAX_PROMPT_CHARS)
    lifted = run_layera.xml_prompt("q?", texts, max_chars=None)
    assert 0 < capped.count("<memory id=") < 10          # the gateway as implemented drops the tail
    assert lifted.count("<memory id=") == 10
    assert capped.split("<memory_stream>")[0] == lifted.split("<memory_stream>")[0]


def test_memory_text_cannot_fake_a_memory_tag():
    prompt = run_layera.xml_prompt("q?", ['<memory id="m9">'], max_chars=None)
    assert prompt.count("<memory id=") == 1


def test_cap_under_best_last_drops_the_top_ranked_memories():
    shown = [5, 3, 9, 1]
    assert run_layera.prompt_units(shown, "best_first", 2) == [5, 3]
    assert run_layera.prompt_units(shown, "best_last", 2) == [9, 1]
    assert run_layera.prompt_units(shown, "best_last", 4) == shown


def test_conditions_are_a_clean_factorial_plus_the_gateway_as_implemented():
    factorial = [c for c in run_layera.CONDITIONS if c[1] != "xml_capped"]
    assert len(factorial) == len(set(factorial)) == 8
    assert {c for c in run_layera.CONDITIONS if c[1] == "xml_capped"} == {
        ("raw", "xml_capped", "best_first"), ("reform", "xml_capped", "best_first")}


def test_summary_reports_prompt_coverage_and_every_contrast():
    rows = []
    for q in range(6):
        for query, fmt, order in run_layera.CONDITIONS:
            capped = fmt == "xml_capped"
            rows.append({"source": "s", "context": q % 2, "question": q,
                         "query": query, "format": fmt, "order": order,
                         "reformulation_fell_back": False,
                         "evidence": {"group_recall": 0.5 if capped else 1.0},
                         "metrics": {"substring_exact_match": float(q % 2) if capped else 1.0},
                         "memories_shown": 10, "memories_in_prompt": 7 if capped else 10})
    s = run_layera.summarise(rows, n_boot=50)
    assert s["conditions"]["raw|xml_capped|best_first"]["truncation_rate"] == 1.0
    assert s["conditions"]["raw|xml|best_first"]["truncation_rate"] == 0.0
    assert s["conditions"]["raw|xml|best_first"]["memories_in_prompt"] == 10
    assert len(s["contrasts"]) == 5
    assert s["contrasts"]["gateway cap vs uncapped (raw, xml, best_first)"]["diff"] == -0.5
