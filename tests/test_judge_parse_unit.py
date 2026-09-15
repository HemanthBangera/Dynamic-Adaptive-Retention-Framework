"""The Layer B judge's verdict parser and prompt (no LLM calls)."""

import pytest

from core.layer_b.evaluator import PROMPT_TEMPLATE, build_judge_prompt, parse_verdict


@pytest.mark.parametrize("raw, verdict", [
    ("YES", "YES"), ("Yes.", "YES"), ("  yes", "YES"), ("'YES'", "YES"), ("**YES**", "YES"),
    ("NO", "NO"), ("No.", "NO"), ("NO, the memories are unrelated", "NO"), ('"no"', "NO"),
    ("NOT SURE", "NEUTRAL"), ("None of the memories", "NEUTRAL"), ("Nope", "NEUTRAL"),
    ("YESTERDAY", "NEUTRAL"), ("Maybe", "NEUTRAL"), ("", "NEUTRAL"), (None, "NEUTRAL"),
])
def test_parse_verdict_only_accepts_whole_word_yes_no(raw, verdict):
    assert parse_verdict(raw) == verdict


def test_prompt_is_the_documented_template():
    prompt = build_judge_prompt("Where is Rollo from?", "Normandy", "Rollo ruled Normandy.\n{braces} kept")
    assert prompt.startswith("You are the DARS Success Evaluator.\nUSER QUERY: Where is Rollo from?\n")
    assert "RETRIEVED MEMORIES: Rollo ruled Normandy.\n{braces} kept\n" in prompt
    assert prompt.endswith("Respond ONLY with 'YES' or 'NO'. No explanation.")


def test_e5_uses_the_system_prompt_and_parser():
    from benchmarks.dars_eval import run_judge

    assert run_judge.PROMPTS["layer_b"] is PROMPT_TEMPLATE
    assert run_judge.parse_verdict is parse_verdict
