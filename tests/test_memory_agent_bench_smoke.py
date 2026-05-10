"""Offline smoke tests for MemoryAgentBench driver wiring."""

from third_party.memoryagentbench_eval import drqa_exact_match_score, get_template
from benchmarks.memory_agent_bench.qa_builder import MAB_AGENT_TEMPLATE_KEY


def test_get_template_ruler_qa_agentic():
    t = get_template("ruler_qa1_197K", "query", MAB_AGENT_TEMPLATE_KEY)
    assert "{question}" in t or "Question" in t


def test_exact_match_score():
    assert drqa_exact_match_score("France", "france") is True
