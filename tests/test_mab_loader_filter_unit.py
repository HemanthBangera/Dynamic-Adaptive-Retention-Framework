"""Loader token-band filter (HF load mocked)."""

from __future__ import annotations

from unittest.mock import patch

from datasets import Dataset

from benchmarks.memory_agent_bench import loader as loader_mod


def test_load_mab_filtered_stats_single_row():
    tiny_table = Dataset.from_dict(
        {
            "context": ["hello world benchmark context text"],
            "questions": [["q?"]],
            "answers": [["a"]],
            "metadata": [{"source": "src1"}],
        }
    )
    with patch.object(loader_mod, "_load_split_table", return_value=tiny_table):
        rows, stats = loader_mod.load_mab_filtered(
            "Accurate_Retrieval",
            "src1",
            max_test_samples=None,
            seed=0,
            revision="main",
            min_context_tokens=None,
            max_context_tokens=None,
            tiktoken_model="gpt-4o-mini",
        )
    assert stats["rows_after_source"] == 1
    assert stats["rows_after_token_filter"] == 1
    assert stats["rows_returned"] == 1
    assert len(rows) == 1
    assert "context" in rows[0]
