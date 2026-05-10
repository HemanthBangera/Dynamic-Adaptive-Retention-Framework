"""
Vendored evaluation helpers from MemoryAgentBench (MIT).

Upstream: https://github.com/HUST-AI-HYZ/MemoryAgentBench
Snapshot: utils/eval_other_utils.py, utils/templates.py from branch `main`
(see VENDOR.md for pin notes; prefer pinning a commit SHA in run manifests).
"""

from .eval_other_utils import (
    calculate_metrics,
    chunk_text_into_sentences,
    count_tokens,
    drqa_exact_match_score,
    metrics_summarization,
    parse_output,
    post_process,
)
from .templates import get_template

__all__ = [
    "calculate_metrics",
    "chunk_text_into_sentences",
    "count_tokens",
    "drqa_exact_match_score",
    "get_template",
    "metrics_summarization",
    "parse_output",
    "post_process",
]
