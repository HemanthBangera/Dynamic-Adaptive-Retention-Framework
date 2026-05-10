"""Token-bounded sentence chunking (vendored MemoryAgentBench logic)."""

from __future__ import annotations

import logging

import nltk

from third_party.memoryagentbench_eval import chunk_text_into_sentences

logger = logging.getLogger(__name__)


def ensure_nltk_punkt() -> None:
    """Download punkt / punkt_tab if missing (NLTK 3.8+)."""
    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            logger.info("Downloading NLTK tokenizer package: %s", pkg)
            nltk.download(pkg, quiet=True)


def chunk_context(text: str, chunk_size: int, tiktoken_model: str = "gpt-4o-mini") -> list[str]:
    ensure_nltk_punkt()
    return chunk_text_into_sentences(text, model_name=tiktoken_model, chunk_size=chunk_size)
