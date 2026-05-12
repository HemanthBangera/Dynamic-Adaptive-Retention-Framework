"""Token-bounded sentence chunking (vendored MemoryAgentBench logic)."""

from __future__ import annotations

import logging

import nltk
import tiktoken

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


def chunk_context(
    text: str,
    chunk_size: int,
    tiktoken_model: str = "gpt-4o-mini",
    overlap_tokens: int = 0,
) -> list[str]:
    """
    Sentence-bounded chunks up to ``chunk_size`` tokens.

    When ``overlap_tokens`` > 0, each chunk after the first is prefixed with the
    tail of the previous chunk (token overlap) to reduce boundary-only answers.
    """
    ensure_nltk_punkt()
    base = chunk_text_into_sentences(text, model_name=tiktoken_model, chunk_size=chunk_size)
    if overlap_tokens <= 0 or len(base) <= 1:
        return base
    try:
        enc = tiktoken.encoding_for_model(tiktoken_model)
    except KeyError:
        enc = tiktoken.encoding_for_model("gpt-4o-mini")
    out: list[str] = [base[0]]
    for i in range(1, len(base)):
        prev = base[i - 1]
        toks = enc.encode(prev, disallowed_special=())
        tail = enc.decode(toks[-overlap_tokens:]) if len(toks) > overlap_tokens else prev
        merged = (tail.strip() + " " + base[i].strip()).strip()
        out.append(merged)
    return out
