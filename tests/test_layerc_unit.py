"""Unit tests for the E6 Layer C helpers (no LLM calls; the extractive test uses MiniLM)."""

import pytest

from benchmarks.dars_eval.run_layerc import (
    ENC,
    answer_retained,
    extractive_compress,
    matched_rate,
    variant_names,
)


def test_variant_names_add_matched_baselines_only_with_semantic():
    assert variant_names(("semantic", "llmlingua2", "extractive")) == [
        "semantic", "llmlingua2", "extractive", "llmlingua2@matched", "extractive@matched"]
    assert variant_names(("semantic", "extractive")) == ["semantic", "extractive", "extractive@matched"]
    assert variant_names(("extractive",)) == ["extractive"]


def test_matched_rate_is_clipped_ratio():
    assert matched_rate(100, 44) == pytest.approx(0.44)
    assert matched_rate(100, 1) == pytest.approx(0.05)
    assert matched_rate(100, 120) == pytest.approx(0.95)
    assert matched_rate(0, 5) == pytest.approx(0.95)


def test_answer_retained_normalises():
    assert answer_retained("Rollo was the first ruler of Normandy.", ["the Normandy"])
    assert not answer_retained("Rollo was a Viking.", ["Normandy"])


def test_extractive_keeps_order_and_respects_budget():
    from core.layer_d.embedding import EmbeddingEngine

    text = ("The Normans settled in northern France. Rollo was their first ruler. "
            "The region became known as Normandy. Its capital was Rouen. "
            "Norman knights later conquered England in 1066.")
    short = extractive_compress(text, EmbeddingEngine(), ratio=0.3)
    half = extractive_compress(text, EmbeddingEngine(), ratio=0.5)
    assert len(ENC.encode(short)) <= len(ENC.encode(half)) <= len(ENC.encode(text))
    kept = [s.strip() for s in short.split(". ") if s.strip()]
    positions = [text.find(s.rstrip(".")) for s in kept]
    assert positions == sorted(positions)            # original sentence order preserved
