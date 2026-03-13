"""
DARS Layer D – Unit Tests: Embedding Engine
=============================================
Tests for the EmbeddingEngine wrapper around sentence-transformers.
These tests require the model to download (~80 MB on first run) but
do NOT require Qdrant connectivity.

Coverage:
    • Model loading (lazy singleton)
    • Single-text encoding
    • Batch encoding
    • Vector dimension validation
    • Cosine similarity computation
"""

import pytest
import numpy as np

from core.layer_d.embedding import EmbeddingEngine


# ═══════════════════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def engine():
    """
    Module-scoped embedding engine (loads model once for all tests).
    """
    return EmbeddingEngine("all-MiniLM-L6-v2")


# ═══════════════════════════════════════════════════════════════════════════════
#  Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestEmbeddingEngine:
    """Unit tests for the EmbeddingEngine."""

    def test_dimension(self, engine):
        """Model output dimension should be 384 for all-MiniLM-L6-v2."""
        assert engine.dimension == 384

    def test_encode_returns_list(self, engine):
        """encode() should return a plain Python list of floats."""
        vec = engine.encode("Test sentence")
        assert isinstance(vec, list)
        assert len(vec) == 384
        assert all(isinstance(v, float) for v in vec)

    def test_encode_non_empty(self, engine):
        """Embedding should not be a zero vector for meaningful text."""
        vec = engine.encode("The client prefers Python 3.12")
        magnitude = np.linalg.norm(vec)
        assert magnitude > 0.1, "Embedding magnitude should be non-trivial"

    def test_encode_different_texts_different_vectors(self, engine):
        """Different texts should produce different embeddings."""
        v1 = engine.encode("Python programming language")
        v2 = engine.encode("Italian pasta recipe")
        # Cosine similarity between unrelated texts should be low
        sim = engine.cosine_similarity(v1, v2)
        assert sim < 0.7, f"Unrelated texts should have low similarity, got {sim}"

    def test_encode_similar_texts_high_similarity(self, engine):
        """Semantically similar texts should have high cosine similarity."""
        v1 = engine.encode("The budget for the project is $50,000")
        v2 = engine.encode("Project funding is fifty thousand dollars")
        sim = engine.cosine_similarity(v1, v2)
        assert sim > 0.5, f"Similar texts should have high similarity, got {sim}"

    def test_encode_batch(self, engine):
        """Batch encoding should return one vector per input text."""
        texts = [
            "First sentence",
            "Second sentence",
            "Third sentence about budgets",
        ]
        vectors = engine.encode_batch(texts)
        assert len(vectors) == 3
        assert all(len(v) == 384 for v in vectors)

    def test_encode_batch_single(self, engine):
        """Batch encoding with one item should work."""
        vectors = engine.encode_batch(["Solo text"])
        assert len(vectors) == 1
        assert len(vectors[0]) == 384

    def test_encode_batch_matches_single(self, engine):
        """Batch result should match individual encode results."""
        text = "Consistency check for encoding"
        single = engine.encode(text)
        batch = engine.encode_batch([text])
        # Allow tiny floating-point differences
        diff = np.abs(np.array(single) - np.array(batch[0]))
        assert np.max(diff) < 1e-5

    def test_cosine_similarity_identical(self, engine):
        """Cosine similarity of a vector with itself should be ~1.0."""
        vec = engine.encode("test vector")
        sim = engine.cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 1e-5

    def test_cosine_similarity_zero_vector(self, engine):
        """Cosine similarity with a zero vector should return 0.0."""
        vec = engine.encode("some text")
        zero = [0.0] * 384
        sim = engine.cosine_similarity(vec, zero)
        assert sim == 0.0

    def test_cosine_similarity_range(self, engine):
        """Cosine similarity should be in [-1, 1]."""
        v1 = engine.encode("apple fruit")
        v2 = engine.encode("machine learning algorithms")
        sim = engine.cosine_similarity(v1, v2)
        assert -1.0 <= sim <= 1.0

    def test_singleton_pattern(self):
        """Multiple EmbeddingEngine instances with same model should share state."""
        e1 = EmbeddingEngine("all-MiniLM-L6-v2")
        e2 = EmbeddingEngine("all-MiniLM-L6-v2")
        assert e1 is e2
