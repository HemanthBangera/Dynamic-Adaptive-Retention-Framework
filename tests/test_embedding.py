"""
DARS Embedding Engine — Real Model Tests
==========================================
Tests the sentence-transformers embedding engine with the real model.
"""

import pytest
import numpy as np
from core.layer_d.embedding import EmbeddingEngine


class TestEmbeddingEngine:

    def test_dimension_is_384(self, embedder):
        assert embedder.dimension == 384

    def test_encode_returns_list_of_floats(self, embedder):
        vec = embedder.encode("Hello world")
        assert isinstance(vec, list)
        assert len(vec) == 384
        assert all(isinstance(v, float) for v in vec)

    def test_different_texts_different_vectors(self, embedder):
        v1 = embedder.encode("Python is a programming language")
        v2 = embedder.encode("The weather is sunny today")
        assert v1 != v2

    def test_similar_texts_high_similarity(self, embedder):
        v1 = embedder.encode("Machine learning is a subset of AI")
        v2 = embedder.encode("ML is part of artificial intelligence")
        sim = embedder.cosine_similarity(v1, v2)
        assert sim > 0.5

    def test_encode_batch(self, embedder):
        texts = ["Hello", "World", "Test"]
        vectors = embedder.encode_batch(texts)
        assert len(vectors) == 3
        assert all(len(v) == 384 for v in vectors)

    def test_batch_matches_single(self, embedder):
        texts = ["Hello world", "Goodbye world"]
        batch = embedder.encode_batch(texts)
        singles = [embedder.encode(t) for t in texts]
        for b, s in zip(batch, singles):
            sim = embedder.cosine_similarity(b, s)
            assert sim > 0.999

    def test_cosine_similarity_identical(self, embedder):
        v = embedder.encode("Test")
        assert abs(embedder.cosine_similarity(v, v) - 1.0) < 0.001

    def test_cosine_similarity_zero_vector(self, embedder):
        v = embedder.encode("Test")
        zero = [0.0] * 384
        assert embedder.cosine_similarity(v, zero) == 0.0

    def test_singleton_pattern(self):
        e1 = EmbeddingEngine()
        e2 = EmbeddingEngine()
        assert e1 is e2
