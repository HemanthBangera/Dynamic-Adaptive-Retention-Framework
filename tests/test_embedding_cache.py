"""
Unit tests for the optional on-disk embedding cache (``DARS_EMBED_CACHE``).

The cache must (1) return the same vectors the model produces, (2) serve repeats
without calling the model, (3) persist across processes/instances, and
(4) never mix vectors from different model namespaces.  Runs locally with the
cached all-MiniLM-L6-v2 snapshot; no credentials needed.
"""

import numpy as np
import pytest

from core.layer_d.embedding import CACHE_ENV, EmbeddingEngine, VectorCache

TEXTS = [
    "The client prefers Python 3.12.",
    "Document 3: the Eiffel Tower is in Paris.",
    "The client prefers Python 3.12.",          # duplicate inside one batch
    "Where can I find a mug?",
]


@pytest.fixture()
def engine(tmp_path, monkeypatch):
    monkeypatch.setenv(CACHE_ENV, str(tmp_path / "emb.sqlite"))
    eng = EmbeddingEngine()
    eng._vcache = None
    yield eng
    eng._vcache = None


def test_cache_matches_model_and_serves_repeats(engine, monkeypatch):
    first = np.asarray(engine.encode_batch(TEXTS))
    uncached = np.asarray(engine._model.encode(TEXTS, convert_to_numpy=True))
    assert first.shape == (4, 384)
    np.testing.assert_allclose(first, uncached, atol=1e-5)
    np.testing.assert_array_equal(first[0], first[2])
    stats = engine.cache_stats()
    assert stats["misses"] == 4 and stats["hits"] == 0

    def boom(*_a, **_k):
        raise AssertionError("model called on a cache hit")

    monkeypatch.setattr(engine._model, "encode", boom)
    again = np.asarray(engine.encode_batch(TEXTS))
    np.testing.assert_array_equal(again, first)           # bit-identical replay
    np.testing.assert_array_equal(np.asarray(engine.encode(TEXTS[1])), first[1])
    assert engine.cache_stats()["hits"] == 5


def test_cache_persists_and_is_namespaced(tmp_path):
    path = tmp_path / "v.sqlite"
    a = VectorCache(path, "model-a")
    vec = np.arange(384, dtype=np.float32)
    a.put_many([(a.key("hello"), vec)])
    reopened = VectorCache(path, "model-a")
    np.testing.assert_array_equal(reopened.get_many([reopened.key("hello")])[reopened.key("hello")], vec)
    other = VectorCache(path, "model-b")
    assert other.get_many([other.key("hello")]) == {}


def test_cache_off_without_env(monkeypatch):
    monkeypatch.delenv(CACHE_ENV, raising=False)
    eng = EmbeddingEngine()
    eng._vcache = None
    assert eng.cache_stats() is None
    assert len(eng.encode("plain uncached call")) == 384
