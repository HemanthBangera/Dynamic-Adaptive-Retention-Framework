"""
DARS Layer D – Embedding Engine
================================
Wraps the ``sentence-transformers`` library to provide deterministic
text → vector conversion using **all-MiniLM-L6-v2** (384 dimensions).

This module is used by:
    • Layer D  – to embed text before storing in Qdrant
    • Layer A  – to embed user queries before semantic search

Architecture Reference:
    Embedding Model (all-MiniLM-L6-v2) sits between Query Reformulator
    and Qdrant Search in the Layer A pipeline.

Optional vector cache
---------------------
When the environment variable ``DARS_EMBED_CACHE`` names a file, vectors are
stored in (and served from) a content-addressed SQLite cache keyed by
sha256(model name, cached HF snapshot revision, max_seq_length, text).  It is
off by default.  The evaluation harness enables it so that repeated runs over
the same memory units (dev/test splits, reader repeats, ablations) embed each
text once and reuse bit-identical vectors.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

CACHE_ENV = "DARS_EMBED_CACHE"


def _hf_revision(model_name: str) -> str:
    """Commit hash of the locally cached HF snapshot of ``model_name`` (or 'unknown')."""
    repo = model_name if "/" in model_name else f"sentence-transformers/{model_name}"
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        ref = Path(HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "refs" / "main"
        return ref.read_text(encoding="utf-8").strip()
    except Exception:
        return "unknown"


class VectorCache:
    """Content-addressed float32 vector store in SQLite (safe across processes)."""

    def __init__(self, path: Union[str, Path], namespace: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), timeout=120, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, dim INTEGER NOT NULL, vec BLOB NOT NULL)"
        )
        self._conn.commit()

    def key(self, text: str) -> str:
        return hashlib.sha256(f"{self.namespace}\x00{text}".encode("utf-8")).hexdigest()

    def get_many(self, keys: Sequence[str]) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        with self._lock:
            for lo in range(0, len(keys), 500):
                chunk = list(keys[lo:lo + 500])
                rows = self._conn.execute(
                    f"SELECT key, dim, vec FROM vectors WHERE key IN ({','.join('?' * len(chunk))})", chunk
                )
                for k, dim, blob in rows:
                    out[k] = np.frombuffer(blob, dtype=np.float32, count=dim)
        return out

    def put_many(self, items: Sequence[Tuple[str, np.ndarray]]) -> None:
        rows = [(k, int(v.shape[0]), np.asarray(v, dtype=np.float32).tobytes()) for k, v in items]
        with self._lock:
            self._conn.executemany("INSERT OR IGNORE INTO vectors (key, dim, vec) VALUES (?, ?, ?)", rows)
            self._conn.commit()

    def stats(self) -> Dict[str, Union[str, int]]:
        return {"path": str(self.path), "namespace": self.namespace, "hits": self.hits, "misses": self.misses}


class EmbeddingEngine:
    """
    Singleton-style embedding wrapper around ``sentence-transformers``.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.  Default: ``all-MiniLM-L6-v2``.

    Usage
    -----
    >>> engine = EmbeddingEngine()
    >>> vec = engine.encode("The client prefers Python 3.12")
    >>> len(vec)
    384
    """

    _instance: "EmbeddingEngine | None" = None
    _model = None

    def __new__(cls, model_name: str = "all-MiniLM-L6-v2"):
        """Ensure only one model instance is loaded into memory."""
        if cls._instance is None or cls._instance._model_name != model_name:
            cls._instance = super().__new__(cls)
            cls._instance._model_name = model_name
            cls._instance._model = None
            cls._instance._vcache = None
        return cls._instance

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name

    # ── Lazy Loading ───────────────────────────────────────────────────

    def _load_model(self):
        """Load the sentence-transformer model on first use."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s ...", self._model_name)

            hf_token = os.getenv("HF_TOKEN")
            if hf_token:
                self._model = SentenceTransformer(self._model_name, token=hf_token)
                logger.info("Embedding model loaded using HF_TOKEN. Dimension: %d", self.dimension)
            else:
                self._model = SentenceTransformer(self._model_name)
                logger.info("Embedding model loaded. Dimension: %d", self.dimension)
        except ImportError:
            raise ImportError(
                "sentence-transformers is required.  "
                "Install via:  pip install sentence-transformers"
            )

    def _cache(self) -> Optional[VectorCache]:
        """The vector cache named by ``DARS_EMBED_CACHE``, or None when caching is off."""
        path = os.getenv(CACHE_ENV, "").strip()
        if not path:
            return None
        if self._vcache is None or self._vcache.path != Path(path):
            self._load_model()
            namespace = (f"{self._model_name}|rev={_hf_revision(self._model_name)}"
                         f"|max_seq_length={self._model.max_seq_length}")
            self._vcache = VectorCache(path, namespace)
            logger.info("Embedding cache enabled at %s (%s)", path, namespace)
        return self._vcache

    def cache_stats(self) -> Optional[Dict[str, Union[str, int]]]:
        """Hit/miss counts of the active vector cache (None when caching is off)."""
        cache = self._cache()
        return cache.stats() if cache is not None else None

    # ── Public API ─────────────────────────────────────────────────────

    @property
    def dimension(self) -> int:
        """Return the output vector dimensionality."""
        self._load_model()
        return self._model.get_sentence_embedding_dimension()

    def encode(self, text: str) -> List[float]:
        """
        Encode a single text string into a dense vector.

        Parameters
        ----------
        text : str
            Input text to embed.

        Returns
        -------
        List[float]
            384-dimensional embedding vector.
        """
        self._load_model()
        cache = self._cache()
        if cache is None:
            return self._model.encode(text, convert_to_numpy=True).tolist()
        key = cache.key(text)
        hit = cache.get_many([key]).get(key)
        if hit is not None:
            cache.hits += 1
            return hit.tolist()
        cache.misses += 1
        vector = np.asarray(self._model.encode(text, convert_to_numpy=True), dtype=np.float32)
        cache.put_many([(key, vector)])
        return vector.tolist()

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Encode a batch of texts into dense vectors.

        Parameters
        ----------
        texts : List[str]
            Input texts.
        batch_size : int
            Processing batch size (default 32).

        Returns
        -------
        List[List[float]]
            List of 384-dimensional embedding vectors.
        """
        self._load_model()
        cache = self._cache()
        if cache is None:
            return self._model.encode(texts, batch_size=batch_size, convert_to_numpy=True).tolist()
        keys = [cache.key(t) for t in texts]
        found = cache.get_many(list(dict.fromkeys(keys)))
        missing: Dict[str, str] = {}
        for k, t in zip(keys, texts):
            if k not in found and k not in missing:
                missing[k] = t
        cache.hits += sum(1 for k in keys if k in found)
        cache.misses += len(keys) - sum(1 for k in keys if k in found)
        if missing:
            vectors = np.asarray(
                self._model.encode(list(missing.values()), batch_size=batch_size, convert_to_numpy=True),
                dtype=np.float32,
            )
            new = list(zip(missing.keys(), vectors))
            cache.put_many(new)
            found.update(new)
        return [found[k].tolist() for k in keys]

    def cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.

        Used by Layer C for semantic drift detection and
        by the Goal Alignment module for dynamic p-score updates.
        """
        a = np.array(vec_a)
        b = np.array(vec_b)
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 0.0
        return float(dot / norm)
