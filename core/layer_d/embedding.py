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
"""

from __future__ import annotations

import logging
from typing import List, Union

import numpy as np

logger = logging.getLogger(__name__)


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
        return cls._instance

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name

    # ── Lazy Loading ───────────────────────────────────────────────────

    def _load_model(self):
        """Load the sentence-transformer model on first use."""
        import os
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
        vector = self._model.encode(text, convert_to_numpy=True)
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
        vectors = self._model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
        return vectors.tolist()

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
