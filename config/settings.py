"""
DARS Configuration Settings
============================
Central configuration for the Dynamic Adaptive Retention Scoring framework.
All tunable hyperparameters are defined here for reproducibility.

Reference: DARS Specification §8 (Scoring Function), §9 (Retention Policy)
"""

import os
from pathlib import Path

from dotenv import load_dotenv


# Load environment variables from project root .env (if present)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


class DARSConfig:
    """Master configuration for the DARS Memory Framework."""

    # ── Qdrant Connection ──────────────────────────────────────────────
    QDRANT_URL: str = os.getenv(
        "QDRANT_URL",
        "",
    )
    QDRANT_API_KEY: str = os.getenv(
        "QDRANT_API_KEY",
        "",
    )

    # ── Collection ─────────────────────────────────────────────────────
    COLLECTION_NAME: str = "dars_memory"
    TEST_COLLECTION_NAME: str = "dars_memory_test"

    # ── Embedding Model ────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    VECTOR_DIMENSION: int = 384          # all-MiniLM-L6-v2 output dim

    # ── DARS Weights (MUST sum to 1.0) ─────────────────────────────────
    #    S = w_r·R + w_f·F + w_u·U + w_p·P
    WEIGHT_RECENCY: float = 0.30         # w_r
    WEIGHT_FREQUENCY: float = 0.20       # w_f
    WEIGHT_UTILITY: float = 0.30         # w_u
    WEIGHT_PREDICTIVE: float = 0.20      # w_p

    # ── DARS Parameters ────────────────────────────────────────────────
    RECENCY_DECAY_LAMBDA: float = 0.01   # λ  – decay rate (per hour)
    FREQUENCY_CAP: int = 50              # Normalisation ceiling for f
    DEFAULT_PREDICTIVE_VALUE: float = 0.5

    # ── Retention Thresholds (§9) ──────────────────────────────────────
    THRESHOLD_RETAIN: float = 0.7        # S > 0.7  → Keep
    THRESHOLD_COMPRESS: float = 0.3      # 0.3 < S ≤ 0.7 → Compress
    #                                      S ≤ 0.3  → Delete

    # ── Search Defaults ────────────────────────────────────────────────
    DEFAULT_FETCH_K: int = 10            # Candidates from vector search
    DEFAULT_TOP_N: int = 3               # Finals after DARS reranking
    RERANK_ALPHA: float = 0.5            # α in  α·sim + (1−α)·DARS

    # ── Distance Metric ────────────────────────────────────────────────
    DISTANCE_METRIC: str = "Cosine"      # Cosine | Euclid | Dot
