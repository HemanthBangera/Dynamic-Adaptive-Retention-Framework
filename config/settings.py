"""
DARS Configuration Settings
============================
Central configuration for the Dynamic Adaptive Retention Scoring framework.
All tunable hyperparameters are defined here for reproducibility.

Reference: DARS Specification §8 (Scoring Function), §9 (Retention Policy)
"""

import os


class DARSConfig:
    """Master configuration for the DARS Memory Framework."""

    # ── Qdrant Connection ──────────────────────────────────────────────
    QDRANT_URL: str = os.getenv(
        "QDRANT_URL",
        "https://42e448da-2232-49dc-9a5d-f4740f076d38.us-east-1-1.aws.cloud.qdrant.io:6333",
    )
    QDRANT_API_KEY: str = os.getenv(
        "QDRANT_API_KEY",
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        ".eyJhY2Nlc3MiOiJtIn0"
        ".AcM1v7exiTX1uTX81tmhnNjaI1Hc2DWt68x6gRDADGk",
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
