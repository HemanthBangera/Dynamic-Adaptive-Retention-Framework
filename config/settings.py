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

    # ── Gemini API Integration ─────────────────────────────────────────
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_PROJECT_NUMBER: str = os.getenv("GEMINI_PROJECT_NUMBER", "805928140149")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

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

    @classmethod
    def validate_and_normalize(cls) -> None:
        """Mathematically normalize DARS weights to guarantee they sum to 1.0 (with precision epsilon)."""
        total = (
            cls.WEIGHT_RECENCY +
            cls.WEIGHT_FREQUENCY +
            cls.WEIGHT_UTILITY +
            cls.WEIGHT_PREDICTIVE
        )
        # Using a tighter epsilon (1e-7) to account for floating-point inaccuracies
        # while preventing false positives during Ablation Studies with tiny weights.
        if abs(total - 1.0) > 1e-7 and total > 0:
            cls.WEIGHT_RECENCY /= total
            cls.WEIGHT_FREQUENCY /= total
            cls.WEIGHT_UTILITY /= total
            cls.WEIGHT_PREDICTIVE /= total

    # ── DARS Parameters ────────────────────────────────────────────────
    RECENCY_DECAY_LAMBDA: float = 0.01   # λ  – decay rate (per hour)
    FREQUENCY_CAP: int = 50              # Normalisation ceiling for f
    DEFAULT_PREDICTIVE_VALUE: float = 0.5
    GOAL_VECTOR: list[float] = [0.0] * 384

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

DARSConfig.validate_and_normalize()
