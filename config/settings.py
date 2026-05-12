"""
DARS Configuration Settings
============================
Central configuration for the Dynamic Adaptive Retention Scoring framework.
All tunable hyperparameters are defined here for reproducibility.

Reference: DARS Specification §8 (Scoring Function), §9 (Retention Policy)
"""

import logging
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
    RECENCY_DECAY_LAMBDA: float = 0.005  # λ  – decay rate (per hour); R ≈ 0.76 after 55 h
    FREQUENCY_CAP: int = 50              # Normalisation ceiling for f
    DEFAULT_PREDICTIVE_VALUE: float = 0.5  # fail-safe when GOAL_VECTOR cannot be computed

    # ── Goal Alignment (Predictive Component P) ───────────────────────
    #    Per-group presets avoid semantic dilution from a generic hybrid string.
    #    Set TRAINING_GROUP in .env to select a preset, or override with GOAL_DESCRIPTION.
    GOAL_PRESETS: dict[str, str] = {
        "MSC": (
            "Personal facts, preferences, and recurring conversation topics "
            "that maintain long-term dialogue coherence and social understanding"
        ),
        "ALFWorld": (
            "Effective action sequences, object locations, and task completion "
            "strategies for interactive household environments"
        ),
        "Narrative": (
            "Long-form narrative coherence: plot progression, character state, "
            "temporal ordering of events, and the most recent scene consistent with the story"
        ),
    }
    TRAINING_GROUP: str = os.getenv("TRAINING_GROUP", "ALFWorld")
    GOAL_DESCRIPTION: str = os.getenv("GOAL_DESCRIPTION", "")
    _goal_vector_cache: list[float] | None = None

    @classmethod
    def _resolve_goal_description(cls) -> str:
        """Return the active goal description: explicit env var > preset > ALFWorld default."""
        if cls.GOAL_DESCRIPTION:
            return cls.GOAL_DESCRIPTION
        return cls.GOAL_PRESETS.get(cls.TRAINING_GROUP, cls.GOAL_PRESETS["ALFWorld"])

    @classmethod
    def get_goal_vector(cls) -> list[float] | None:
        """Lazily encode GOAL_DESCRIPTION into a 384-dim vector. Returns None on failure."""
        if cls._goal_vector_cache is not None:
            return cls._goal_vector_cache
        try:
            from core.layer_d.embedding import EmbeddingEngine
            desc = cls._resolve_goal_description()
            cls._goal_vector_cache = EmbeddingEngine().encode(desc)
            return cls._goal_vector_cache
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "GOAL_VECTOR computation failed (%s); "
                "predictive will use DEFAULT_PREDICTIVE_VALUE=%.2f",
                exc, cls.DEFAULT_PREDICTIVE_VALUE,
            )
            return None
    
    # ── Layer C Resilience ─────────────────────────────────────────────
    SHUTDOWN_TIMEOUT_SECONDS: float = 15.0
    MAX_CONCURRENT_DISTILLATIONS: int = 3

    # ── Retention Thresholds (§9) ──────────────────────────────────────
    THRESHOLD_RETAIN: float = 0.7        # S > 0.7  → Keep
    THRESHOLD_COMPRESS: float = 0.3      # 0.3 < S ≤ 0.7 → Compress
    #                                      S ≤ 0.3  → Delete

    # ── Search Defaults ────────────────────────────────────────────────
    DEFAULT_FETCH_K: int = 10            # Candidates from vector search
    DEFAULT_TOP_N: int = 3               # Finals after DARS reranking
    RERANK_ALPHA: float = 0.5            # α in  α·sim + (1−α)·DARS

    # ── Gemini API Resilience ──────────────────────────────────────────
    GEMINI_TIMEOUT: float = 30.0          # HTTP timeout (seconds)
    GEMINI_MAX_RETRIES: int = 2           # Retry attempts on transient failures
    GEMINI_MAX_EXPANSION_CHARS: int = 500 # Cap reformulator output length

    # ── Distance Metric ────────────────────────────────────────────────
    DISTANCE_METRIC: str = "Cosine"      # Cosine | Euclid | Dot

    # ── MemoryAgentBench driver defaults (benchmarks/memory_agent_bench) ─
    MAB_HF_REVISION: str = os.getenv("MAB_HF_REVISION", "main")
    MAB_TIKTOKEN_MODEL: str = os.getenv("MAB_TIKTOKEN_MODEL", "gpt-4o-mini")
    # Virtual clock: advance recency/created_at per chunk so hour-scale λ decay is meaningful in fast replays
    MAB_USE_VIRTUAL_TIME: bool = os.getenv("MAB_USE_VIRTUAL_TIME", "").lower() in ("1", "true", "yes")
    MAB_VIRTUAL_TIME_STEP_S: float = float(os.getenv("MAB_VIRTUAL_TIME_STEP_S", "3600"))
    # Acquisition phase: mild Laplace prior so first-query retrieval is not stuck at zero-utility cold start
    MAB_INJECTION_INITIAL_SUCCESS: int = int(os.getenv("MAB_INJECTION_INITIAL_SUCCESS", "0"))
    # Tombstone / neighbor expansion (MemoryAgentBench narrative stack)
    MAB_TOMBSTONE_SIM_THRESHOLD: float = float(os.getenv("MAB_TOMBSTONE_SIM_THRESHOLD", "0.88"))
    MAB_EXPAND_NEIGHBOR_CHUNKS: bool = os.getenv(
        "MAB_EXPAND_NEIGHBOR_CHUNKS", "1"
    ).lower() in ("1", "true", "yes")
    # Merge retrieval from reformulated + raw query (narrative profile enables in apply_mab_narrative_profile)
    MAB_DUAL_QUERY_RETRIEVAL: bool = os.getenv(
        "MAB_DUAL_QUERY_RETRIEVAL", "0"
    ).lower() in ("1", "true", "yes")


def apply_mab_narrative_profile() -> None:
    """
    Aggressive DARS + goal preset for narrative / EventQA-style benchmarks.
    Call before constructing MemoryVault for a MemoryAgentBench run.
    """
    DARSConfig.WEIGHT_RECENCY = 0.6
    DARSConfig.WEIGHT_FREQUENCY = 0.1
    DARSConfig.WEIGHT_UTILITY = 0.1
    DARSConfig.WEIGHT_PREDICTIVE = 0.2
    DARSConfig.RECENCY_DECAY_LAMBDA = 0.01
    DARSConfig.TRAINING_GROUP = "Narrative"
    DARSConfig.DEFAULT_FETCH_K = max(DARSConfig.DEFAULT_FETCH_K, 25)
    DARSConfig.DEFAULT_TOP_N = max(DARSConfig.DEFAULT_TOP_N, 5)
    DARSConfig.MAB_USE_VIRTUAL_TIME = True
    DARSConfig.MAB_DUAL_QUERY_RETRIEVAL = True
    DARSConfig.validate_and_normalize()
    DARSConfig._goal_vector_cache = None  # type: ignore[attr-defined]


DARSConfig.validate_and_normalize()
