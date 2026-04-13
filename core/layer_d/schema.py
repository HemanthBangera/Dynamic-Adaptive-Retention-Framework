"""
DARS Layer D – Data Schema
===========================
Defines the canonical data structures for memory points in the DARS framework.

Each memory is stored as a Qdrant Point comprising:
    • A dense vector  (semantic embedding, 384-dim from all-MiniLM-L6-v2)
    • A structured payload  (DARS metadata: u, f, r, p + bookkeeping fields)

Reference – DARS Specification §15 (Formal Memory Representation):
    m_i = ⟨e_i, s_i, t_i, a_i, u_i, p_i⟩
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List


# ═══════════════════════════════════════════════════════════════════════════════
#  Payload  –  metadata stored alongside each vector in Qdrant
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryPayload:
    """
    DARS Payload Schema for a single memory point.

    Updated by:
        Layer B  → success_count, failure_count, utility, frequency, recency
        Layer C  → is_compressed (after summarisation)
        Layer D  → created_at (on insert)

    Fields
    ------
    text_content   : str    – The factual text of the memory.
    success_count  : int    – Times this memory led to a successful outcome.
    failure_count  : int    – Times this memory was misleading / unhelpful.
    utility        : float  – Derived score  U = success / (success + failure + 1).
    frequency      : int    – Total retrieval count  (access_count).
    recency        : float  – Unix timestamp of last access.
    predictive     : float  – Estimated future relevance  p ∈ [0, 1].
    created_at     : float  – Unix timestamp of creation.
    is_compressed  : bool   – True after Layer C compression.
    source         : str    – Origin tag  ("user" | "agent" | "system").
    tags           : list   – Optional classification labels.
    """

    # ── Core Content ───────────────────────────────────────────────────
    text_content: str

    # ── DARS Parameters ────────────────────────────────────────────────
    success_count: int = 0
    failure_count: int = 0
    utility: float = 0.0

    frequency: int = 0

    recency: float = field(default_factory=time.time)

    predictive: float = 0.5

    # ── Bookkeeping ────────────────────────────────────────────────────
    created_at: float = field(default_factory=time.time)
    is_compressed: bool = False
    source: str = ""
    tags: List[str] = field(default_factory=list)
    original_vector: list[float] | None = field(default=None)

    # ── Serialisation ──────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Flatten to a plain dict for Qdrant ``upsert``."""
        data = asdict(self)
        if data.get("original_vector") is None:
            data.pop("original_vector", None)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MemoryPayload:
        """Reconstruct from a Qdrant payload dict (tolerant of missing keys)."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        parsed_data = {k: v for k, v in data.items() if k in known}
        # Explicitly handle original_vector if needed, although kwargs handles it
        return cls(**parsed_data)

    # ── Derived Values ─────────────────────────────────────────────────

    def compute_utility(self) -> float:
        """Recompute utility using Laplacian Smoothing in-place."""
        total_attempts = self.success_count + self.failure_count
        self.utility = (self.success_count + 1) / (total_attempts + 2)
        return self.utility


# ═══════════════════════════════════════════════════════════════════════════════
#  MemoryPoint  –  complete point (vector + payload) used for I/O
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryPoint:
    """
    Full memory representation: embedding vector + DARS payload.

    Used as the return type from retrieval methods and as input for batch inserts.
    """

    point_id: str                                   # UUID-4 string
    vector: List[float]                             # 384-dim embedding
    payload: MemoryPayload
    score: Optional[float] = None                   # Cosine similarity (search)
    dars_score: Optional[float] = None              # Computed DARS score

    @staticmethod
    def generate_id() -> str:
        """Generate a new UUID-4 point identifier."""
        return str(uuid.uuid4())


# ═══════════════════════════════════════════════════════════════════════════════
#  DARS Weight Vector
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DARSWeights:
    """Tunable weight vector for the DARS scoring function."""

    w_r: float = 0.30   # Recency
    w_f: float = 0.20   # Frequency
    w_u: float = 0.30   # Utility
    w_p: float = 0.20   # Predictive

    def validate(self) -> bool:
        """Return True iff weights sum to 1.0 (within FP tolerance)."""
        return abs(self.w_r + self.w_f + self.w_u + self.w_p - 1.0) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
#  Retention Decision  –  output of the DARS triage classifier
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RetentionDecision:
    """
    Result of DARS retention classification (Layer C triage).

    Possible actions:  "retain" | "compress" | "delete"
    """

    action: str
    dars_score: float
    point_id: str
    text_preview: str        # First 80 chars of text_content
