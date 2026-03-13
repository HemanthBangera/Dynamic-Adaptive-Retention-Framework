"""
DARS Layer D – Shared Test Fixtures
=====================================
Provides reusable fixtures for both unit and integration tests.
"""

import time
import uuid
import pytest

from config.settings import DARSConfig
from core.layer_d.schema import MemoryPayload, MemoryPoint, DARSWeights


# ═══════════════════════════════════════════════════════════════════════════════
#  Connectivity helper
# ═══════════════════════════════════════════════════════════════════════════════

def _qdrant_reachable() -> bool:
    """Return True if the Qdrant cloud cluster responds."""
    try:
        from qdrant_client import QdrantClient
        cfg = DARSConfig()
        client = QdrantClient(url=cfg.QDRANT_URL, api_key=cfg.QDRANT_API_KEY, timeout=10)
        client.get_collections()
        return True
    except Exception:
        return False


# Evaluate once per session, cache result
_QDRANT_OK = None

def qdrant_available() -> bool:
    global _QDRANT_OK
    if _QDRANT_OK is None:
        _QDRANT_OK = _qdrant_reachable()
    return _QDRANT_OK


# Skip-marker for integration tests
requires_qdrant = pytest.mark.skipif(
    not qdrant_available(),
    reason="Qdrant cloud cluster unreachable – skipping integration test",
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Config fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def config():
    """Return a DARSConfig instance."""
    return DARSConfig()


@pytest.fixture
def test_collection_name():
    """Generate a unique collection name for test isolation."""
    return f"dars_test_{uuid.uuid4().hex[:8]}"


# ═══════════════════════════════════════════════════════════════════════════════
#  Schema fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_payload():
    """Return a realistic MemoryPayload for testing."""
    now = time.time()
    return MemoryPayload(
        text_content="The client prefers Python 3.12 for all backend services.",
        success_count=3,
        failure_count=1,
        utility=0.6,
        frequency=5,
        recency=now,
        predictive=0.7,
        created_at=now - 86400,  # created 1 day ago
        is_compressed=False,
        source="user",
        tags=["python", "preference"],
    )


@pytest.fixture
def fresh_payload():
    """Return a brand-new memory payload (zero scores)."""
    now = time.time()
    return MemoryPayload(
        text_content="New observation with no history.",
        recency=now,
        created_at=now,
    )


@pytest.fixture
def default_weights():
    """Return the default DARS weight vector."""
    return DARSWeights()


@pytest.fixture
def sample_memories_data():
    """Return a list of dict inputs suitable for batch-insert."""
    return [
        {
            "text": "The project deadline is March 30, 2026.",
            "predictive_value": 0.8,
            "source": "user",
            "tags": ["deadline", "timeline"],
        },
        {
            "text": "The client prefers Python 3.12 for backend services.",
            "predictive_value": 0.7,
            "source": "user",
            "tags": ["python", "preference"],
        },
        {
            "text": "Budget for Phase 1 is capped at $50,000.",
            "predictive_value": 0.6,
            "source": "agent",
            "tags": ["budget", "finance"],
        },
        {
            "text": "Use PostgreSQL for the main relational database.",
            "predictive_value": 0.5,
            "source": "user",
            "tags": ["database", "postgres"],
        },
        {
            "text": "The team meets every Monday at 10 AM EST.",
            "predictive_value": 0.4,
            "source": "system",
            "tags": ["meeting", "schedule"],
        },
    ]
