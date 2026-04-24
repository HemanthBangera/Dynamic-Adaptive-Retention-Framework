"""
DARS Layer D – Integration Tests: Storage Engine (Qdrant Cloud)
=================================================================
End-to-end tests that operate on the real Qdrant cloud cluster.
These tests create a temporary collection, exercise the full
MemoryVault API, then clean up.

**Automatically skipped** if the Qdrant cluster is unreachable.

Coverage:
    • Collection lifecycle (create / info / delete)
    • Single memory store + retrieve
    • Batch memory insert
    • Semantic search (pure vector similarity)
    • Two-stage search and rerank
    • Atomic payload patching (recency, frequency, utility)
    • Combined retrieval update (Layer B workflow)
    • Memory deletion (single + batch)
    • Full triage scan (Layer C workflow)
    • Health check endpoint
"""

import time
import uuid
import pytest

from config.settings import DARSConfig
from core.layer_d.storage import MemoryVault
from core.layer_d.schema import MemoryPayload, MemoryPoint

from tests.conftest import requires_qdrant


# ═══════════════════════════════════════════════════════════════════════════════
#  Fixtures (integration-specific)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def vault():
    """
    Module-scoped MemoryVault with an isolated test collection.
    Created once, cleaned up after all tests in this module.
    """
    config = DARSConfig()
    collection = f"dars_integration_{uuid.uuid4().hex[:8]}"
    v = MemoryVault(config=config, collection_name=collection)
    v.initialize_collection(recreate=True)
    yield v
    # Teardown: remove the test collection
    try:
        v.delete_collection()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def clean_vault(vault, request):
    """
    Clear all points between tests to simulate a fresh collection
    without the IO overhead of re-creating it on Qdrant Cloud.
    Skip if test is marked with @pytest.mark.preserve_data.
    """
    if request.node.get_closest_marker("preserve_data"):
        yield
        return

    from qdrant_client.models import Filter
    vault.client.delete(
        collection_name=vault.collection_name,
        points_selector=Filter()  # Matches all points
    )
    yield


@pytest.fixture
def sample_texts():
    return [
        "The project deadline is March 30, 2026.",
        "The client prefers Python 3.12 for backend services.",
        "Budget for Phase 1 is capped at $50,000.",
        "Use PostgreSQL for the main relational database.",
        "The team meets every Monday at 10 AM EST.",
    ]


# ═══════════════════════════════════════════════════════════════════════════════
#  Collection Management
# ═══════════════════════════════════════════════════════════════════════════════

@requires_qdrant
class TestCollectionManagement:

    def test_collection_exists(self, vault):
        """Collection should exist after initialization."""
        info = vault.get_collection_info()
        assert info["name"] == vault.collection_name
        assert info["status"] in ["Status.GREEN", "green"]

    def test_collection_vector_config(self, vault):
        """Vector size should match embedding dimension (384)."""
        info = vault.get_collection_info()
        assert info["vector_size"] == 384

    def test_health_check(self, vault):
        """Health check should report connected = True."""
        health = vault.health_check()
        assert health["connected"] is True
        assert health["collection_exists"] is True


# ═══════════════════════════════════════════════════════════════════════════════
#  Memory CRUD
# ═══════════════════════════════════════════════════════════════════════════════

@requires_qdrant
class TestMemoryCRUD:

    def test_store_single_memory(self, vault):
        """Storing a memory should return a UUID and increment count."""
        initial = vault.count_memories()
        pid = vault.store_memory(
            text="The client prefers Python 3.12.",
            predictive_value=0.7,
            source="user",
            tags=["python"],
        )
        assert isinstance(pid, str)
        assert len(pid) == 36  # UUID format
        # Wait for indexing
        time.sleep(1)
        assert vault.count_memories() == initial + 1

    def test_retrieve_stored_memory(self, vault):
        """Retrieved memory should match what was stored."""
        pid = vault.store_memory(
            text="Budget is $50,000 for Phase 1.",
            predictive_value=0.6,
            source="agent",
        )
        time.sleep(1)
        mem = vault.get_memory(pid)
        assert mem is not None
        assert mem.point_id == pid
        assert mem.payload.text_content == "Budget is $50,000 for Phase 1."
        assert mem.payload.predictive == 0.6
        assert mem.payload.source == "agent"
        assert mem.payload.success_count == 0
        assert mem.payload.frequency == 0

    def test_retrieve_nonexistent_memory(self, vault):
        """Getting a non-existent ID should return None."""
        fake_id = str(uuid.uuid4())
        result = vault.get_memory(fake_id)
        assert result is None

    def test_batch_store(self, vault):
        """Batch insert should create all memories."""
        initial = vault.count_memories()
        memories = [
            {"text": "Meeting on Tuesday at 2 PM.", "source": "user"},
            {"text": "Use Docker for deployment.", "source": "system"},
            {"text": "The API uses REST architecture.", "source": "agent"},
        ]
        ids = vault.store_memories_batch(memories)
        assert len(ids) == 3
        time.sleep(1)
        assert vault.count_memories() == initial + 3

    def test_delete_single_memory(self, vault):
        """Deleting a memory should remove it from the collection."""
        pid = vault.store_memory(text="Temporary memory to delete.")
        time.sleep(1)
        assert vault.get_memory(pid) is not None
        vault.delete_memory(pid)
        time.sleep(1)
        assert vault.get_memory(pid) is None

    def test_delete_batch(self, vault):
        """Batch deletion should remove multiple memories."""
        ids = vault.store_memories_batch([
            {"text": "Delete me 1"},
            {"text": "Delete me 2"},
        ])
        time.sleep(1)
        vault.delete_memories_batch(ids)
        time.sleep(1)
        for pid in ids:
            assert vault.get_memory(pid) is None


# ═══════════════════════════════════════════════════════════════════════════════
#  Semantic Search
# ═══════════════════════════════════════════════════════════════════════════════

@requires_qdrant
class TestSemanticSearch:

    def test_basic_search(self, vault, sample_texts):
        """Search should return semantically relevant results."""
        # Ensure memories exist
        vault.store_memories_batch([{"text": t} for t in sample_texts])
        time.sleep(2)

        results = vault.semantic_search("What is the project deadline?", top_k=3)
        assert len(results) > 0
        # The deadline memory should be among top results
        texts = [r.payload.text_content for r in results]
        assert any("deadline" in t.lower() or "march" in t.lower() for t in texts)

    def test_search_returns_scores(self, vault):
        """Each result should have a cosine similarity score."""
        results = vault.semantic_search("Python programming", top_k=3)
        for r in results:
            assert r.score is not None
            assert 0.0 <= r.score <= 1.0

    def test_search_with_utility_filter(self, vault):
        """Utility threshold filter should exclude low-utility memories."""
        # Store a high-utility memory
        pid = vault.store_memory(text="High utility test memory.", predictive_value=0.9)
        vault.patch_payload(pid, {"utility": 0.9, "success_count": 9, "failure_count": 0})
        time.sleep(1)

        results = vault.semantic_search(
            "test memory", top_k=10, utility_threshold=0.8
        )
        for r in results:
            assert r.payload.utility >= 0.8

    def test_search_empty_query(self, vault):
        """Empty query should still return results (based on vector)."""
        results = vault.semantic_search("", top_k=3)
        # May return results based on default vector encoding
        assert isinstance(results, list)


# ═══════════════════════════════════════════════════════════════════════════════
#  Two-Stage Search & Rerank (Layer A Pipeline)
# ═══════════════════════════════════════════════════════════════════════════════

@requires_qdrant
class TestSearchAndRerank:

    def test_rerank_returns_top_n(self, vault):
        """search_and_rerank should return at most top_n results."""
        results = vault.search_and_rerank("project schedule", top_n=3)
        assert len(results) <= 3

    def test_rerank_scores_populated(self, vault):
        """Reranked results should have both score and dars_score."""
        results = vault.search_and_rerank("budget allocation", top_n=3)
        for r in results:
            assert r.score is not None
            assert r.dars_score is not None

    def test_rerank_sorted_descending(self, vault):
        """Results should be sorted by combined score, descending."""
        results = vault.search_and_rerank("Python backend", top_n=5)
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].score >= results[i + 1].score


# ═══════════════════════════════════════════════════════════════════════════════
#  Atomic Payload Updates (Layer B Interface)
# ═══════════════════════════════════════════════════════════════════════════════

@requires_qdrant
class TestAtomicUpdates:

    def test_patch_payload(self, vault):
        """Raw payload patch should update specific fields."""
        pid = vault.store_memory(text="Patch test memory.")
        time.sleep(1)
        vault.patch_payload(pid, {"utility": 0.75, "frequency": 10})
        time.sleep(0.5)
        mem = vault.get_memory(pid)
        assert mem.payload.utility == 0.75
        assert mem.payload.frequency == 10

    def test_update_recency(self, vault):
        """update_recency should set timestamp to current time."""
        pid = vault.store_memory(text="Recency update test.")
        time.sleep(1)
        # Force old recency
        vault.patch_payload(pid, {"recency": 1000000.0})
        time.sleep(0.5)
        new_ts = vault.update_recency(pid)
        assert abs(new_ts - time.time()) < 5
        mem = vault.get_memory(pid)
        assert abs(mem.payload.recency - new_ts) < 1

    def test_increment_frequency(self, vault):
        """increment_frequency should add 1 to the access count."""
        pid = vault.store_memory(text="Frequency increment test.")
        time.sleep(1)
        assert vault.get_memory(pid).payload.frequency == 0
        new_freq = vault.increment_frequency(pid)
        assert new_freq == 1
        time.sleep(0.5)
        new_freq = vault.increment_frequency(pid)
        assert new_freq == 2

    def test_update_utility_success(self, vault):
        """Recording a success should increase utility."""
        pid = vault.store_memory(text="Utility success test.")
        time.sleep(1)
        u1 = vault.update_utility(pid, success=True)
        assert abs(u1 - (2 / 3)) < 1e-5  # (1+1)/(1+0+2) ≈ 0.667
        u2 = vault.update_utility(pid, success=True)
        assert abs(u2 - (3 / 4)) < 1e-5  # (2+1)/(2+0+2) = 0.75
        assert u2 > u1

    def test_update_utility_failure(self, vault):
        """Recording a failure should decrease utility (using Laplacian smoothing)."""
        pid = vault.store_memory(text="Utility failure test.")
        time.sleep(1)
        u = vault.update_utility(pid, success=False)
        assert abs(u - (1 / 3)) < 1e-5   # (0+1)/(0+1+2) ≈ 0.333

    def test_update_on_retrieval(self, vault):
        """Combined Layer B update should touch recency, frequency, utility."""
        pid = vault.store_memory(text="Full retrieval update test.")
        time.sleep(1)
        updates = vault.update_on_retrieval(pid, success=True)
        assert "recency" in updates
        assert updates["frequency"] == 1
        assert updates["utility"] > 0.0
        assert updates["success_count"] == 1

    def test_update_nonexistent_raises(self, vault):
        """Updating a non-existent memory should raise ValueError."""
        fake_id = str(uuid.uuid4())
        with pytest.raises(ValueError):
            vault.increment_frequency(fake_id)
        with pytest.raises(ValueError):
            vault.update_utility(fake_id, success=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Triage Scan (Layer C Interface)
# ═══════════════════════════════════════════════════════════════════════════════

@requires_qdrant
@pytest.mark.preserve_data
class TestTriageScan:

    @pytest.fixture(autouse=True, scope="class")
    def setup_triage_data(self, vault):
        """Populate initial data once for all triage tests."""
        memories = [
            {"text": "The project deadline is March 30, 2026.", "source": "user"},
            {"text": "The client prefers Python 3.12 for backend services.", "source": "user"},
            {"text": "Budget for Phase 1 is capped at $50,000.", "source": "agent"},
            {"text": "Use PostgreSQL for the main relational database.", "source": "user"},
            {"text": "The team meets every Monday at 10 AM EST.", "source": "system"}
        ]
        vault.store_memories_batch(memories)

    @pytest.mark.asyncio
    async def test_triage_all(self, vault):
        """triage_all_memories should return decisions for all memories."""
        import asyncio
        await asyncio.sleep(0.5)
        decisions = vault.triage_all_memories(limit=100)
        assert len(decisions) > 0
        for d in decisions:
            assert d.action in ("retain", "compress", "delete")
            assert 0.0 <= d.dars_score <= 1.0
            assert isinstance(d.point_id, str)

    def test_triage_sorted_ascending(self, vault):
        """Decisions should be sorted by DARS score ascending (worst first)."""
        decisions = vault.triage_all_memories(limit=100)
        if len(decisions) > 1:
            for i in range(len(decisions) - 1):
                assert decisions[i].dars_score <= decisions[i + 1].dars_score

    def test_get_all_memories(self, vault):
        """get_all_memories should return MemoryPoint objects."""
        memories = vault.get_all_memories(limit=10)
        assert len(memories) > 0
        for m in memories:
            assert isinstance(m, MemoryPoint)
            assert isinstance(m.payload, MemoryPayload)
            assert len(m.payload.text_content) > 0
