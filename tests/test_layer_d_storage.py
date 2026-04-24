"""
DARS Layer D — Real Qdrant Integration Tests
=============================================
Every test hits the live Qdrant Cloud instance.  Zero mocks.
"""

import math
import time
import pytest
from core.layer_d.storage import MemoryVault
from core.layer_d.schema import MemoryPayload, MemoryPoint, DARSWeights, RetentionDecision


# ═══════════════════════════════════════════════════════════════════════════════
#  1.  Collection Management
# ═══════════════════════════════════════════════════════════════════════════════

class TestCollectionManagement:

    def test_collection_exists_after_init(self, vault):
        assert vault.client.collection_exists(vault.collection_name)

    def test_collection_vector_dimension(self, vault):
        info = vault.get_collection_info()
        assert info["vector_size"] == 384

    def test_collection_distance_metric(self, vault):
        info = vault.get_collection_info()
        assert "Cosine" in info["distance"]

    def test_health_check_connected(self, vault):
        health = vault.health_check()
        assert health["connected"] is True
        assert health["collection_exists"] is True

    def test_recreate_collection_returns_true(self, vault):
        result = vault.initialize_collection(recreate=True)
        assert result is True

    def test_idempotent_init_returns_false(self, vault):
        vault.initialize_collection(recreate=True)
        result = vault.initialize_collection(recreate=False)
        assert result is False


# ═══════════════════════════════════════════════════════════════════════════════
#  2.  Memory CRUD
# ═══════════════════════════════════════════════════════════════════════════════

class TestMemoryCRUD:

    def test_store_returns_uuid(self, vault):
        pid = vault.store_memory("Test memory")
        assert isinstance(pid, str) and len(pid) == 36

    def test_retrieve_stored_memory(self, vault):
        pid = vault.store_memory("The client prefers Python 3.12")
        mem = vault.get_memory(pid)
        assert mem is not None
        assert mem.payload.text_content == "The client prefers Python 3.12"

    def test_retrieve_nonexistent_returns_none(self, vault):
        assert vault.get_memory("00000000-0000-0000-0000-000000000000") is None

    def test_stored_payload_has_correct_defaults(self, vault):
        pid = vault.store_memory("Default check")
        mem = vault.get_memory(pid)
        p = mem.payload
        assert p.success_count == 0
        assert p.failure_count == 0
        assert p.utility == 0.0
        assert p.frequency == 0
        assert p.is_compressed is False

    def test_store_with_explicit_predictive_value(self, vault):
        pid = vault.store_memory("Explicit p", predictive_value=0.9)
        mem = vault.get_memory(pid)
        assert abs(mem.payload.predictive - 0.9) < 0.01

    def test_store_clamps_predictive_above_one(self, vault):
        pid = vault.store_memory("Clamp high", predictive_value=5.0)
        mem = vault.get_memory(pid)
        assert mem.payload.predictive <= 1.0

    def test_store_clamps_predictive_below_zero(self, vault):
        pid = vault.store_memory("Clamp low", predictive_value=-2.0)
        mem = vault.get_memory(pid)
        assert mem.payload.predictive >= 0.0

    def test_batch_store(self, vault):
        mems = [{"text": f"Batch item {i}"} for i in range(5)]
        pids = vault.store_memories_batch(mems)
        assert len(pids) == 5
        for pid in pids:
            assert vault.get_memory(pid) is not None

    def test_delete_memory(self, vault):
        pid = vault.store_memory("Delete me")
        vault.delete_memory(pid)
        assert vault.get_memory(pid) is None

    def test_batch_delete(self, vault):
        pids = [vault.store_memory(f"Batch del {i}") for i in range(3)]
        vault.delete_memories_batch(pids)
        for pid in pids:
            assert vault.get_memory(pid) is None

    def test_store_with_tags(self, vault):
        pid = vault.store_memory("Tagged memory", tags=["important", "test"])
        mem = vault.get_memory(pid)
        assert "important" in mem.payload.tags
        assert "test" in mem.payload.tags

    def test_store_with_source(self, vault):
        pid = vault.store_memory("Source memory", source="user")
        mem = vault.get_memory(pid)
        assert mem.payload.source == "user"


# ═══════════════════════════════════════════════════════════════════════════════
#  3.  Semantic Search
# ═══════════════════════════════════════════════════════════════════════════════

class TestSemanticSearch:

    def test_basic_search_returns_results(self, vault):
        vault.store_memory("Python is a programming language used for AI")
        vault.store_memory("JavaScript is used for web development")
        time.sleep(1)
        results = vault.semantic_search("programming language", top_k=5)
        assert len(results) >= 1

    def test_search_relevance_order(self, vault):
        vault.store_memory("Python is a great language for machine learning")
        vault.store_memory("The weather today is sunny and warm")
        time.sleep(1)
        results = vault.semantic_search("machine learning language", top_k=5)
        assert results[0].payload.text_content.startswith("Python")

    def test_search_returns_scores(self, vault):
        vault.store_memory("Test score return")
        time.sleep(1)
        results = vault.semantic_search("Test score", top_k=1)
        assert results[0].score is not None and results[0].score > 0

    def test_search_with_utility_filter(self, vault):
        pid_high = vault.store_memory("High utility memory")
        vault.patch_payload(pid_high, {"utility": 0.95})
        pid_low = vault.store_memory("Low utility memory")
        vault.patch_payload(pid_low, {"utility": 0.05})
        time.sleep(1)
        results = vault.semantic_search("utility memory", top_k=10, utility_threshold=0.5)
        pids_found = [r.point_id for r in results]
        assert pid_high in pids_found
        assert pid_low not in pids_found

    def test_empty_collection_search(self, vault):
        results = vault.semantic_search("anything", top_k=5)
        assert results == []


# ═══════════════════════════════════════════════════════════════════════════════
#  4.  Search and Rerank
# ═══════════════════════════════════════════════════════════════════════════════

class TestSearchAndRerank:

    def test_rerank_returns_top_n(self, vault):
        for i in range(5):
            vault.store_memory(f"Memory about software engineering topic {i}")
        time.sleep(1)
        results = vault.search_and_rerank("software engineering", top_n=3)
        assert len(results) <= 3

    def test_rerank_scores_populated(self, vault):
        vault.store_memory("Reranking score test")
        time.sleep(1)
        results = vault.search_and_rerank("Reranking score", top_n=1)
        if results:
            assert results[0].score is not None
            assert results[0].dars_score is not None

    def test_rerank_sorted_descending(self, vault):
        for i in range(5):
            vault.store_memory(f"Memory about data science topic {i}")
        time.sleep(1)
        results = vault.search_and_rerank("data science", top_n=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  5.  Atomic Payload Updates
# ═══════════════════════════════════════════════════════════════════════════════

class TestAtomicUpdates:

    def test_patch_payload(self, vault):
        pid = vault.store_memory("Patch test")
        vault.patch_payload(pid, {"utility": 0.88})
        mem = vault.get_memory(pid)
        assert abs(mem.payload.utility - 0.88) < 0.01

    def test_update_recency(self, vault):
        pid = vault.store_memory("Recency test")
        old_mem = vault.get_memory(pid)
        time.sleep(0.5)
        new_ts = vault.update_recency(pid)
        assert new_ts > old_mem.payload.recency

    def test_increment_frequency(self, vault):
        pid = vault.store_memory("Freq test")
        new_freq = vault.increment_frequency(pid)
        assert new_freq == 1
        new_freq = vault.increment_frequency(pid)
        assert new_freq == 2

    def test_update_utility_success(self, vault):
        pid = vault.store_memory("Util success test")
        new_util = vault.update_utility(pid, success=True)
        mem = vault.get_memory(pid)
        assert mem.payload.success_count == 1
        assert mem.payload.failure_count == 0
        expected = (0 + 1 + 1) / (0 + 1 + 0 + 2)
        assert abs(new_util - expected) < 0.01

    def test_update_utility_failure(self, vault):
        pid = vault.store_memory("Util failure test")
        new_util = vault.update_utility(pid, success=False)
        mem = vault.get_memory(pid)
        assert mem.payload.success_count == 0
        assert mem.payload.failure_count == 1

    def test_update_on_retrieval(self, vault):
        pid = vault.store_memory("Retrieval update test")
        updates = vault.update_on_retrieval(pid, success=True)
        assert updates["frequency"] == 1
        assert updates["success_count"] == 1

    def test_update_nonexistent_raises(self, vault):
        with pytest.raises(ValueError, match="Memory not found"):
            vault.update_utility("00000000-0000-0000-0000-000000000000", success=True)

    def test_increment_nonexistent_raises(self, vault):
        with pytest.raises(ValueError, match="Memory not found"):
            vault.increment_frequency("00000000-0000-0000-0000-000000000000")


# ═══════════════════════════════════════════════════════════════════════════════
#  6.  DARS Score Computation
# ═══════════════════════════════════════════════════════════════════════════════

class TestDARSScoring:

    def test_fresh_memory_score(self, vault):
        pid = vault.store_memory("Fresh memory", predictive_value=0.5)
        mem = vault.get_memory(pid)
        score = vault.compute_dars_score(mem.payload.to_dict())
        assert 0.3 < score < 0.8

    def test_perfect_memory_high_score(self, vault):
        pid = vault.store_memory("Perfect memory", predictive_value=1.0)
        vault.patch_payload(pid, {
            "success_count": 20, "failure_count": 0,
            "utility": 0.95, "frequency": 30,
            "recency": time.time(), "predictive": 1.0,
        })
        mem = vault.get_memory(pid)
        score = vault.compute_dars_score(mem.payload.to_dict())
        assert score > 0.8

    def test_stale_memory_low_score(self, vault):
        pid = vault.store_memory("Stale memory")
        vault.patch_payload(pid, {
            "recency": time.time() - 30 * 24 * 3600,
            "frequency": 0, "success_count": 0, "failure_count": 5,
            "utility": 0.1, "predictive": 0.0,
        })
        mem = vault.get_memory(pid)
        score = vault.compute_dars_score(mem.payload.to_dict())
        assert score < 0.3

    def test_score_clamped_0_to_1(self, vault):
        pid = vault.store_memory("Clamp test")
        mem = vault.get_memory(pid)
        score = vault.compute_dars_score(mem.payload.to_dict())
        assert 0.0 <= score <= 1.0

    def test_recency_decay_slower_after_fix(self, vault):
        """After BUG #3 fix (λ=0.005), R should be ~0.76 after 55 hours, not 0.25."""
        now = time.time()
        r_55h = vault._compute_recency(now - 55 * 3600, now)
        assert r_55h > 0.70, f"Expected R > 0.70 after 55h with λ=0.005, got {r_55h:.3f}"

    def test_frequency_log_scaling(self, vault):
        f0 = vault._compute_frequency(0)
        f1 = vault._compute_frequency(1)
        f50 = vault._compute_frequency(50)
        assert f0 == 0.0
        assert 0.0 < f1 < f50
        assert abs(f50 - 1.0) < 0.001

    def test_utility_laplacian_smoothing(self, vault):
        u_zero = vault._compute_utility_score(0, 0)
        u_perfect = vault._compute_utility_score(10, 0)
        u_terrible = vault._compute_utility_score(0, 10)
        assert abs(u_zero - 0.5) < 0.01
        assert u_perfect > 0.8
        assert u_terrible < 0.2


# ═══════════════════════════════════════════════════════════════════════════════
#  7.  Retention Classification
# ═══════════════════════════════════════════════════════════════════════════════

class TestRetentionClassification:

    def test_retain_above_07(self, vault):
        assert vault.classify_memory(0.85) == "retain"

    def test_compress_between_03_and_07(self, vault):
        assert vault.classify_memory(0.5) == "compress"

    def test_delete_below_03(self, vault):
        assert vault.classify_memory(0.1) == "delete"

    def test_boundary_07_is_compress(self, vault):
        assert vault.classify_memory(0.7) == "compress"

    def test_boundary_03_is_delete(self, vault):
        assert vault.classify_memory(0.3) == "delete"


# ═══════════════════════════════════════════════════════════════════════════════
#  8.  Triage Scan
# ═══════════════════════════════════════════════════════════════════════════════

class TestTriageScan:

    def test_triage_all_returns_decisions(self, vault):
        vault.store_memory("Triage test memory A")
        vault.store_memory("Triage test memory B")
        time.sleep(1)
        decisions = vault.triage_all_memories()
        assert len(decisions) >= 2
        assert all(isinstance(d, RetentionDecision) for d in decisions)

    def test_triage_sorted_ascending(self, vault):
        for i in range(3):
            vault.store_memory(f"Triage sort test {i}")
        time.sleep(1)
        decisions = vault.triage_all_memories()
        scores = [d.dars_score for d in decisions]
        assert scores == sorted(scores)


# ═══════════════════════════════════════════════════════════════════════════════
#  9.  Schema Integrity
# ═══════════════════════════════════════════════════════════════════════════════

class TestSchemaIntegrity:

    def test_payload_to_dict_round_trip(self):
        p = MemoryPayload(text_content="Round trip", predictive=0.7)
        d = p.to_dict()
        p2 = MemoryPayload.from_dict(d)
        assert p2.text_content == "Round trip"
        assert abs(p2.predictive - 0.7) < 0.01

    def test_from_dict_ignores_extra_keys(self):
        d = {"text_content": "Extra keys", "unknown_field": 42}
        p = MemoryPayload.from_dict(d)
        assert p.text_content == "Extra keys"

    def test_compute_utility_laplacian(self):
        p = MemoryPayload(text_content="Test", success_count=3, failure_count=1)
        u = p.compute_utility()
        expected = (3 + 1) / (3 + 1 + 2)
        assert abs(u - expected) < 0.001

    def test_dars_weights_validate(self):
        w = DARSWeights()
        assert w.validate() is True

    def test_dars_weights_invalid(self):
        w = DARSWeights(w_r=0.5, w_f=0.5, w_u=0.5, w_p=0.5)
        assert w.validate() is False

    def test_memory_point_generate_id_unique(self):
        ids = [MemoryPoint.generate_id() for _ in range(100)]
        assert len(set(ids)) == 100
