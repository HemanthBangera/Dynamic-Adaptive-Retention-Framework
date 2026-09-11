"""
Layer D revision tests (local Qdrant, no LLM)
=============================================
Backends, version-guarded optimistic concurrency, rank modes and score
components, per-vault weights, the virtual clock, and best-effort triage.
"""

import math
from unittest.mock import AsyncMock, Mock

import pytest
from qdrant_client.models import PointStruct

from config.settings import DARSConfig
from core.layer_c.janitor import DecisionEngine
from core.layer_c.triage import TriageOrchestrator
from core.layer_d.schema import DARSWeights
from core.layer_d.storage import RANK_MODES, MemoryVault


def _fresh_vault(name: str, **kwargs) -> MemoryVault:
    v = MemoryVault(collection_name=name, location=":memory:", **kwargs)
    v.initialize_collection(recreate=True)
    return v


# ═══════════════════════════════════════════════════════════════════════════════
#  Backends
# ═══════════════════════════════════════════════════════════════════════════════

class TestBackends:

    def test_local_memory_backend(self):
        v = _fresh_vault("rev_backend")
        assert v.backend == "local-memory"
        assert v.health_check()["connected"] is True

    def test_missing_store_configuration_raises(self):
        class NoStore(DARSConfig):
            QDRANT_URL = ""
            QDRANT_LOCATION = ""

        with pytest.raises(ValueError, match="No vector store configured"):
            MemoryVault(config=NoStore())

    def test_invalid_weights_raise(self):
        with pytest.raises(ValueError, match="must sum to 1.0"):
            MemoryVault(location=":memory:", weights=DARSWeights(w_r=0.5, w_f=0.5, w_u=0.5, w_p=0.5))


# ═══════════════════════════════════════════════════════════════════════════════
#  Optimistic concurrency
# ═══════════════════════════════════════════════════════════════════════════════

class TestOptimisticConcurrency:

    def test_new_memory_starts_at_version_zero(self, vault):
        pid = vault.store_memory("Versioned memory")
        assert vault._read_raw_payload(pid)["version"] == 0

    def test_each_guarded_update_bumps_version(self, vault):
        pid = vault.store_memory("Bump test")
        vault.update_utility(pid, success=True)
        vault.increment_frequency(pid)
        raw = vault._read_raw_payload(pid)
        assert raw["version"] == 2
        assert raw["frequency"] == 1 and raw["success_count"] == 1

    def test_stale_write_is_detected_and_not_applied(self, vault):
        """A writer holding an outdated version must raise, not silently overwrite."""
        pid = vault.store_memory("Race test")
        stale = vault._read_raw_payload(pid)          # writer A reads version 0
        vault.increment_frequency(pid)                # writer B commits version 1
        with pytest.raises(RuntimeError, match="Optimistic lock conflict"):
            vault._conditional_patch(pid, stale, {"frequency": 99}, "stale write")
        raw = vault._read_raw_payload(pid)
        assert raw["frequency"] == 1                  # B's update survives
        assert raw["version"] == 1

    def test_legacy_point_without_version_field(self, vault):
        """Points created before versioning are guarded on the field being absent."""
        pid = "11111111-1111-1111-1111-111111111111"
        vault.client.upsert(
            collection_name=vault.collection_name,
            points=[PointStruct(
                id=pid,
                vector=vault.embedder.encode("legacy"),
                payload={"text_content": "legacy", "frequency": 4},
            )],
        )
        assert vault.increment_frequency(pid) == 5
        assert vault._read_raw_payload(pid)["version"] == 1

    def test_update_on_retrieval_uses_virtual_time(self, vault):
        pid = vault.store_memory("Virtual retrieval", sim_timestamp=1_000.0)
        updates = vault.update_on_retrieval(pid, success=False, current_time=5_000.0)
        raw = vault._read_raw_payload(pid)
        assert updates["recency"] == 5_000.0 and raw["recency"] == 5_000.0
        assert raw["failure_count"] == 1 and raw["frequency"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
#  Rank modes and score components
# ═══════════════════════════════════════════════════════════════════════════════

class TestRankModes:

    TEXTS = [
        "Python is a programming language for machine learning",
        "Machine learning models are trained on data",
        "The recipe needs two cups of flour",
        "Deep learning is a subfield of machine learning",
        "The train to Mysore leaves at nine",
    ]

    def _seed(self, vault):
        for i, t in enumerate(self.TEXTS):
            vault.store_memory(t, predictive_value=0.1 * i)

    def test_similarity_mode_matches_semantic_search(self, vault):
        self._seed(vault)
        base = [m.point_id for m in vault.semantic_search("machine learning", top_k=5)]
        ranked = vault.search_and_rerank("machine learning", fetch_k=5, top_n=5, rank_mode="similarity")
        assert [m.point_id for m in ranked] == base

    def test_wrrf_without_dars_vote_matches_similarity(self, vault):
        self._seed(vault)
        sim = vault.search_and_rerank("machine learning", fetch_k=5, top_n=5, rank_mode="similarity")
        wrrf = vault.search_and_rerank(
            "machine learning", fetch_k=5, top_n=5, rank_mode="wrrf", beta_sim=1.0, beta_dars=0.0
        )
        assert [m.point_id for m in wrrf] == [m.point_id for m in sim]

    def test_legacy_use_rrf_equals_rrf_mode(self, vault):
        self._seed(vault)
        now = 2_000_000_000.0
        legacy = vault.search_and_rerank("machine learning", fetch_k=5, top_n=3, current_time=now)
        explicit = vault.search_and_rerank(
            "machine learning", fetch_k=5, top_n=3, current_time=now, rank_mode="rrf"
        )
        assert [m.point_id for m in legacy] == [m.point_id for m in explicit]

    def test_components_are_attached_and_consistent(self, vault):
        self._seed(vault)
        out = vault.search_and_rerank(
            "machine learning", fetch_k=5, top_n=5, rank_mode="rrf", return_components=True
        )
        for m in out:
            c = m.components
            assert set(c) >= {"R", "F", "U", "P", "S", "sim", "sim_rank", "dars_rank"}
            assert c["S"] == m.dars_score == vault.score_from_components(c)

    def test_unknown_rank_mode_raises(self, vault):
        with pytest.raises(ValueError, match="Unknown rank_mode"):
            vault.search_and_rerank("anything", rank_mode="magic")
        assert set(RANK_MODES) == {"similarity", "rrf", "wrrf", "blend"}

    def test_query_vector_reuse(self, vault):
        self._seed(vault)
        qv = vault.embedder.encode("machine learning")
        a = vault.semantic_search("machine learning", top_k=3)
        b = vault.semantic_search("ignored text", top_k=3, query_vector=qv)
        assert [m.point_id for m in a] == [m.point_id for m in b]


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-vault weights and the virtual clock
# ═══════════════════════════════════════════════════════════════════════════════

class TestWeightsAndClock:

    def test_recency_only_weights_score_equals_R(self):
        v = _fresh_vault("rev_weights", weights=DARSWeights(w_r=1.0, w_f=0.0, w_u=0.0, w_p=0.0))
        pid = v.store_memory("Clock test", sim_timestamp=0.0, predictive_value=0.9)
        payload = v.get_memory(pid).payload.to_dict()
        now = 100 * 3600.0
        expected = round(math.exp(-DARSConfig.RECENCY_DECAY_LAMBDA * 100), 6)
        assert v.compute_dars_score(payload, current_time=now) == expected

    def test_batch_store_timestamps_vectors_and_extra(self, vault):
        vec = vault.embedder.encode("precomputed")
        pids = vault.store_memories_batch(
            [
                {"text": "a", "timestamp": 10.0, "extra": {"unit_index": 0}},
                {"text": "precomputed", "vector": vec, "timestamp": 20.0},
                {"text": "c"},
            ],
            current_time=30.0,
        )
        raws = [vault._read_raw_payload(p) for p in pids]
        assert [r["recency"] for r in raws] == [10.0, 20.0, 30.0]
        assert raws[0]["unit_index"] == 0
        assert all(r["version"] == 0 for r in raws)

    @pytest.mark.asyncio
    async def test_grace_period_follows_virtual_clock(self, vault):
        pid = vault.store_memory("Grace clock", sim_timestamp=0.0)
        vault.patch_payload(pid, {"failure_count": 10, "predictive": 0.0})
        janitor = DecisionEngine(vault=vault, compressor=Mock())

        await janitor.triage_memory(vault.get_memory(pid), current_time=3_600.0)   # 1 h: inside grace
        assert vault.get_memory(pid) is not None

        await janitor.triage_memory(vault.get_memory(pid), current_time=400 * 3600.0)  # 400 h: stale
        assert vault.get_memory(pid) is None

    @pytest.mark.asyncio
    async def test_configurable_grace_period(self, vault):
        pid = vault.store_memory("Short grace", sim_timestamp=0.0)
        vault.patch_payload(pid, {"failure_count": 10, "predictive": 0.0})
        janitor = DecisionEngine(vault=vault, compressor=Mock(), grace_period_s=0.0)
        await janitor.triage_memory(vault.get_memory(pid), current_time=400 * 3600.0)
        assert vault.get_memory(pid) is None


# ═══════════════════════════════════════════════════════════════════════════════
#  Best-effort maintenance
# ═══════════════════════════════════════════════════════════════════════════════

class TestBestEffortTriage:

    @pytest.mark.asyncio
    async def test_one_failure_does_not_stop_the_cycle(self, vault):
        pids = [vault.store_memory(f"Triage point {i}") for i in range(5)]
        seen = []

        async def triage(point, current_time=None):
            seen.append(point.point_id)
            if point.point_id == pids[2]:
                raise RuntimeError("boom")

        janitor = Mock()
        janitor.triage_memory = AsyncMock(side_effect=triage)
        orch = TriageOrchestrator(vault=vault, janitor=janitor)

        with pytest.raises(RuntimeError, match="1 failed point"):
            await orch.run_maintenance()
        assert sorted(seen) == sorted(pids)   # every point was attempted
