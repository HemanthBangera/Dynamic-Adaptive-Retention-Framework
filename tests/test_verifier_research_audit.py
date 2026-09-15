import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from config.settings import DARSConfig
from core.layer_a.gateway import CognitiveGateway
from core.layer_a.reformulator import QueryReformulator
from core.layer_b.engine import LearningEngine
from core.layer_c.janitor import DecisionEngine
from core.layer_c.triage import TriageOrchestrator
from core.layer_d.storage import MemoryVault


# -----------------------------------------------------------------------------
# Verifier audit: critical layer contracts
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reformulator_empty_generation_falls_back_to_raw_query():
    """Layer A should fail-open when Gemini returns an empty expansion string."""
    raw_query = "Pista budget"

    with patch("config.settings.DARSConfig.GEMINI_API_KEY", "dummy_key"):
        reformulator = QueryReformulator()

    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "   "}]}}]
        }
        mock_post.return_value.__aenter__.return_value = mock_resp

        result = await reformulator.reformulate_query(raw_query)

    assert result == raw_query


@pytest.mark.asyncio
async def test_ingest_new_facts_predictive_value_is_bounded():
    """Layer B → D: a new fact's persisted predictive value stays in [0, 1],
    even when the goal vector is exactly anti-aligned with the fact (cosine = −1)."""
    vault = MemoryVault(collection_name="verifier_p_bound", location=":memory:")
    vault.initialize_collection(recreate=True)
    fact = "negative alignment fact"
    anti_goal = [-x for x in vault.embedder.encode(fact)]

    engine = LearningEngine(evaluator=Mock(), vault=vault, embedder=Mock())
    with patch.object(DARSConfig, "get_goal_vector", return_value=anti_goal):
        await engine.ingest_new_facts([fact])

    stored = vault.get_all_memories(limit=10)
    assert len(stored) == 1
    assert 0.0 <= stored[0].payload.predictive <= 1.0


@pytest.mark.asyncio
async def test_triage_orchestrator_surfaces_maintenance_failures():
    """Layer C scheduler should surface triage failures to its caller for observability."""
    vault = Mock()
    vault.get_all_memories = Mock(side_effect=RuntimeError("scroll failed"))

    orchestrator = TriageOrchestrator(vault=vault, janitor=Mock())

    with pytest.raises(RuntimeError):
        await orchestrator.run_maintenance()


# -----------------------------------------------------------------------------
# Verifier audit: reference checks that should remain stable
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gateway_handoff_uses_expanded_query_for_search_but_raw_for_prompt():
    reformulator = Mock()
    reformulator.reformulate_query = AsyncMock(return_value="expanded query")

    reranker = Mock()
    reranker.rerank = Mock(return_value=[])

    gateway = CognitiveGateway(reformulator=reformulator, reranker=reranker, alpha=0.6)

    prompt = await gateway.process_query("raw user question")

    assert reranker.rerank.call_count == 1
    assert reranker.rerank.call_args.kwargs["query"] == "expanded query"
    assert "raw user question" in prompt


@pytest.mark.asyncio
async def test_decision_engine_skips_fresh_memories_during_grace_period():
    vault = Mock()
    vault.compute_dars_score = Mock(return_value=0.1)

    compressor = Mock()
    compressor.compress_memory = AsyncMock(return_value=True)

    janitor = DecisionEngine(vault=vault, compressor=compressor)

    payload = SimpleNamespace(
        created_at=time.time(),
        recency=time.time(),
        is_compressed=False,
        text_content="fresh memory",
        to_dict=lambda: {},
    )
    memory_point = SimpleNamespace(point_id="m-fresh", payload=payload)

    await janitor.triage_memory(memory_point)

    assert vault.compute_dars_score.call_count == 0
    assert vault.patch_payload.call_count == 0
    assert vault.delete_memory.call_count == 0
    assert compressor.compress_memory.call_count == 0


@pytest.mark.asyncio
async def test_feedback_loop_patches_all_retrieved_memories():
    evaluator = Mock()
    evaluator.evaluate_success = AsyncMock(return_value="YES")

    vault = Mock()
    engine = LearningEngine(evaluator=evaluator, vault=vault, embedder=Mock())

    memories = [
        {
            "id": "m-1",
            "payload": {
                "text_content": "first",
                "success_count": 0,
                "failure_count": 0,
                "frequency": 1,
            },
        },
        {
            "id": "m-2",
            "payload": {
                "text_content": "second",
                "success_count": 1,
                "failure_count": 0,
                "frequency": 2,
            },
        },
    ]

    await engine.process_feedback_loop("q", "r", memories)

    # Every retrieved memory receives all three version-guarded Layer D updates.
    for method in ("update_utility", "increment_frequency", "update_recency"):
        called_ids = [c.args[0] for c in getattr(vault, method).call_args_list]
        assert called_ids == ["m-1", "m-2"], method
    assert all(c.args[1] is True for c in vault.update_utility.call_args_list)
    assert vault.patch_payload.call_count == 0  # no unguarded writes on the feedback path
