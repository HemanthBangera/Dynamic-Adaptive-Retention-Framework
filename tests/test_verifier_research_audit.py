import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from core.layer_a.gateway import CognitiveGateway
from core.layer_a.reformulator import QueryReformulator
from core.layer_b.engine import LearningEngine
from core.layer_c.janitor import DecisionEngine
from core.layer_c.triage import TriageOrchestrator


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
    """Layer B should keep predictive value in [0, 1] before persisting to Layer D."""
    evaluator = Mock()
    evaluator.evaluate_success = AsyncMock(return_value="YES")

    embedder = Mock()
    embedder.encode = Mock(return_value=[[-1.0, 0.0]])

    vault = Mock()
    vault.store_memory = Mock(return_value="pid-1")

    engine = LearningEngine(evaluator=evaluator, vault=vault, embedder=embedder)

    with patch("config.settings.DARSConfig.GOAL_VECTOR", [1.0, 0.0]):
        await engine.ingest_new_facts(["negative alignment fact"])

    _, kwargs = vault.store_memory.call_args
    assert 0.0 <= kwargs["predictive_value"] <= 1.0


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

    assert vault.patch_payload.call_count == 2
    for call in vault.patch_payload.call_args_list:
        updates = call.args[1]
        assert "frequency" in updates
        assert "recency" in updates
