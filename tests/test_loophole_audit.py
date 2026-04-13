import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from core.layer_a.prompt_constructor import PromptConstructor
from core.layer_a.reformulator import QueryReformulator
from core.layer_b.calculator import ScoreCalculator
from core.layer_b.engine import LearningEngine
from core.layer_b.evaluator import SuccessEvaluator
from core.layer_d.schema import MemoryPayload, MemoryPoint
from core.layer_d.storage import MemoryVault


# -----------------------------------------------------------------------------
# Layer B <-> Layer D contract checks
# -----------------------------------------------------------------------------

def test_score_calculator_emits_schema_compatible_keys():
    """Layer B updates must align with Layer D payload fields: frequency + recency."""
    updates = ScoreCalculator.calculate_updates(
        success=True,
        current_success_count=1,
        current_failure_count=1,
        current_access_count=2,
    )

    assert "frequency" in updates, "Expected 'frequency' key for Layer D schema compatibility"
    assert "recency" in updates, "Expected 'recency' key for Layer D schema compatibility"
    assert "access_count" not in updates
    assert "last_accessed" not in updates


@pytest.mark.asyncio
async def test_learning_engine_feedback_loop_patches_schema_fields():
    """Feedback loop should patch Layer D using schema-native keys (frequency, recency)."""
    evaluator = Mock()
    evaluator.evaluate_success = AsyncMock(return_value="YES")

    vault = Mock()
    engine = LearningEngine(evaluator=evaluator, vault=vault, embedder=Mock())

    memories = [
        {
            "id": "m-1",
            "payload": {
                "text_content": "budget constraints",
                "success_count": 0,
                "failure_count": 0,
                "frequency": 4,
            },
        }
    ]

    await engine.process_feedback_loop("query", "response", memories)

    assert vault.patch_payload.call_count == 1
    call_args = vault.patch_payload.call_args.args
    assert call_args[0] == "m-1"
    updates = call_args[1]
    assert "frequency" in updates
    assert "recency" in updates
    assert "access_count" not in updates
    assert "last_accessed" not in updates


@pytest.mark.asyncio
async def test_learning_engine_ingest_new_facts_forwards_predictive_value_to_storage():
    """New-fact ingestion should compute P and pass it into store_memory()."""
    evaluator = Mock()
    evaluator.evaluate_success = AsyncMock(return_value="YES")

    embedder = Mock()
    embedder.encode = Mock(return_value=[0.2, 0.1, 0.3, 0.4])

    vault = Mock()
    vault.store_memory = Mock(return_value="pid-1")

    engine = LearningEngine(evaluator=evaluator, vault=vault, embedder=embedder)

    with patch("config.settings.DARSConfig.GOAL_VECTOR", [0.1, 0.1, 0.1, 0.1]):
        await engine.ingest_new_facts(["I like pista ice cream"])

    assert vault.store_memory.call_count == 1
    _, kwargs = vault.store_memory.call_args
    assert "predictive_value" in kwargs


# -----------------------------------------------------------------------------
# Fail-open safety checks
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_success_evaluator_non_binary_returns_neutral():
    with patch("config.settings.DARSConfig.GEMINI_API_KEY", "real_key"):
        evaluator = SuccessEvaluator()

    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "MAYBE"}]}}]
        }
        mock_post.return_value.__aenter__.return_value = mock_resp

        verdict = await evaluator.evaluate_success("q", "r", "m")
        assert verdict == "NEUTRAL"


@pytest.mark.asyncio
async def test_reformulator_timeout_returns_raw_query():
    raw_query = "What is the phase one budget allocation details?"

    with patch("config.settings.DARSConfig.GEMINI_API_KEY", "dummy_key"):
        reformulator = QueryReformulator(timeout=0.01)

    with patch("aiohttp.ClientSession.post", side_effect=asyncio.TimeoutError):
        result = await reformulator.reformulate_query(raw_query)

    assert result == raw_query


@pytest.mark.asyncio
async def test_reformulator_long_query_length_guard_fallback():
    raw_query = "Please provide the final budget allocation and timeline constraints"

    with patch("config.settings.DARSConfig.GEMINI_API_KEY", "dummy_key"):
        reformulator = QueryReformulator()

    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": raw_query + " plus extra verbose expansion that exceeds guard threshold significantly"}]}}]
        }
        mock_post.return_value.__aenter__.return_value = mock_resp

        result = await reformulator.reformulate_query(raw_query)

    assert result == raw_query


# -----------------------------------------------------------------------------
# Prompt / safety checks
# -----------------------------------------------------------------------------

def test_prompt_constructor_escapes_xml_special_characters():
    """Prompt XML must escape unsafe content from memories/query."""
    memories = [
        MemoryPoint(
            point_id="x-1",
            vector=[],
            payload=MemoryPayload(text_content='Danger <tag> & "quoted" value'),
            score=0.9,
        )
    ]
    query = "Can you parse <xml> safely & correctly?"

    prompt = PromptConstructor.build(query, memories)

    # Robust XML safety expectation:
    assert "<tag>" not in prompt
    assert "& \"quoted\"" not in prompt
    assert "&lt;tag&gt;" in prompt


# -----------------------------------------------------------------------------
# Optimistic-locking checks (storage)
# -----------------------------------------------------------------------------

def _fake_vault_for_atomic_ops() -> MemoryVault:
    vault = MemoryVault.__new__(MemoryVault)
    vault.collection_name = "dummy"
    vault.client = Mock()
    return vault


def test_increment_frequency_detects_conflict_when_update_not_applied():
    """If optimistic update applies to zero rows, operation should fail loudly."""
    vault = _fake_vault_for_atomic_ops()
    payload = MemoryPayload(text_content="x", frequency=2)
    vault.get_memory = Mock(return_value=MemoryPoint(point_id="id-1", vector=[], payload=payload, score=0.5))

    # Simulate storage backend reporting no-op / zero updated records
    vault.client.set_payload = Mock(return_value={"updated": 0})

    with pytest.raises(RuntimeError):
        vault.increment_frequency("id-1")


def test_update_utility_detects_conflict_when_update_not_applied():
    """Utility updates should detect and report optimistic-lock misses."""
    vault = _fake_vault_for_atomic_ops()
    payload = MemoryPayload(text_content="x", success_count=0, failure_count=0)
    vault.get_memory = Mock(return_value=MemoryPoint(point_id="id-1", vector=[], payload=payload, score=0.5))

    vault.client.set_payload = Mock(return_value={"updated": 0})

    with pytest.raises(RuntimeError):
        vault.update_utility("id-1", success=True)


# -----------------------------------------------------------------------------
# Docs/runtime consistency checks
# -----------------------------------------------------------------------------

def test_docs_and_settings_have_single_epsilon_policy():
    """Documentation should not conflict with runtime epsilon setting."""
    from config.settings import DARSConfig

    with open("docs/changelog.md", "r", encoding="utf-8") as f:
        changelog = f.read()

    with open("docs/implementation_guide.md", "r", encoding="utf-8") as f:
        impl = f.read()

    runtime_epsilon = "1e-7"
    assert runtime_epsilon in changelog
    assert runtime_epsilon in impl
    assert "1e-5" not in changelog
    assert "1e-5" not in impl
    assert abs((
        DARSConfig.WEIGHT_RECENCY
        + DARSConfig.WEIGHT_FREQUENCY
        + DARSConfig.WEIGHT_UTILITY
        + DARSConfig.WEIGHT_PREDICTIVE
    ) - 1.0) <= 1e-7
