"""
Phase-2 Verifier: Adversarial lifecycle and production-risk tests.

Intent:
- This suite is designed to FIND failure modes before real-data rollout.
- Failing tests are signals of production risk and should NOT be force-fixed by relaxing assertions.
"""

import time
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from freezegun import freeze_time

from config.settings import DARSConfig
from core.layer_a.prompt_constructor import PromptConstructor
from core.layer_b.engine import LearningEngine
from core.layer_c.triage import TriageOrchestrator
from core.layer_d.schema import DARSWeights, MemoryPayload, MemoryPoint
from core.layer_d.storage import MemoryVault


# -----------------------------------------------------------------------------
# Helpers (offline, no network)
# -----------------------------------------------------------------------------


def _offline_vault_math() -> MemoryVault:
    vault = MemoryVault.__new__(MemoryVault)
    vault.config = DARSConfig()
    vault.weights = DARSWeights(
        w_r=vault.config.WEIGHT_RECENCY,
        w_f=vault.config.WEIGHT_FREQUENCY,
        w_u=vault.config.WEIGHT_UTILITY,
        w_p=vault.config.WEIGHT_PREDICTIVE,
    )
    return vault


class _SingleChunkVault:
    """Minimal triage vault stub with one chunk."""

    def __init__(self, points):
        self._points = points

    def count_memories(self):
        return len(self._points)

    def get_all_memories(self, limit=100, scroll_yield=False, with_vectors=False):
        def _gen():
            yield self._points, None

        if scroll_yield:
            return _gen()
        return self._points


# -----------------------------------------------------------------------------
# A) Time-travel decay risks
# -----------------------------------------------------------------------------


@freeze_time("2026-04-15 12:00:00")
def test_phase2_time_travel_neutral_memory_should_be_delete_candidate_by_day3():
    """
    Production-risk assertion:
    A memory with neutral utility/frequency should decay aggressively enough to be
    a delete candidate by Day-3 if never reused.
    """
    vault = _offline_vault_math()
    t0 = time.time()

    payload = {
        "recency": t0,
        "frequency": 0,
        "success_count": 0,
        "failure_count": 0,
        "predictive": 0.5,
    }

    # Time-travel to 72 hours later
    with freeze_time("2026-04-18 12:00:00"):
        s_day3 = vault.compute_dars_score(payload, current_time=time.time())

    # Strict policy expectation for stale neutral memory.
    assert s_day3 <= 0.3


@pytest.mark.asyncio
@freeze_time("2026-04-15 12:00:00")
async def test_phase2_grace_period_bypass():
    """
    Verify the 24-hour grace period skips deletion/compression for new memories.
    """
    from core.layer_c.janitor import DecisionEngine
    from core.layer_d.schema import RetentionDecision
    
    t0 = time.time()
    vault = _offline_vault_math()
    decision_engine = DecisionEngine(vault)
    
    point = MemoryPoint(
        point_id="p1", vector=[],
        payload=MemoryPayload(
            text_content="test", created_at=t0, recency=t0, predictive=0.0
        )
    )
    
    # Time-travel 23 hours
    with freeze_time("2026-04-16 11:00:00"):
        vault.compute_dars_score = Mock()
        await decision_engine.triage_memory(point)
        
        # It should exit before scoring because it's under 24 hours old
        vault.compute_dars_score.assert_not_called()




# -----------------------------------------------------------------------------
# B) Payload stress / type rigidity
# -----------------------------------------------------------------------------


def test_phase2_payload_null_bytes_must_be_removed_from_prompt():
    """
    Production-risk assertion:
    Null bytes in user/memory payload should be removed before prompt assembly.
    Some downstream parsers or middleware choke on '\x00'.
    """
    memories = [
        MemoryPoint(
            point_id="n1",
            vector=[],
            payload=MemoryPayload(text_content="alpha\x00beta <x>", utility=0.4, recency=time.time()),
        )
    ]

    prompt = PromptConstructor.build("query\x00with\x00null", memories)

    assert "\x00" not in prompt


def test_phase2_payload_large_text_must_respect_prompt_size_budget():
    """
    Production-risk assertion:
    Very large memory streams should be budgeted/truncated to avoid context explosion.
    """
    huge = "A" * 120_000
    memories = [
        MemoryPoint(
            point_id="big",
            vector=[],
            payload=MemoryPayload(text_content=huge, utility=0.9, recency=time.time()),
        )
    ]

    prompt = PromptConstructor.build("what is this", memories)

    # Conservative production ceiling for this verifier.
    assert len(prompt) <= 20_000


# -----------------------------------------------------------------------------
# C) Idempotency and re-entry resilience
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_phase2_reentry_triage_should_continue_processing_despite_one_point_failure():
    """
    Production-risk assertion:
    One bad memory should not abort a whole maintenance cycle.
    """
    points = [
        MemoryPoint(point_id="p1", vector=[], payload=MemoryPayload(text_content="a", created_at=1.0, recency=1.0)),
        MemoryPoint(point_id="p2", vector=[], payload=MemoryPayload(text_content="b", created_at=1.0, recency=1.0)),
        MemoryPoint(point_id="p3", vector=[], payload=MemoryPayload(text_content="c", created_at=1.0, recency=1.0)),
    ]

    vault = _SingleChunkVault(points)

    processed = []

    async def _triage(pt):
        processed.append(pt.point_id)
        if pt.point_id == "p2":
            raise RuntimeError("simulated per-point failure")

    janitor = SimpleNamespace(triage_memory=_triage)
    orchestrator = TriageOrchestrator(vault=vault, janitor=janitor)

    with pytest.raises(RuntimeError):
        await orchestrator.run_maintenance()

    # Best-effort expectation: all points attempted even if one fails.
    assert processed == ["p1", "p2", "p3"]


# -----------------------------------------------------------------------------
# D) Cross-layer contract hardening
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_phase2_ingest_new_facts_must_call_embedder_with_string_not_list():
    """
    Production-risk assertion:
    Embedder single-text API should receive str, not list.
    Passing list to a single-text encoder can cause shape/type drift in real models.
    """
    evaluator = Mock()
    evaluator.evaluate_success = AsyncMock(return_value="YES")

    embedder = Mock()
    embedder.encode = Mock(return_value=[0.1, 0.2, 0.3, 0.4])

    vault = Mock()
    vault.store_memory = Mock(return_value="pid-1")

    engine = LearningEngine(evaluator=evaluator, vault=vault, embedder=embedder)

    await engine.ingest_new_facts(["fact-1"])

    embedder.encode.assert_called_once_with("fact-1")


def test_phase2_store_memory_must_clamp_out_of_range_predictive_values():
    """
    Production-risk assertion:
    Caller-provided predictive values should be clamped to [0,1] before persistence.
    """
    vault = MemoryVault.__new__(MemoryVault)
    vault.config = DARSConfig()
    vault.collection_name = "dummy"
    vault.client = Mock()
    vault.embedder = Mock()
    vault.embedder.encode = Mock(return_value=[0.01] * 384)

    _ = vault.store_memory(text="x", predictive_value=3.7)

    payload = vault.client.upsert.call_args.kwargs["points"][0].payload
    assert 0.0 <= payload["predictive"] <= 1.0


@pytest.mark.asyncio
async def test_phase2_feedback_loop_should_be_best_effort_when_one_patch_conflicts():
    """
    Production-risk assertion:
    A single optimistic-lock conflict should not stop patching all remaining memories.
    """
    evaluator = Mock()
    evaluator.evaluate_success = AsyncMock(return_value="YES")

    vault = Mock()

    def _patch(pid, updates):
        if pid == "m2":
            raise RuntimeError("optimistic lock conflict")

    vault.patch_payload = Mock(side_effect=_patch)

    engine = LearningEngine(evaluator=evaluator, vault=vault, embedder=Mock())
    retrieved = [
        {"id": "m1", "payload": {"text_content": "a", "success_count": 0, "failure_count": 0, "frequency": 0}},
        {"id": "m2", "payload": {"text_content": "b", "success_count": 0, "failure_count": 0, "frequency": 0}},
        {"id": "m3", "payload": {"text_content": "c", "success_count": 0, "failure_count": 0, "frequency": 0}},
    ]

    with pytest.raises(RuntimeError):
        await engine.process_feedback_loop("q", "r", retrieved)

    # Best-effort expectation: all patches attempted.
    attempted_ids = [c.args[0] for c in vault.patch_payload.call_args_list]
    assert attempted_ids == ["m1", "m2", "m3"]
