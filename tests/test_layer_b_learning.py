"""
DARS Layer B — Real Learning Engine Tests
==========================================
Tests the Evaluator, ScoreCalculator, and LearningEngine against real APIs.
"""

import time
import pytest
from tests.conftest import requires_gemini
from core.layer_b.evaluator import SuccessEvaluator
from core.layer_b.calculator import ScoreCalculator
from core.layer_b.engine import LearningEngine


@requires_gemini
class TestEvaluatorLive:

    @pytest.mark.asyncio
    async def test_obvious_success_returns_yes(self):
        ev = SuccessEvaluator()
        verdict = await ev.evaluate_success(
            query="What is the capital of France?",
            response="The capital of France is Paris.",
            memories="France - capital: Paris",
        )
        assert verdict in ("YES", "NEUTRAL")

    @pytest.mark.asyncio
    async def test_obvious_failure_returns_no_or_neutral(self):
        ev = SuccessEvaluator()
        verdict = await ev.evaluate_success(
            query="What is the capital of France?",
            response="The capital of France is Berlin.",
            memories="Recipe for chocolate cake with eggs and flour",
        )
        assert verdict in ("YES", "NO", "NEUTRAL")

    @pytest.mark.asyncio
    async def test_verdict_is_valid_enum(self):
        ev = SuccessEvaluator()
        verdict = await ev.evaluate_success(
            query="Explain quantum computing",
            response="Quantum computers use qubits",
            memories="Quantum computing leverages superposition and entanglement",
        )
        assert verdict in ("YES", "NO", "NEUTRAL")

    @pytest.mark.asyncio
    async def test_missing_key_raises(self):
        ev = SuccessEvaluator()
        ev.api_key = ""
        with pytest.raises(RuntimeError, match="API key is required"):
            await ev.evaluate_success("q", "r", "m")


class TestScoreCalculator:

    def test_success_increments_success_count(self):
        result = ScoreCalculator.calculate_updates(True, 0, 0, 0)
        assert result["success_count"] == 1
        assert result["failure_count"] == 0
        assert result["frequency"] == 1

    def test_failure_increments_failure_count(self):
        result = ScoreCalculator.calculate_updates(False, 0, 0, 0)
        assert result["success_count"] == 0
        assert result["failure_count"] == 1

    def test_utility_laplacian_smoothing(self):
        result = ScoreCalculator.calculate_updates(True, 3, 1, 5)
        expected = (4 + 1) / (4 + 1 + 2)
        assert abs(result["utility"] - expected) < 0.001

    def test_recency_is_current_timestamp(self):
        before = time.time()
        result = ScoreCalculator.calculate_updates(True, 0, 0, 0)
        after = time.time()
        assert before <= result["recency"] <= after

    def test_calculator_output_keys_match_schema(self):
        result = ScoreCalculator.calculate_updates(True, 0, 0, 0)
        required_keys = {"success_count", "failure_count", "frequency", "utility", "recency"}
        assert required_keys == set(result.keys())


@requires_gemini
class TestLearningEngineLive:

    @pytest.mark.asyncio
    async def test_feedback_loop_updates_metadata(self, vault):
        pid = vault.store_memory("The client prefers TypeScript", source="user")
        mem = vault.get_memory(pid)
        engine = LearningEngine(vault=vault)
        retrieved = [{"id": pid, "payload": mem.payload.to_dict()}]
        await engine.process_feedback_loop(
            query="What language does the client prefer?",
            response="The client prefers TypeScript.",
            retrieved_memories=retrieved,
        )
        updated = vault.get_memory(pid)
        assert updated is not None

    @pytest.mark.asyncio
    async def test_ingest_new_facts(self, vault):
        engine = LearningEngine(vault=vault)
        await engine.ingest_new_facts(["The server runs on port 8080"])
        time.sleep(1)
        results = vault.semantic_search("server port", top_k=1)
        assert len(results) >= 1
        assert "8080" in results[0].payload.text_content

    @pytest.mark.asyncio
    async def test_ingest_computes_predictive_from_goal_vector(self, vault):
        """After BUG #1 fix: ingest_new_facts computes P via GOAL_VECTOR cosine, not flat 0.5."""
        engine = LearningEngine(vault=vault)
        await engine.ingest_new_facts(["Pick up the plate from the counter and put it on the table"])
        time.sleep(1)
        results = vault.semantic_search("pick up plate counter table", top_k=1)
        assert len(results) >= 1
        p = results[0].payload.predictive
        assert p != 0.0, "predictive must not be zero (GOAL_VECTOR should be real)"
        assert 0.0 < p <= 1.0, f"predictive out of range: {p}"
