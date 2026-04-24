"""
DARS Cross-Layer Contract Tests
================================
Tests contracts and data flow between layers.  All real APIs.
"""

import time
import pytest
from tests.conftest import requires_gemini
from core.layer_a.reranker import DARSReranker
from core.layer_a.prompt_constructor import PromptConstructor
from core.layer_b.calculator import ScoreCalculator
from core.layer_b.engine import LearningEngine
from core.layer_c.janitor import DecisionEngine


class TestLayerAToD:

    def test_search_results_have_required_payload_fields(self, vault):
        vault.store_memory("Test memory for field check")
        time.sleep(1)
        results = vault.semantic_search("field check", top_k=1)
        assert len(results) >= 1
        p = results[0].payload
        assert hasattr(p, "text_content")
        assert hasattr(p, "utility")
        assert hasattr(p, "recency")
        assert hasattr(p, "frequency")
        assert hasattr(p, "predictive")

    def test_rerank_output_feeds_prompt_constructor(self, vault):
        vault.store_memory("Important project info about deployment")
        time.sleep(1)
        reranker = DARSReranker(vault=vault)
        memories = reranker.rerank("deployment info", fetch_k=5, top_n=1)
        prompt = PromptConstructor.build("Tell me about deployment", memories)
        assert "<memory_stream>" in prompt
        assert "deployment" in prompt.lower()


class TestLayerBToD:

    def test_calculator_output_patchable(self, vault):
        pid = vault.store_memory("Patchable test")
        updates = ScoreCalculator.calculate_updates(True, 0, 0, 0)
        vault.patch_payload(pid, updates)
        mem = vault.get_memory(pid)
        assert mem.payload.success_count == 1
        assert mem.payload.frequency == 1

    def test_calculator_vs_vault_utility_formula_agreement(self, vault):
        calc = ScoreCalculator.calculate_updates(True, 5, 2, 10)
        vault_u = vault._compute_utility_score(6, 2)
        assert abs(calc["utility"] - vault_u) < 0.001

    @requires_gemini
    @pytest.mark.asyncio
    async def test_feedback_loop_writes_to_real_db(self, vault):
        pid = vault.store_memory("Feedback loop DB test")
        mem = vault.get_memory(pid)
        engine = LearningEngine(vault=vault)
        retrieved = [{"id": pid, "payload": mem.payload.to_dict()}]
        await engine.process_feedback_loop(
            "What was stored?",
            "Feedback loop DB test was stored.",
            retrieved,
        )
        updated = vault.get_memory(pid)
        assert updated is not None


class TestLayerCToD:

    @pytest.mark.asyncio
    async def test_triage_retains_high_utility_after_55h(self, vault):
        """After BUG #3 fix: high-utility 55h-old memory is now retained."""
        pid = vault.store_memory("Score read test")
        vault.patch_payload(pid, {
            "recency": time.time() - 200000,
            "created_at": time.time() - 200000,
            "success_count": 10, "failure_count": 0,
            "utility": 0.9, "frequency": 20, "predictive": 0.8,
        })
        decisions = vault.triage_all_memories()
        found = [d for d in decisions if d.point_id == pid]
        assert len(found) == 1
        assert found[0].action == "retain"

    @pytest.mark.asyncio
    async def test_triage_delete_removes_from_db(self, vault):
        pid = vault.store_memory("Delete target")
        vault.patch_payload(pid, {
            "recency": time.time() - 300000,
            "created_at": time.time() - 300000,
            "frequency": 0, "success_count": 0, "failure_count": 10,
            "utility": 0.05, "predictive": 0.0,
        })
        mem = vault.get_memory(pid)
        janitor = DecisionEngine(vault=vault)
        await janitor.triage_memory(mem)
        assert vault.get_memory(pid) is None


class TestLayerBCInteraction:

    def test_repeated_failures_lower_triage_score(self, vault):
        pid = vault.store_memory("Degrading memory")
        vault.patch_payload(pid, {
            "created_at": time.time() - 200000,
            "recency": time.time() - 100000,
        })
        for _ in range(5):
            vault.update_utility(pid, success=False)
        mem = vault.get_memory(pid)
        score = vault.compute_dars_score(mem.payload.to_dict())
        assert score < 0.5

    def test_repeated_successes_raise_triage_score(self, vault):
        pid = vault.store_memory("Improving memory")
        vault.patch_payload(pid, {"created_at": time.time() - 200000})
        for _ in range(10):
            vault.update_utility(pid, success=True)
            vault.increment_frequency(pid)
        vault.update_recency(pid)
        mem = vault.get_memory(pid)
        score = vault.compute_dars_score(mem.payload.to_dict())
        assert score > 0.6
