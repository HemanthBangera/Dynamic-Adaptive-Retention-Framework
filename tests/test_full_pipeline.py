"""
DARS Full Pipeline — End-to-End Integration Test
=================================================
Simulates the complete DARS lifecycle.  All real APIs.  Zero mocks.
"""

import asyncio
import time
import pytest
from tests.conftest import requires_gemini
from core.layer_a.gateway import CognitiveGateway
from core.layer_a.reformulator import QueryReformulator
from core.layer_a.reranker import DARSReranker
from core.layer_a.prompt_constructor import PromptConstructor
from core.layer_b.evaluator import SuccessEvaluator
from core.layer_b.calculator import ScoreCalculator
from core.layer_b.engine import LearningEngine
from core.layer_c.compressor import SemanticCompressor
from core.layer_c.janitor import DecisionEngine
from core.layer_c.triage import TriageOrchestrator


@requires_gemini
class TestFullPipelineE2E:

    @pytest.mark.asyncio
    async def test_complete_lifecycle(self, vault):
        # PHASE 1: Seed memories
        memories_to_store = [
            "The client's preferred programming language is Python 3.12",
            "Project deadline is set for March 15, 2025",
            "The team uses PostgreSQL for the main database",
            "Frontend is built with React and TypeScript",
            "The deployment target is AWS ECS with Fargate",
        ]
        pids = []
        for text in memories_to_store:
            pids.append(vault.store_memory(text, source="user"))
        time.sleep(2)
        assert vault.count_memories() >= 5

        # PHASE 2: User query → Layer A pipeline
        raw_query = "What programming language does the client use?"
        reformulator = QueryReformulator()
        expanded = await reformulator.reformulate_query(raw_query)
        assert isinstance(expanded, str) and len(expanded) > 0

        reranker = DARSReranker(vault=vault)
        loop = asyncio.get_running_loop()
        top_memories = await loop.run_in_executor(
            None, lambda: reranker.rerank(query=expanded, fetch_k=10, top_n=3),
        )
        assert len(top_memories) >= 1

        prompt = PromptConstructor.build(query=raw_query, memories=top_memories)
        assert "<system_context>" in prompt
        assert raw_query in prompt

        # PHASE 3: Layer B — Evaluate + update metadata
        evaluator = SuccessEvaluator()
        agent_response = "The client prefers Python 3.12."
        memory_texts = "\n".join([m.payload.text_content for m in top_memories])
        verdict = await evaluator.evaluate_success(raw_query, agent_response, memory_texts)
        assert verdict in ("YES", "NO", "NEUTRAL")

        if verdict != "NEUTRAL":
            success = verdict == "YES"
            for mem in top_memories:
                vault.update_utility(mem.point_id, success=success)
                vault.increment_frequency(mem.point_id)
                vault.update_recency(mem.point_id)

        # PHASE 4: Layer C — Triage stale memory
        stale_pid = pids[-1]
        vault.patch_payload(stale_pid, {
            "recency": time.time() - 300000,
            "created_at": time.time() - 300000,
            "frequency": 0, "success_count": 0, "failure_count": 8,
            "utility": 0.05, "predictive": 0.0,
        })
        janitor = DecisionEngine(vault=vault)
        stale_mem = vault.get_memory(stale_pid)
        await janitor.triage_memory(stale_mem)
        assert vault.get_memory(stale_pid) is None

        # Fresh memories survive
        for pid in pids[:4]:
            assert vault.get_memory(pid) is not None

    @pytest.mark.asyncio
    async def test_gateway_to_feedback_loop(self, vault):
        vault.store_memory("The API uses REST with JSON responses")
        vault.store_memory("Authentication is done via JWT tokens")
        time.sleep(2)

        reranker = DARSReranker(vault=vault)
        gateway = CognitiveGateway(reranker=reranker)
        prompt = await gateway.process_query("How does the API authentication work?")
        assert "<system_context>" in prompt

        loop = asyncio.get_running_loop()
        memories = await loop.run_in_executor(
            None, lambda: reranker.rerank("API authentication", fetch_k=5, top_n=2),
        )
        engine = LearningEngine(vault=vault)
        retrieved = [{"id": m.point_id, "payload": m.payload.to_dict()} for m in memories]
        await engine.process_feedback_loop(
            query="How does the API authentication work?",
            response="The API uses JWT tokens for authentication.",
            retrieved_memories=retrieved,
        )

    @pytest.mark.asyncio
    async def test_ingest_then_retrieve(self, vault):
        engine = LearningEngine(vault=vault)
        await engine.ingest_new_facts(["The database backup runs at 2 AM UTC daily"])
        time.sleep(2)
        results = vault.semantic_search("database backup schedule", top_k=1)
        assert len(results) >= 1
        assert "backup" in results[0].payload.text_content.lower()


@requires_gemini
class TestMultiInteractionLifecycle:

    @pytest.mark.asyncio
    async def test_score_evolution_over_interactions(self, vault):
        pid = vault.store_memory("The project uses microservices architecture", predictive_value=0.5)
        time.sleep(1)
        initial_mem = vault.get_memory(pid)
        initial_score = vault.compute_dars_score(initial_mem.payload.to_dict())

        for _ in range(3):
            vault.update_utility(pid, success=True)
            vault.increment_frequency(pid)
            vault.update_recency(pid)
            time.sleep(0.3)

        final_mem = vault.get_memory(pid)
        final_score = vault.compute_dars_score(final_mem.payload.to_dict())
        assert final_score > initial_score
        assert final_mem.payload.frequency >= 3
        assert final_mem.payload.success_count >= 3

    @pytest.mark.asyncio
    async def test_memory_degradation_lifecycle(self, vault):
        pid = vault.store_memory("Incorrect information that keeps failing")
        vault.patch_payload(pid, {
            "created_at": time.time() - 200000,
            "recency": time.time() - 100000,
        })
        for _ in range(8):
            vault.update_utility(pid, success=False)

        mem = vault.get_memory(pid)
        score = vault.compute_dars_score(mem.payload.to_dict())
        assert score < 0.4

        janitor = DecisionEngine(vault=vault)
        mem = vault.get_memory(pid)
        await janitor.triage_memory(mem)

        classification = vault.classify_memory(score)
        if classification == "delete":
            assert vault.get_memory(pid) is None
        elif classification == "compress":
            assert vault.get_memory(pid) is not None
