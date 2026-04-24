"""
DARS Layer A — Real Gemini + Qdrant Integration Tests
=====================================================
Tests the Cognitive Gateway pipeline: Reformulator → Reranker → PromptConstructor.
All calls hit real APIs.
"""

import time
import pytest
from tests.conftest import requires_gemini
from core.layer_a.reformulator import QueryReformulator
from core.layer_a.reranker import DARSReranker
from core.layer_a.prompt_constructor import PromptConstructor
from core.layer_a.gateway import CognitiveGateway
from core.layer_d.schema import MemoryPayload, MemoryPoint


@requires_gemini
class TestReformulatorLive:

    @pytest.mark.asyncio
    async def test_expansion_returns_string(self):
        r = QueryReformulator()
        result = await r.reformulate_query("What is the budget?")
        assert isinstance(result, str) and len(result) > 0

    @pytest.mark.asyncio
    async def test_proper_noun_preserved(self):
        r = QueryReformulator()
        result = await r.reformulate_query("What is the Pista project deadline?")
        if result != "What is the Pista project deadline?":
            assert "Pista" in result

    @pytest.mark.asyncio
    async def test_expansion_respects_char_limit(self):
        r = QueryReformulator()
        result = await r.reformulate_query("Explain machine learning in detail")
        assert len(result) <= r.max_expansion_chars

    @pytest.mark.asyncio
    async def test_fallback_on_missing_key(self):
        r = QueryReformulator()
        r.api_key = ""
        result = await r.reformulate_query("Test fallback")
        assert result == "Test fallback"


class TestRerankerLive:

    def test_rerank_returns_memory_points(self, vault):
        vault.store_memory("Python is used for machine learning")
        vault.store_memory("JavaScript is used for frontend")
        time.sleep(1)
        reranker = DARSReranker(vault=vault)
        results = reranker.rerank("machine learning", fetch_k=5, top_n=2)
        assert all(isinstance(r, MemoryPoint) for r in results)

    def test_rerank_empty_collection(self, vault):
        reranker = DARSReranker(vault=vault)
        results = reranker.rerank("anything", fetch_k=5, top_n=2)
        assert results == []


class TestPromptConstructor:

    def _make_memory(self, text, utility=0.5, dars_score=None):
        return MemoryPoint(
            point_id="test-id", vector=[],
            payload=MemoryPayload(text_content=text, utility=utility),
            dars_score=dars_score,
        )

    def test_xml_structure(self):
        mem = self._make_memory("Test memory text")
        prompt = PromptConstructor.build("What is this?", [mem])
        assert "<system_context>" in prompt
        assert "<memory_stream>" in prompt
        assert "<current_user_query>" in prompt

    def test_xml_escaping(self):
        mem = self._make_memory("x < y & z > w")
        prompt = PromptConstructor.build("test", [mem])
        assert "&lt;" in prompt
        assert "&amp;" in prompt
        assert "&gt;" in prompt

    def test_null_bytes_stripped(self):
        mem = self._make_memory("before\x00after")
        prompt = PromptConstructor.build("test", [mem])
        assert "\x00" not in prompt
        assert "beforeafter" in prompt

    def test_large_text_truncated(self):
        mem = self._make_memory("A" * 15000)
        prompt = PromptConstructor.build("test", [mem])
        assert "[TRUNCATED FOR BUDGET" in prompt

    def test_max_prompt_budget(self):
        mems = [self._make_memory(f"Memory content {'X' * 5000}") for _ in range(10)]
        prompt = PromptConstructor.build("test query", mems)
        assert len(prompt) <= 25000


@requires_gemini
class TestCognitiveGatewayLive:

    @pytest.mark.asyncio
    async def test_gateway_produces_xml_prompt(self, vault):
        vault.store_memory("The project deadline is March 2025")
        time.sleep(1)
        reranker = DARSReranker(vault=vault)
        gateway = CognitiveGateway(reranker=reranker)
        prompt = await gateway.process_query("When is the project deadline?")
        assert "<system_context>" in prompt
        assert "<current_user_query>" in prompt

    @pytest.mark.asyncio
    async def test_gateway_uses_raw_query_in_prompt(self, vault):
        vault.store_memory("Some relevant memory")
        time.sleep(1)
        reranker = DARSReranker(vault=vault)
        gateway = CognitiveGateway(reranker=reranker)
        raw = "my exact question"
        prompt = await gateway.process_query(raw)
        assert "my exact question" in prompt

    @pytest.mark.asyncio
    async def test_gateway_empty_collection(self, vault):
        reranker = DARSReranker(vault=vault)
        gateway = CognitiveGateway(reranker=reranker)
        prompt = await gateway.process_query("anything")
        assert "<memory_stream>" in prompt
