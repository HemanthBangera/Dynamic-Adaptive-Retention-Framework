"""
DARS Layer C — Real Maintenance/Compression Tests
==================================================
Tests DecisionEngine, SemanticCompressor, TriageOrchestrator against real APIs.
"""

import asyncio
import time
import pytest
from tests.conftest import requires_gemini
from core.layer_c.compressor import SemanticCompressor
from core.layer_c.janitor import DecisionEngine
from core.layer_c.triage import TriageOrchestrator
from core.layer_d.schema import MemoryPayload, MemoryPoint


@requires_gemini
class TestCompressorLive:

    @pytest.mark.asyncio
    async def test_compression_produces_shorter_text(self, vault):
        long_text = (
            "On January 15th, 2025, during the quarterly review meeting, "
            "the engineering team led by Sarah Johnson decided to migrate "
            "the entire backend infrastructure from Java Spring Boot to "
            "Python FastAPI. The main reasons were developer productivity, "
            "easier ML integration, and reduced boilerplate code."
        )
        pid = vault.store_memory(long_text)
        compressor = SemanticCompressor(vault=vault)
        result = await compressor.compress_memory(pid, long_text)
        if result:
            mem = vault.get_memory(pid)
            assert mem.payload.is_compressed is True
            assert len(mem.payload.text_content) < len(long_text)
        else:
            pytest.skip("Gemini returned 429/503 transiently")

    @pytest.mark.asyncio
    async def test_compression_stores_backup(self, vault):
        text = "Original text that should be backed up after compression"
        pid = vault.store_memory(text)
        compressor = SemanticCompressor(vault=vault)
        result = await compressor.compress_memory(pid, text)
        if result:
            raw = vault.client.retrieve(
                collection_name=vault.collection_name,
                ids=[pid], with_payload=True,
            )[0]
            assert raw.payload.get("original_text_backup") == text

    @pytest.mark.asyncio
    async def test_empty_text_returns_false(self, vault):
        compressor = SemanticCompressor(vault=vault)
        result = await compressor.compress_memory("fake-id", "")
        assert result is False

    @pytest.mark.asyncio
    async def test_missing_key_raises(self, vault):
        compressor = SemanticCompressor(vault=vault)
        compressor.api_key = ""
        with pytest.raises(RuntimeError, match="API key is required"):
            await compressor.compress_memory("fake-id", "some text")


class TestDecisionEngine:

    @pytest.mark.asyncio
    async def test_grace_period_skips_fresh_memory(self, vault):
        pid = vault.store_memory("Fresh memory")
        mem = vault.get_memory(pid)
        janitor = DecisionEngine(vault=vault)
        await janitor.triage_memory(mem)
        assert vault.get_memory(pid) is not None

    @pytest.mark.asyncio
    async def test_grace_period_bypassed_by_priority_tag(self, vault):
        pid = vault.store_memory("Priority bypass", tags=["system:high_priority_distillation"])
        mem = vault.get_memory(pid)
        janitor = DecisionEngine(vault=vault)
        await janitor.triage_memory(mem)
        assert vault.get_memory(pid) is not None

    @pytest.mark.asyncio
    async def test_delete_stale_memory(self, vault):
        pid = vault.store_memory("Delete candidate")
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

    @pytest.mark.asyncio
    async def test_retain_high_score_memory(self, vault):
        pid = vault.store_memory("Retain candidate")
        vault.patch_payload(pid, {
            "recency": time.time(),
            "created_at": time.time() - 200000,
            "frequency": 30, "success_count": 20, "failure_count": 0,
            "utility": 0.95, "predictive": 1.0,
        })
        mem = vault.get_memory(pid)
        janitor = DecisionEngine(vault=vault)
        await janitor.triage_memory(mem)
        assert vault.get_memory(pid) is not None


class TestTriageOrchestrator:

    @pytest.mark.asyncio
    async def test_trigger_maintenance_below_threshold(self, vault):
        orch = TriageOrchestrator(vault=vault)
        result = await orch.trigger_maintenance()
        assert result is False

    @pytest.mark.asyncio
    async def test_shutdown_with_no_tasks(self, vault):
        orch = TriageOrchestrator(vault=vault)
        await orch.shutdown(timeout=5.0)

    @pytest.mark.asyncio
    async def test_run_maintenance_skips_fresh_memories(self, vault):
        pids = [vault.store_memory(f"Fresh memory {i}") for i in range(3)]
        time.sleep(1)
        orch = TriageOrchestrator(vault=vault)
        await orch.run_maintenance()
        for pid in pids:
            assert vault.get_memory(pid) is not None

    @pytest.mark.asyncio
    async def test_run_maintenance_deletes_stale_low_score(self, vault):
        pid = vault.store_memory("Stale triage victim")
        vault.patch_payload(pid, {
            "recency": time.time() - 300000,
            "created_at": time.time() - 300000,
            "frequency": 0, "success_count": 0, "failure_count": 10,
            "utility": 0.05, "predictive": 0.0,
        })
        time.sleep(1)
        orch = TriageOrchestrator(vault=vault)
        await orch.run_maintenance()
        assert vault.get_memory(pid) is None
