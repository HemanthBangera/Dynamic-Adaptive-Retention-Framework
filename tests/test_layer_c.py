import pytest
import time
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from core.layer_d.storage import MemoryVault
from core.layer_c.triage import TriageOrchestrator
from core.layer_c.janitor import DecisionEngine
from core.layer_c.compressor import SemanticCompressor

@pytest.fixture(scope="module")
def vault():
    v = MemoryVault(collection_name="test_dars_layer_c")
    v.initialize_collection(recreate=True)
    yield v

@pytest.fixture(autouse=True)
def clear_vault(vault):
    # clear points before each test
    try:
        from qdrant_client.models import Filter
        vault.client.delete(
            collection_name=vault.collection_name, 
            points_selector=Filter()
        )
    except Exception:
        pass

@pytest.mark.asyncio
async def test_retention_policy(vault):
    """Ingest a memory with S=0.85 and verify it remains untouched after triage."""
    pid = vault.store_memory("Retention memory text")
    # artificially age it beyond 24h grace period
    past_time = time.time() - 90000
    vault.patch_payload(pid, {"created_at": past_time, "recency": past_time})
    
    janitor = DecisionEngine(vault=vault)
    with patch.object(vault, "compute_dars_score", return_value=0.85):
        pt = vault.get_memory(pid)
        await janitor.triage_memory(pt)
        
    retrieved = vault.get_memory(pid)
    assert retrieved is not None
    # Payload not deleted. Should have updated last_triage_timestamp
    raw_payload = vault.client.retrieve(collection_name=vault.collection_name, ids=[pid])[0].payload
    assert "last_triage_timestamp" in raw_payload
    assert vault.count_memories() == 1


@pytest.mark.asyncio
async def test_deletion_policy(vault):
    """Ingest a memory with S=0.15 and verify it is successfully removed."""
    pid = vault.store_memory("Deletion memory text")
    # artificially age it
    past_time = time.time() - 90000
    vault.patch_payload(pid, {"created_at": past_time, "recency": past_time})
    
    janitor = DecisionEngine(vault=vault)
    with patch.object(vault, "compute_dars_score", return_value=0.15):
        pt = vault.get_memory(pid)
        await janitor.triage_memory(pt)
        
    retrieved = vault.get_memory(pid)
    assert retrieved is None
    assert vault.count_memories() == 0


@pytest.mark.asyncio
async def test_compression_shadow_integrity(vault):
    """Ingest a memory S=0.5, verify text is summarized, original text backup exists,
    and semantic search for old keyword still returns it (Shadow Index works)."""
    original_text = "The launch sequence code is ALBATROSS in system memory."
    pid = vault.store_memory(original_text)
    
    # age it
    past_time = time.time() - 90000
    vault.patch_payload(pid, {"created_at": past_time, "recency": past_time})
    
    compressor = SemanticCompressor(vault=vault)
    compressor.api_key = "real_key"
    
    janitor = DecisionEngine(vault=vault, compressor=compressor)
    with patch.object(vault, "compute_dars_score", return_value=0.5), patch("aiohttp.ClientSession.post") as mock_post:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "[Summary] launch sequence note."}]}}]
        }
        mock_post.return_value.__aenter__.return_value = mock_resp

        pt = vault.get_memory(pid)
        await janitor.triage_memory(pt)
        
    # Check payload
    raw_payload = vault.client.retrieve(collection_name=vault.collection_name, ids=[pid])[0].payload
    assert raw_payload.get("is_compressed") is True
    assert "ALBATROSS" not in raw_payload.get("text_content", "")
    assert "ALBATROSS" in raw_payload.get("original_text_backup", "")
    
    # CRITICAL: Semantic Search using the target "ALBATROSS" 
    results = vault.semantic_search("ALBATROSS sequence", top_k=1)
    assert len(results) > 0
    assert results[0].point_id == str(pid)

@pytest.mark.asyncio
async def test_volume_trigger():
    """Verify run_maintenance() only executes when volume threshold is met."""
    mock_vault = Mock()
    mock_vault.count_memories.return_value = 1001
    
    orchestrator = TriageOrchestrator(vault=mock_vault)
    orchestrator.run_maintenance = AsyncMock()
    
    triggered = await orchestrator.trigger_maintenance()
    assert triggered is True
    orchestrator.run_maintenance.assert_called_once()
    
    # Below threshold
    mock_vault.count_memories.return_value = 999
    orchestrator.run_maintenance.reset_mock()
    triggered = await orchestrator.trigger_maintenance()
    assert triggered is False
    orchestrator.run_maintenance.assert_not_called()
