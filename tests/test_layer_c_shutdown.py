import asyncio
import pytest
from unittest.mock import Mock, patch
from core.layer_c.triage import TriageOrchestrator
from config.settings import DARSConfig

@pytest.mark.asyncio
async def test_triage_shutdown():
    orchestrator = TriageOrchestrator()
    config = DARSConfig()
    
    async def slow_task():
        await asyncio.sleep(0.5)
        
    task = orchestrator.dispatch_maintenance(slow_task())
    assert len(orchestrator._active_tasks) == 1
    
    await orchestrator.shutdown(config.SHUTDOWN_TIMEOUT_SECONDS)
    assert len(orchestrator._active_tasks) == 0
