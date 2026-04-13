import logging
from typing import Optional
from core.layer_d.storage import MemoryVault
from core.layer_c.janitor import DecisionEngine
import asyncio

logger = logging.getLogger(__name__)

class TriageOrchestrator:
    """The Scheduler - Handles the execution of maintenance cycles."""

    MAX_MEMORY_THRESHOLD = 1000

    def __init__(self, vault: Optional[MemoryVault] = None, janitor: Optional[DecisionEngine] = None):
        self.vault = vault or MemoryVault()
        self.janitor = janitor or DecisionEngine(vault=self.vault)

    async def trigger_maintenance(self) -> bool:
        """Volume-Based trigger: runs if total memory count > MAX_MEMORY_THRESHOLD."""
        loop = asyncio.get_running_loop()
        count = await loop.run_in_executor(None, self.vault.count_memories)
        
        if count > self.MAX_MEMORY_THRESHOLD:
            logger.info(f"Volume Trigger Met ({count} > {self.MAX_MEMORY_THRESHOLD}). Starting triage.")
            await self.run_maintenance()
            return True
        else:
            logger.info(f"Volume ({count}) below threshold ({self.MAX_MEMORY_THRESHOLD}). Maintenance skipped.")
            return False

    async def run_maintenance(self) -> None:
        """Manual/Scheduled trigger: iterates DB in chunks to avoid memory spikes."""
        logger.info("Initializing DARS Maintenance Cycle (Layer C)...")
        loop = asyncio.get_running_loop()
        
        def start_generator():
            return self.vault.get_all_memories(limit=100, scroll_yield=True, with_vectors=False)
            
        try:
            gen = await loop.run_in_executor(None, start_generator)
            
            def next_chunk():
                try:
                    return next(gen)
                except StopIteration:
                    return None

            while True:
                chunk_data = await loop.run_in_executor(None, next_chunk)
                if chunk_data is None:
                    break
                
                chunk_points, next_offset = chunk_data
                
                tasks = [self.janitor.triage_memory(pt) for pt in chunk_points]
                if tasks:
                    await asyncio.gather(*tasks)
                
                if next_offset is None:
                    break

            logger.info("Maintenance Cycle Completed successfully.")
        except Exception as e:
            logger.error(f"Maintenance cycle encountered a critical error: {e}")
            raise
