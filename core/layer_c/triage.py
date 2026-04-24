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
        from config.settings import DARSConfig
        if vault is None:
            raise TypeError("TriageOrchestrator requires an explicit vault instance")
        self.vault = vault
        self.janitor = janitor or DecisionEngine(vault=self.vault)
        self._active_tasks: set[asyncio.Task] = set()
        self.config = DARSConfig()

    def dispatch_maintenance(self, coro):
        task = asyncio.create_task(coro)
        self._active_tasks.add(task)
        task.add_done_callback(self._active_tasks.discard)
        return task

    async def shutdown(self, timeout: float = None):
        if timeout is None:
            timeout = self.config.SHUTDOWN_TIMEOUT_SECONDS
            
        if self._active_tasks:
            logger.info(f"Finalizing {len(self._active_tasks)} tasks...")
            await asyncio.wait_for(
                asyncio.gather(*self._active_tasks, return_exceptions=True),
                timeout=timeout
            )

    async def process_high_priority_queue(self) -> None:
        """
        Queries Qdrant for memories with `system:high_priority_distillation`
        and processes them asynchronously with bounded concurrency.
        """
        logger.info("Checking High-Priority Distillation Queue...")
        loop = asyncio.get_running_loop()
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        # Build filter for the priority tag
        flag_filter = Filter(
            must=[FieldCondition(key="tags", match=MatchValue(value="system:high_priority_distillation"))]
        )
        
        def _get_all_flagged():
            # fetch all because there should be few outliers
            return self.vault.client.scroll(
                collection_name=self.vault.collection_name,
                scroll_filter=flag_filter,
                limit=100,
                with_vectors=False,
            )
            
        try:
            records, _ = await loop.run_in_executor(None, _get_all_flagged)
            if not records:
                return
                
            logger.info(f"Found {len(records)} high-priority memories.")
            
            # Map PointStructs to MemoryPoint
            from core.layer_d.schema import MemoryPoint, MemoryPayload
            
            points = []
            for r in records:
                p = MemoryPoint(
                    point_id=r.id,
                    vector=[],
                    payload=MemoryPayload(**r.payload) if isinstance(r.payload, dict) else r.payload
                )
                points.append(p)
            
            semaphore = asyncio.Semaphore(self.config.MAX_CONCURRENT_DISTILLATIONS)
            
            async def _process_with_semaphore(pt):
                try:
                    await asyncio.wait_for(semaphore.acquire(), timeout=30.0)
                    try:
                        await self.janitor.triage_memory(pt)
                    finally:
                        semaphore.release()
                except asyncio.TimeoutError:
                    logger.warning(f"Distillation queue saturated, skipping memory {pt.point_id} for this cycle")
                except Exception as e:
                    logger.error(f"Error processing priority memory {pt.point_id}: {e}")

            tasks = [_process_with_semaphore(pt) for pt in points]
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Failed to process high priority queue: {e}")

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
