import time
import logging
from typing import Optional
from core.layer_d.storage import MemoryVault
from core.layer_c.compressor import SemanticCompressor
import asyncio

logger = logging.getLogger(__name__)

class DecisionEngine:
    """Implement the logic to classify and act on memories based on the S score."""

    def __init__(self, vault: Optional[MemoryVault] = None, compressor: Optional[SemanticCompressor] = None):
        self.vault = vault or MemoryVault()
        self.compressor = compressor or SemanticCompressor(vault=self.vault)
        # Using 1e-7 epsilon for final score comparisons
        self.epsilon = 1e-7

    async def triage_memory(self, memory_point) -> None:
        """Evaluate a single memory and execute pruning, compression, or retention."""
        now = time.time()
        payload = memory_point.payload
        
        # Guard: Do not triage memories created or accessed within the last 24 hours (86400s)
        created_at = getattr(payload, "created_at", 0.0)
        recency = getattr(payload, "recency", 0.0)
        
        if now - created_at < 86400 or now - recency < 86400:
            logger.info(f"Skipping triage for {memory_point.point_id}: Under 24h grace period.")
            return

        # Explicitly calculate DARS score
        score_payload = payload.to_dict() if hasattr(payload, 'to_dict') else payload
        score = self.vault.compute_dars_score(score_payload, current_time=now)
        
        # Decision Policy boundaries (epsilon protected)
        retain_threshold = 0.7 - self.epsilon
        compress_lower = 0.3 - self.epsilon
        compress_upper = 0.7 + self.epsilon
        
        loop = asyncio.get_running_loop()

        if score > retain_threshold:
            # RETAIN (> 0.7)
            updates = {"last_triage_timestamp": now}
            await loop.run_in_executor(None, self.vault.patch_payload, memory_point.point_id, updates)
            logger.info(f"Operation: RETAIN | Memory: {memory_point.point_id} | Score: {score:.3f}")
        
        elif compress_lower <= score <= compress_upper:
            # COMPRESS (0.3 <= S <= 0.7)
            is_compressed = getattr(payload, "is_compressed", False)
            if not is_compressed:
                logger.info(f"Operation: COMPRESS | Memory: {memory_point.point_id} | Score: {score:.3f}")
                text_to_compress = getattr(payload, "text_content", "")
                await self.compressor.compress_memory(memory_point.point_id, text_to_compress)
            else:
                logger.info(f"Memory {memory_point.point_id} already compressed. Skipping.")
                
        else:
            # DELETE (< 0.3)
            # High Severity: Ensure DELETE operations are logged explicitly for audit purposes
            logger.critical(f"AUDIT LOG: Operation: DELETE | Memory: {memory_point.point_id} | Score: {score:.3f} | Triggering permanent removal.")
            await loop.run_in_executor(None, self.vault.delete_memory, memory_point.point_id)
