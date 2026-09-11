import time
import logging
from typing import Optional
from config.settings import DARSConfig
from core.layer_d.storage import MemoryVault
from core.layer_c.compressor import SemanticCompressor
import asyncio

logger = logging.getLogger(__name__)

class DecisionEngine:
    """Implement the logic to classify and act on memories based on the S score."""

    def __init__(
        self,
        vault: Optional[MemoryVault] = None,
        compressor: Optional[SemanticCompressor] = None,
        grace_period_s: Optional[float] = None,
    ):
        if vault is None:
            raise TypeError("DecisionEngine requires an explicit vault instance")
        self.vault = vault
        self.compressor = compressor or SemanticCompressor(vault=self.vault)
        self.grace_period_s = (
            float(grace_period_s) if grace_period_s is not None
            else float(DARSConfig.GRACE_PERIOD_SECONDS)
        )

    async def triage_memory(self, memory_point, current_time: Optional[float] = None) -> None:
        """Evaluate a single memory and execute pruning, compression, or retention.

        ``current_time`` lets simulated (virtual-clock) runs apply the grace period
        and recency decay on the simulated timeline instead of wall-clock time.
        """
        now = time.time() if current_time is None else float(current_time)
        payload = memory_point.payload

        raw_tags = getattr(payload, "tags", [])
        tags = raw_tags if raw_tags is not None else []
        is_priority = "system:high_priority_distillation" in tags

        created_at = getattr(payload, "created_at", None) or 0.0
        recency = getattr(payload, "recency", None) or 0.0

        grace = self.grace_period_s
        if not is_priority and (now - created_at < grace or now - recency < grace):
            logger.info("Skipping triage for %s: Under %.0fs grace period.", memory_point.point_id, grace)
            return

        loop = asyncio.get_running_loop()

        if is_priority:
            logger.info("Operation: COMPRESS (PRIORITY) | Memory: %s | Priority Override Active.", memory_point.point_id)
            text_to_compress = getattr(payload, "text_content", "")
            await self.compressor.compress_memory(memory_point.point_id, text_to_compress)
            return

        score_payload = payload.to_dict() if hasattr(payload, 'to_dict') else payload
        score = self.vault.compute_dars_score(score_payload, current_time=now)
        action = self.vault.classify_memory(score)

        if action == "retain":
            updates = {"last_triage_timestamp": now}
            await loop.run_in_executor(None, self.vault.patch_payload, memory_point.point_id, updates)
            logger.info("Operation: RETAIN | Memory: %s | Score: %.3f", memory_point.point_id, score)

        elif action == "compress":
            is_compressed = getattr(payload, "is_compressed", False)
            if not is_compressed:
                logger.info("Operation: COMPRESS | Memory: %s | Score: %.3f", memory_point.point_id, score)
                text_to_compress = getattr(payload, "text_content", "")
                await self.compressor.compress_memory(memory_point.point_id, text_to_compress)
            else:
                logger.info("Memory %s already compressed. Skipping.", memory_point.point_id)

        else:
            logger.critical("AUDIT LOG: Operation: DELETE | Memory: %s | Score: %.3f | Triggering permanent removal.", memory_point.point_id, score)
            await loop.run_in_executor(None, self.vault.delete_memory, memory_point.point_id)
