import asyncio
import functools
import logging
from typing import List

from core.layer_b.evaluator import SuccessEvaluator
from core.layer_b.calculator import ScoreCalculator
from core.layer_d.storage import MemoryVault
from core.layer_d.embedding import EmbeddingEngine
from config.settings import DARSConfig

logger = logging.getLogger(__name__)


class LearningEngine:
    """The Orchestrator - Coordinates the feedback loop and async updates."""

    def __init__(self, evaluator: SuccessEvaluator = None, vault: MemoryVault = None, embedder: EmbeddingEngine = None):
        if vault is None:
            raise TypeError("LearningEngine requires an explicit vault instance")
        self.evaluator = evaluator or SuccessEvaluator()
        self.vault = vault
        self.embedder = embedder or EmbeddingEngine()

    async def process_feedback_loop(self, query: str, response: str, retrieved_memories: List[dict]):
        """
        Triggered asynchronously after user response to judge utility and
        execute atomic DB patches using optimistic-locked operations.
        """
        memory_texts = "\n".join([m.get("payload", {}).get("text_content", "") for m in retrieved_memories])
        judgment = await self.evaluator.evaluate_success(query, response, memory_texts)

        if judgment == "NEUTRAL":
            logger.info("Feedback loop received NEUTRAL evaluation. Skipping metadata patching.")
            return

        success = (judgment == "YES")
        loop = asyncio.get_running_loop()

        errors = []
        for mem in retrieved_memories:
            pid = mem.get("id")
            if not pid:
                continue
            try:
                await loop.run_in_executor(None, self.vault.update_utility, pid, success)
                await loop.run_in_executor(None, self.vault.increment_frequency, pid)
                await loop.run_in_executor(None, self.vault.update_recency, pid)
                logger.info("Atomically updated memory %s (success=%s).", pid, success)
            except Exception as e:
                logger.warning("Failed to update memory %s: %s. Continuing loop.", pid, e)
                errors.append(e)

        if errors:
            raise errors[0]

    async def ingest_new_facts(self, new_facts: List[str]):
        """Ingests 'New Facts' identified by Layer A into storage.

        Predictive value (P) is computed inside store_memory via
        cosine similarity against the active GOAL_VECTOR.  If the
        goal vector is unavailable, store_memory falls back to
        DEFAULT_PREDICTIVE_VALUE (fail-safe).
        """
        loop = asyncio.get_running_loop()
        for fact in new_facts:
            await loop.run_in_executor(
                None,
                functools.partial(self.vault.store_memory, text=fact),
            )
            logger.info("Ingested new fact successfully: %s...", fact[:40])
