import asyncio
import functools
import logging
from typing import List, Optional

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

    async def process_feedback_loop(
        self,
        query: str,
        response: str,
        retrieved_memories: List[dict],
        current_time: Optional[float] = None,
    ) -> str:
        """
        Triggered asynchronously after user response to judge utility and
        execute atomic DB patches using optimistic-locked operations.

        Returns the judge verdict ("YES", "NO" or "NEUTRAL").
        """
        memory_texts = "\n".join([m.get("payload", {}).get("text_content", "") for m in retrieved_memories])
        judgment = await self.evaluator.evaluate_success(query, response, memory_texts)

        if judgment == "NEUTRAL":
            logger.info("Feedback loop received NEUTRAL evaluation. Skipping metadata patching.")
            return judgment

        await self.apply_feedback(
            [m.get("id") for m in retrieved_memories],
            success=(judgment == "YES"),
            current_time=current_time,
        )
        return judgment

    async def apply_feedback(
        self,
        point_ids: List[str],
        success: bool,
        current_time: Optional[float] = None,
    ) -> None:
        """Apply one success/failure signal to each memory (utility, frequency, recency).

        Used by the judge-driven loop above and by experiments that supply the
        feedback signal from another source (e.g. benchmark ground truth).
        Best-effort: every memory is attempted; the first error is re-raised afterwards.
        """
        loop = asyncio.get_running_loop()

        errors = []
        for pid in point_ids:
            if not pid:
                continue
            try:
                await loop.run_in_executor(None, self.vault.update_utility, pid, success)
                await loop.run_in_executor(None, self.vault.increment_frequency, pid)
                await loop.run_in_executor(
                    None, functools.partial(self.vault.update_recency, pid, current_time=current_time)
                )
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
