import asyncio
import logging
from typing import List
import numpy as np

from core.layer_b.evaluator import SuccessEvaluator
from core.layer_b.calculator import ScoreCalculator
from core.layer_d.storage import MemoryVault
from core.layer_d.embedding import EmbeddingEngine
from config.settings import DARSConfig

logger = logging.getLogger(__name__)

class LearningEngine:
    """The Orchestrator - Coordinates the feedback loop and async updates."""
    
    def __init__(self, evaluator: SuccessEvaluator = None, vault: MemoryVault = None, embedder: EmbeddingEngine = None):
        self.evaluator = evaluator or SuccessEvaluator()
        self.vault = vault or MemoryVault()
        self.embedder = embedder or EmbeddingEngine()

    async def process_feedback_loop(self, query: str, response: str, retrieved_memories: List[dict]):
        """
        Triggered asynchronously after user response to judge utility and execute atomic DB patches.
        `retrieved_memories` should be passed containing 'id' and 'payload' attributes.
        """
        memory_texts = "\n".join([m.get("payload", {}).get("text_content", "") for m in retrieved_memories])
        judgment = await self.evaluator.evaluate_success(query, response, memory_texts)
        
        if judgment == "NEUTRAL":
            logger.info("Feedback loop received NEUTRAL evaluation. Skipping metadata patching.")
            return
            
        success = (judgment == "YES")
        loop = asyncio.get_running_loop()
        
        for mem in retrieved_memories:
            pid = mem.get("id")
            payload = mem.get("payload", {})
            
            updates = ScoreCalculator.calculate_updates(
                success=success,
                current_success_count=payload.get("success_count", 0),
                current_failure_count=payload.get("failure_count", 0),
                current_access_count=payload.get("frequency", 0)
            )
            
            # Pushing blocking patch calls onto executor to avoid pausing Gateway IO
            await loop.run_in_executor(None, self.vault.patch_payload, pid, updates)
            logger.info(f"Asynchronously patched memory {pid} with learning updates.")
            
    async def ingest_new_facts(self, new_facts: List[str]):
        """
        Ingests 'New Facts' identified by Layer A into storage seamlessly.
        """
        loop = asyncio.get_running_loop()
        for fact in new_facts:
            # We calculate P here using static GOAL_VECTOR for completeness
            vector = await loop.run_in_executor(None, self.embedder.encode, [fact])
            # Handle mock returning 1D array vs real app returning list of lists
            vec = vector[0] if len(vector) > 0 and isinstance(vector[0], (list, np.ndarray)) else vector
            
            goal = np.array(DARSConfig.GOAL_VECTOR)
            v = np.array(vec)
            
            if np.linalg.norm(goal) > 0 and np.linalg.norm(v) > 0:
                p = float(np.dot(v, goal) / (np.linalg.norm(v) * np.linalg.norm(goal)))
                p = max(0.0, min(1.0, p))
            else:
                p = DARSConfig.DEFAULT_PREDICTIVE_VALUE
            
            import functools
            await loop.run_in_executor(
                None, 
                functools.partial(self.vault.store_memory, text=fact, predictive_value=float(p))
            )
            logger.info(f"Ingested new fact successfully: {fact[:20]}...")
