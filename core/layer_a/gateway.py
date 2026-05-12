import logging
import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

from config.settings import DARSConfig
from core.layer_a.reformulator import QueryReformulator
from core.layer_a.reranker import DARSReranker
from core.layer_a.prompt_constructor import PromptConstructor

logger = logging.getLogger(__name__)

class CognitiveGateway:
    """
    Main Execution Pipeline for Layer A.
    Orchestrates query reformulation, DARS reranking, and XML prompt string construction.
    Non-blocking async implementation.
    """

    def __init__(
        self,
        reformulator: Optional[QueryReformulator] = None,
        reranker: Optional[DARSReranker] = None,
        alpha: float = 0.5,
        fetch_k: Optional[int] = None,
        top_n: Optional[int] = None,
    ):
        if reranker is None:
            raise TypeError("CognitiveGateway requires an explicit reranker instance")

        self.reformulator = reformulator or QueryReformulator()
        self.reranker = reranker
        self.alpha = alpha
        self.fetch_k = int(fetch_k) if fetch_k is not None else int(DARSConfig.DEFAULT_FETCH_K)
        self.top_n = int(top_n) if top_n is not None else int(DARSConfig.DEFAULT_TOP_N)

    async def process_query(
        self, raw_query: str, *, current_time: Optional[float] = None
    ) -> str:
        """
        Executes the three-stage Interaction Layer pipeline.
        
        Args:
            raw_query: Raw user query from UI/Brain.
            
        Returns:
            The augmented XML-injected prompt ready for Brain LLM inference.
        """
        logger.info(f"Processing query through CognitiveGateway: {raw_query}")
        
        # 1. Intent Normalization (Async LLM Call)
        expanded_query = await self.reformulator.reformulate_query(raw_query)
        logger.debug(f"Expanded Query: {expanded_query}")
        
        # 2. DARS Scoring Engine (Run sync vault search in executor to be truly non-blocking)
        loop = asyncio.get_running_loop()
        memories = await loop.run_in_executor(
            None,
            lambda: self.reranker.rerank(
                query=expanded_query,
                fetch_k=self.fetch_k,
                top_n=self.top_n,
                alpha=self.alpha,
                current_time=current_time,
                secondary_query=raw_query,
            ),
        )

        # 3. XML Injection
        prompt = PromptConstructor.build(query=raw_query, memories=memories)

        return prompt

    async def process_query_timed(
        self,
        raw_query: str,
        *,
        current_time: Optional[float] = None,
    ) -> Tuple[str, Dict[str, Any], List]:
        """
        Same as ``process_query`` but returns wall-time breakdown and ranked memories.

        Returns
        -------
        prompt : str
            XML-augmented prompt.
        timings : dict
            Keys: ``reformulate_s``, ``retrieve_s``, ``xml_build_s``, ``gateway_total_s``.
        memories : list
            Top memories returned by reranker (same as used for XML build).
        """
        t0 = time.perf_counter()
        expanded_query = await self.reformulator.reformulate_query(raw_query)
        t1 = time.perf_counter()
        loop = asyncio.get_running_loop()
        memories = await loop.run_in_executor(
            None,
            lambda: self.reranker.rerank(
                query=expanded_query,
                fetch_k=self.fetch_k,
                top_n=self.top_n,
                alpha=self.alpha,
                current_time=current_time,
                secondary_query=raw_query,
            ),
        )
        t2 = time.perf_counter()
        prompt = PromptConstructor.build(query=raw_query, memories=memories)
        t3 = time.perf_counter()
        rq = raw_query.strip()
        eq = expanded_query.strip()
        timings: Dict[str, Any] = {
            "reformulate_s": t1 - t0,
            "retrieve_s": t2 - t1,
            "xml_build_s": t3 - t2,
            "gateway_total_s": t3 - t0,
            "expanded_query": expanded_query,
            "narrative_query_clock": current_time,
            "reformulate_changed": bool(eq and eq != rq),
        }
        return prompt, timings, memories

