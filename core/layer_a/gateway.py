import logging
import asyncio
from typing import Optional, List

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

    def __init__(self, reformulator: Optional[QueryReformulator] = None, reranker: Optional[DARSReranker] = None, alpha: float = 0.5):
        if reranker is None:
            raise TypeError("CognitiveGateway requires an explicit reranker instance")
        self.reformulator = reformulator or QueryReformulator()
        self.reranker = reranker
        self.alpha = alpha

    async def process_query(self, raw_query: str) -> str:
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
                fetch_k=15, 
                top_n=3, 
                alpha=self.alpha
            )
        )
        
        # 3. XML Injection 
        prompt = PromptConstructor.build(query=raw_query, memories=memories)
        
        return prompt

