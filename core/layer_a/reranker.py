import logging
from typing import List, Optional

from core.layer_d.storage import MemoryVault
from core.layer_d.schema import MemoryPoint

logger = logging.getLogger(__name__)

class DARSReranker:
    """
    Scoring Engine component of the Cognitive Gateway (Layer A).
    Narrows candidates from "semantically similar" to "contextually optimal".
    Delegates the heavy lifting of DARS math and alpha-blending to Layer D's MemoryVault.
    """

    def __init__(self, vault: Optional[MemoryVault] = None):
        self.vault = vault or MemoryVault()

    def rerank(self, query: str, fetch_k: int = 15, top_n: int = 3, alpha: float = 0.5) -> List[MemoryPoint]:
        """
        Retrieves candidates and applies DARS scoring engine.
        
        Args:
            query: The reformulated query.
            fetch_k: Number of candidates to fetch via semantic search.
            top_n: Number of final memories to return.
            alpha: Blend factor for Hybrid search (similarity vs DARS).
            
        Returns:
            List of top-ranked MemoryPoint objects.
        """
        logger.debug(f"Executing DARS search_and_rerank for query: {query}")
        return self.vault.search_and_rerank(
            query_text=query,
            fetch_k=fetch_k,
            top_n=top_n,
            alpha=alpha
        )
