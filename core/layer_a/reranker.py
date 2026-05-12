import logging
from typing import List, Optional, Set

from core.layer_d.storage import MemoryVault, chunk_index_from_tags
from core.layer_d.schema import MemoryPoint

logger = logging.getLogger(__name__)


class DARSReranker:
    """
    Scoring Engine component of the Cognitive Gateway (Layer A).
    Narrows candidates from "semantically similar" to "contextually optimal".
    Delegates the heavy lifting of DARS math and alpha-blending to Layer D's MemoryVault.
    """

    def __init__(self, vault: Optional[MemoryVault] = None):
        if vault is None:
            raise TypeError("DARSReranker requires an explicit vault instance")
        self.vault = vault

    def rerank(
        self,
        query: str,
        fetch_k: int = 15,
        top_n: int = 3,
        alpha: float = 0.5,
        *,
        current_time: Optional[float] = None,
        expand_neighbor_chunks: Optional[bool] = None,
        secondary_query: Optional[str] = None,
    ) -> List[MemoryPoint]:
        """
        Retrieves candidates and applies DARS scoring engine.

        Args:
            query: The reformulated query.
            fetch_k: Number of candidates to fetch via semantic search.
            top_n: Number of final memories to return before neighbor expansion.
            alpha: Blend factor for Hybrid search (similarity vs DARS).
            current_time: Reference time for DARS recency (e.g. end-of-narrative virtual clock).
            expand_neighbor_chunks: Pull chunk N±1 around each hit (default from config).
            secondary_query: Optional raw user string; merged when ``MAB_DUAL_QUERY_RETRIEVAL`` is on.
        """
        logger.debug("Executing DARS search_and_rerank for query: %s", query)
        if expand_neighbor_chunks is None:
            expand_neighbor_chunks = bool(self.vault.config.MAB_EXPAND_NEIGHBOR_CHUNKS)

        dual = bool(getattr(self.vault.config, "MAB_DUAL_QUERY_RETRIEVAL", False))
        sec = (secondary_query or "").strip()
        qstrip = query.strip()
        if dual and sec and sec != qstrip:
            r1 = self.vault.search_and_rerank(
                query_text=query,
                fetch_k=fetch_k,
                top_n=fetch_k,
                alpha=alpha,
                current_time=current_time,
            )
            r2 = self.vault.search_and_rerank(
                query_text=sec,
                fetch_k=fetch_k,
                top_n=fetch_k,
                alpha=alpha,
                current_time=current_time,
            )
            merged: dict[str, MemoryPoint] = {}
            for m in r1 + r2:
                prev = merged.get(m.point_id)
                sc = m.score if m.score is not None else 0.0
                if prev is None or sc > (prev.score or 0.0):
                    merged[m.point_id] = m
            ranked = sorted(
                merged.values(), key=lambda x: x.score or 0.0, reverse=True
            )[:top_n]
        else:
            ranked = self.vault.search_and_rerank(
                query_text=query,
                fetch_k=fetch_k,
                top_n=top_n,
                alpha=alpha,
                current_time=current_time,
            )
        if not expand_neighbor_chunks or not ranked:
            return ranked

        neighbor_idx: Set[int] = set()
        for m in ranked:
            ci = chunk_index_from_tags(m.payload.tags)
            if ci is not None:
                neighbor_idx.add(ci - 1)
                neighbor_idx.add(ci + 1)

        neighbor_points = self.vault.fetch_points_for_chunk_indices(neighbor_idx)
        merged: dict[str, MemoryPoint] = {}
        order_keys: List[str] = []

        def _add(pt: MemoryPoint) -> None:
            pid = pt.point_id
            if pid not in merged:
                merged[pid] = pt
                order_keys.append(pid)

        for m in ranked:
            _add(m)
        for n in neighbor_points:
            _add(n)

        combined = [merged[k] for k in order_keys]

        def _sort_key(mp: MemoryPoint) -> tuple:
            ci = chunk_index_from_tags(mp.payload.tags)
            return (ci if ci is not None else 10**9, mp.point_id)

        combined.sort(key=_sort_key)
        return combined
