"""
DARS Layer D – The Memory Vault  (Core Storage Engine)
=======================================================
The persistent foundation of the DARS framework.

Responsibilities
----------------
1. **Collection management**  – create / delete / inspect the Qdrant collection.
2. **Memory CRUD**            – store, retrieve, update, delete memory points.
3. **Semantic search**        – KNN vector search with optional payload filters.
4. **Atomic payload patch**   – update DARS metadata (u, f, r) without
                                re-uploading the heavy embedding vector.
5. **DARS scoring**           – compute S = w_r·R + w_f·F + w_u·U + w_p·P.
6. **Two-stage reranking**    – semantic search → DARS rerank → top-N selection.
7. **Retention classification** – classify memories into retain / compress / delete.

Architecture Reference
----------------------
Layer D stores ``Point Structure: Vector(384) + Payload(id, text, u, f, r, p)``
It serves:
    • Layer A  (search & rerank)
    • Layer B  (atomic payload updates after success evaluation)
    • Layer C  (full-scan scoring & triage)

Tech Stack:  Qdrant (cloud/local) · Python qdrant-client · all-MiniLM-L6-v2
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HasIdCondition,
    MatchValue,
    PointIdsList,
    PointStruct,
    Range,
    VectorParams,
)
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import DARSConfig
from core.layer_d.embedding import EmbeddingEngine
from core.layer_d.schema import (
    DARSWeights,
    MemoryPayload,
    MemoryPoint,
    RetentionDecision,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  MemoryVault  –  Layer D Core Engine
# ═══════════════════════════════════════════════════════════════════════════════


class MemoryVault:
    """
    Layer D: The Storage Layer (The Memory Vault).

    Parameters
    ----------
    config : DARSConfig, optional
        Override the default DARS configuration.
    collection_name : str, optional
        Override the collection name from config (useful for testing).

    Example
    -------
    >>> vault = MemoryVault()
    >>> vault.initialize_collection()
    >>> pid = vault.store_memory("The client prefers Python 3.12")
    >>> results = vault.search_and_rerank("What language does the client use?")
    """

    def __init__(
        self,
        config: Optional[DARSConfig] = None,
        collection_name: Optional[str] = None,
    ):
        self.config = config or DARSConfig()
        self.collection_name = collection_name or self.config.COLLECTION_NAME

        # ── Qdrant client ──────────────────────────────────────────────
        kwargs = {}
        if self.config.QDRANT_URL.startswith("localhost") or self.config.QDRANT_URL.startswith("127.0.0.1"):
            pass # Keep defaults
        else:
            kwargs["prefer_grpc"] = False

        self.client = QdrantClient(
            url=self.config.QDRANT_URL,
            api_key=self.config.QDRANT_API_KEY,
            timeout=60.0,
            **kwargs
        )

        # ── Embedding engine (lazy-loaded) ─────────────────────────────
        self.embedder = EmbeddingEngine(self.config.EMBEDDING_MODEL)

        # ── DARS weight vector ─────────────────────────────────────────
        self.weights = DARSWeights(
            w_r=self.config.WEIGHT_RECENCY,
            w_f=self.config.WEIGHT_FREQUENCY,
            w_u=self.config.WEIGHT_UTILITY,
            w_p=self.config.WEIGHT_PREDICTIVE,
        )
        assert self.weights.validate(), (
            f"DARS weights must sum to 1.0, got "
            f"{self.weights.w_r + self.weights.w_f + self.weights.w_u + self.weights.w_p}"
        )

        logger.info(
            "MemoryVault initialised  [collection=%s, url=%s]",
            self.collection_name,
            self.config.QDRANT_URL[:50] + "...",
        )

    # ═══════════════════════════════════════════════════════════════════
    #  1.  COLLECTION MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=8), reraise=True)
    def initialize_collection(self, recreate: bool = False) -> bool:
        """
        Create the Qdrant collection for DARS memory storage.

        Parameters
        ----------
        recreate : bool
            If True, delete existing collection and create fresh.

        Returns
        -------
        bool
            True if collection was created, False if it already existed.
        """
        exists = self.client.collection_exists(self.collection_name)

        if exists and recreate:
            logger.warning("Recreating collection: %s", self.collection_name)
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            distance_map = {
                "Cosine": Distance.COSINE,
                "Euclid": Distance.EUCLID,
                "Dot": Distance.DOT,
            }
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.config.VECTOR_DIMENSION,
                    distance=distance_map.get(
                        self.config.DISTANCE_METRIC, Distance.COSINE
                    ),
                ),
            )
            
            logger.info(
                "Collection created: %s  (dim=%d, distance=%s)",
                self.collection_name,
                self.config.VECTOR_DIMENSION,
                self.config.DISTANCE_METRIC,
            )
            # Fall through to index creation below...

        # Verify or Create payload indices for optimistic locking (even on existing)
        for field in ["frequency", "success_count", "failure_count"]:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema="integer"
                )
            except Exception as e:
                # If the index already exists, it might throw, or just log info
                logger.debug("Payload index for %s might already exist: %s", field, e)

        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="utility",
                field_schema="float"
            )
        except Exception as e:
            logger.debug("Payload index for utility might already exist: %s", e)

        if not exists:
            return True
            
        logger.info("Collection already exists: %s", self.collection_name)
        return False

    def get_collection_info(self) -> Dict[str, Any]:
        """Return collection statistics (point count, config, etc.)."""
        info = self.client.get_collection(self.collection_name)
        vectors_count = getattr(
            info,
            "vectors_count",
            getattr(info, "indexed_vectors_count", 0),
        )
        return {
            "name": self.collection_name,
            "points_count": info.points_count,
            "vectors_count": vectors_count,
            "status": str(info.status),
            "vector_size": info.config.params.vectors.size,
            "distance": str(info.config.params.vectors.distance),
        }

    def delete_collection(self) -> bool:
        """
        Permanently delete the entire collection.  **USE WITH CAUTION.**

        Returns True if deletion succeeded.
        """
        result = self.client.delete_collection(self.collection_name)
        logger.warning("Collection deleted: %s", self.collection_name)
        return result

    # ═══════════════════════════════════════════════════════════════════
    #  2.  MEMORY CREATION
    # ═══════════════════════════════════════════════════════════════════

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=8), reraise=True)
    def store_memory(
        self,
        text: str,
        predictive_value: Optional[float] = None,
        source: str = "",
        tags: Optional[List[str]] = None,
        vector_override: Optional[List[float]] = None,
    ) -> str:
        """
        Store a new memory with initial DARS metadata.

        Parameters
        ----------
        text : str
            The factual text content of the memory.
        predictive_value : float, optional
            Initial p-score ∈ [0, 1].  Defaults to config value.
        source : str
            Origin label  ("user" | "agent" | "system").
        tags : list of str, optional
            Classification tags.
        vector_override : list of float, optional
            Pre-computed embedding vector (e.g. centroid from clustering).
            When provided, skips internal text encoding.

        Returns
        -------
        str
            The UUID of the newly created point.

        Reference – DARS Specification §17 (Memory Creation):
            m_new = f(observation, action, outcome)
            Access count = 0,  Utility = neutral,  Predictive = estimated.
        """
        point_id = MemoryPoint.generate_id()
        now = time.time()

        vector = vector_override if vector_override is not None else self.embedder.encode(text)

        # Build payload with initial DARS metadata
        p_val = predictive_value
        if p_val is None:
            goal_vec = self.config.get_goal_vector()
            if goal_vec is not None:
                p_val = max(0.0, self.embedder.cosine_similarity(vector, goal_vec))
            else:
                p_val = self.config.DEFAULT_PREDICTIVE_VALUE
        p_val = max(0.0, min(1.0, p_val))
            
        payload = MemoryPayload(
            text_content=text,
            success_count=0,
            failure_count=0,
            utility=0.0,
            frequency=0,
            recency=now,
            predictive=p_val,
            created_at=now,
            is_compressed=False,
            source=source,
            tags=tags or [],
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload.to_dict(),
                )
            ],
        )
        logger.debug("Stored memory %s: '%s...'", point_id, text[:60])
        return point_id

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=8), reraise=True)
    def store_memories_batch(
        self,
        memories: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Batch-insert multiple memories in a single Qdrant upsert.

        Parameters
        ----------
        memories : list of dict
            Each dict must contain ``"text"``; optional keys:
            ``"predictive_value"``, ``"source"``, ``"tags"``.

        Returns
        -------
        list of str
            UUIDs of all created points.
        """
        texts = [m["text"] for m in memories]
        vectors = self.embedder.encode_batch(texts)
        now = time.time()
        point_ids: List[str] = []
        points: List[PointStruct] = []

        for mem, vec in zip(memories, vectors):
            pid = MemoryPoint.generate_id()
            point_ids.append(pid)
            
            p_val = mem.get("predictive_value")
            if p_val is None:
                goal_vec = self.config.get_goal_vector()
                if goal_vec is not None:
                    p_val = max(0.0, self.embedder.cosine_similarity(vec, goal_vec))
                else:
                    p_val = self.config.DEFAULT_PREDICTIVE_VALUE
            p_val = max(0.0, min(1.0, p_val))
                
            payload = MemoryPayload(
                text_content=mem["text"],
                predictive=p_val,
                recency=now,
                created_at=now,
                source=mem.get("source", ""),
                tags=mem.get("tags", []),
            )
            points.append(
                PointStruct(id=pid, vector=vec, payload=payload.to_dict())
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        logger.info("Batch-stored %d memories.", len(points))
        return point_ids

    # ═══════════════════════════════════════════════════════════════════
    #  3.  MEMORY RETRIEVAL
    # ═══════════════════════════════════════════════════════════════════

    def get_memory(self, point_id: str) -> Optional[MemoryPoint]:
        """
        Retrieve a single memory by its point ID.

        Returns None if the point does not exist.
        """
        results = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id],
            with_payload=True,
            with_vectors=True,
        )
        if not results:
            return None
        pt = results[0]
        return MemoryPoint(
            point_id=str(pt.id),
            vector=pt.vector,
            payload=MemoryPayload.from_dict(pt.payload),
        )

    def get_all_memories(
        self, limit: int = 100, with_vectors: bool = False, scroll_yield: bool = False
    ) -> Any:
        """
        Scroll through all memories in the collection.

        Used by Layer C (maintenance scan) to evaluate every memory.

        Parameters
        ----------
        limit : int
            Maximum number of points to return per request/chunk.
        with_vectors : bool
            Whether to include the heavy embedding vectors.
        scroll_yield : bool
            If True, yields (chunk_of_MemoryPoints, next_offset) tuples in a loop.
            If False, returns a single list of up to `limit` MemoryPoints.
        """
        def _generator():
            offset = None
            while True:
                records, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=limit,
                    offset=offset,
                    with_payload=True,
                    with_vectors=with_vectors,
                )
                points = [
                    MemoryPoint(
                        point_id=str(r.id),
                        vector=r.vector if r.vector else [],
                        payload=MemoryPayload.from_dict(r.payload),
                    )
                    for r in records
                ]
                yield points, next_offset
                if next_offset is None:
                    break
                offset = next_offset

        if scroll_yield:
            return _generator()
        
        records, _next_offset = self.client.scroll(
            collection_name=self.collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=with_vectors,
        )
        return [
            MemoryPoint(
                point_id=str(r.id),
                vector=r.vector if r.vector else [],
                payload=MemoryPayload.from_dict(r.payload),
            )
            for r in records
        ]

    def semantic_search(
        self,
        query_text: str,
        top_k: int = 10,
        utility_threshold: Optional[float] = None,
        score_threshold: Optional[float] = None,
    ) -> List[MemoryPoint]:
        """
        Perform pure semantic similarity search.

        Parameters
        ----------
        query_text : str
            Natural-language query to embed and search.
        top_k : int
            Number of nearest neighbours to return.
        utility_threshold : float, optional
            If set, only return memories with utility ≥ this value.
        score_threshold : float, optional
            If set, only return points with cosine similarity ≥ this value.

        Returns
        -------
        list of MemoryPoint
            Ranked by cosine similarity (descending).
        """
        query_vector = self.embedder.encode(query_text)

        # Build optional filter
        query_filter = None
        if utility_threshold is not None:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="utility",
                        range=Range(gte=utility_threshold),
                    )
                ]
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False,
        ).points

        return [
            MemoryPoint(
                point_id=str(hit.id),
                vector=[],
                payload=MemoryPayload.from_dict(hit.payload),
                score=hit.score,
            )
            for hit in results
        ]

    def search_and_rerank(
        self,
        query_text: str,
        fetch_k: Optional[int] = None,
        top_n: Optional[int] = None,
        alpha: Optional[float] = None,
        use_rrf: bool = True,
        rrf_k: int = 60,
        current_time: Optional[float] = None,
    ) -> List[MemoryPoint]:
        """
        Two-stage retrieval  (Layer A pipeline, Stage 2).

        1.  **Semantic search** → fetch ``fetch_k`` candidates.
        2.  **Reranking** → RRF (default) or weighted-sum fallback.
        3.  **Selection** → return the top ``top_n`` results.

        Parameters
        ----------
        query_text : str
            Natural-language query.
        fetch_k : int
            First-stage candidate count  (default: config.DEFAULT_FETCH_K).
        top_n : int
            Final output count  (default: config.DEFAULT_TOP_N).
        alpha : float
            Blend factor for weighted-sum mode (default: config.RERANK_ALPHA).
            Ignored when ``use_rrf=True``.
        use_rrf : bool
            When True, use Reciprocal Rank Fusion (Cormack et al., 2009).
            When False, use legacy weighted-sum: α·sim + (1−α)·DARS.
        rrf_k : int
            RRF smoothing constant (default 60, standard in literature).
        current_time : float, optional
            Reference timestamp for recency calculation.  Defaults to
            ``time.time()``.  Pass a virtual-clock value during simulated
            training / evaluation.

        Returns
        -------
        list of MemoryPoint
            Top-N memories sorted by combined score (descending).
        """
        fetch_k = fetch_k or self.config.DEFAULT_FETCH_K
        top_n = top_n or self.config.DEFAULT_TOP_N
        alpha = alpha if alpha is not None else self.config.RERANK_ALPHA
        now = current_time if current_time is not None else time.time()

        candidates = self.semantic_search(query_text, top_k=fetch_k)

        if not candidates:
            return []

        for mem in candidates:
            mem.dars_score = self.compute_dars_score(
                mem.payload.to_dict(), current_time=now
            )

        if use_rrf:
            sim_ranked = sorted(
                candidates, key=lambda m: m.score or 0.0, reverse=True
            )
            dars_ranked = sorted(
                candidates, key=lambda m: m.dars_score or 0.0, reverse=True
            )
            sim_rank = {id(m): rank for rank, m in enumerate(sim_ranked, 1)}
            dars_rank = {id(m): rank for rank, m in enumerate(dars_ranked, 1)}

            for mem in candidates:
                r_sim = sim_rank[id(mem)]
                r_dars = dars_rank[id(mem)]
                mem.score = 1.0 / (rrf_k + r_sim) + 1.0 / (rrf_k + r_dars)
        else:
            sim_scores = [c.score for c in candidates if c.score is not None]
            min_sim = min(sim_scores) if sim_scores else 0.0
            max_sim = max(sim_scores) if sim_scores else 1.0
            range_sim = max_sim - min_sim

            if range_sim < 0.05:
                for mem in candidates:
                    mem.score = alpha * 1.0 + (1 - alpha) * mem.dars_score
            else:
                for mem in candidates:
                    raw_sim = mem.score if mem.score is not None else 0.0
                    norm_sim = (raw_sim - min_sim) / range_sim
                    mem.score = alpha * norm_sim + (1 - alpha) * mem.dars_score

        candidates.sort(key=lambda m: m.score or 0.0, reverse=True)
        return candidates[:top_n]

    # ═══════════════════════════════════════════════════════════════════
    #  4.  ATOMIC PAYLOAD UPDATES  (Core Layer D Capability)
    # ═══════════════════════════════════════════════════════════════════
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=8), reraise=True)
    def patch_payload(self, point_id: str, updates: Dict[str, Any]) -> None:
        """
        Atomic payload patch  –  update metadata WITHOUT re-uploading the vector.

        This is the key efficiency primitive of Layer D.
        Layer B uses it after every interaction to update u, f, r.

        Parameters
        ----------
        point_id : str
            Target memory UUID.
        updates : dict
            Key-value pairs to merge into the existing payload.
            Example: ``{"utility": 0.85, "frequency": 6, "recency": 1710200000}``
        """
        self.client.set_payload(
            collection_name=self.collection_name,
            payload=updates,
            points=[point_id],
        )
        logger.debug("Patched payload for %s: %s", point_id, updates)

    def update_recency(self, point_id: str) -> float:
        """
        Touch a memory  –  set recency to current time.

        Returns the new timestamp.

        Reference – DARS Specification §19:
            t_i ← current_time
        """
        now = time.time()
        self.patch_payload(point_id, {"recency": now})
        return now

    def increment_frequency(self, point_id: str) -> int:
        """
        Increment the access counter by 1.

        Must first retrieve the current count (Qdrant has no atomic increment).

        Reference – DARS Specification §19:
            a_i ← a_i + 1

        Returns the new frequency value.
        """
        mem = self.get_memory(point_id)
        if mem is None:
            raise ValueError(f"Memory not found: {point_id}")
            
        old_freq = mem.payload.frequency
        new_freq = old_freq + 1
        
        res = self.client.set_payload(
            collection_name=self.collection_name,
            payload={"frequency": new_freq},
            points=Filter(
                must=[
                    FieldCondition(key="frequency", match=MatchValue(value=old_freq)),
                    HasIdCondition(has_id=[point_id])
                ]
            )
        )
        
        if isinstance(res, dict) and res.get("updated") == 0:
            raise RuntimeError(f"Optimistic lock conflict for incrementing frequency of {point_id}")
        elif hasattr(res, "updated") and res.updated == 0:
            raise RuntimeError(f"Optimistic lock conflict for incrementing frequency of {point_id}")
            
        logger.debug("Optimistic increment frequency for %s, %d -> %d", point_id, old_freq, new_freq)
        return new_freq

    def update_utility(self, point_id: str, success: bool) -> float:
        """
        Update utility after a success/failure signal from Layer B.

        Increments the appropriate counter and recomputes:
            U = success_count / (success_count + failure_count + 1)

        Parameters
        ----------
        point_id : str
            Target memory UUID.
        success : bool
            True if the memory contributed to a successful outcome.

        Returns
        -------
        float
            The new utility score.

        Reference – DARS Specification §22 (Utility as Credit Assignment).
        """
        mem = self.get_memory(point_id)
        if mem is None:
            raise ValueError(f"Memory not found: {point_id}")

        payload = mem.payload
        old_success = payload.success_count
        old_failure = payload.failure_count

        if success:
            payload.success_count += 1
        else:
            payload.failure_count += 1
        new_utility = payload.compute_utility()

        updates = {
            "success_count": payload.success_count,
            "failure_count": payload.failure_count,
            "utility": new_utility,
        }

        res = self.client.set_payload(
            collection_name=self.collection_name,
            payload=updates,
            points=Filter(
                must=[
                    FieldCondition(key="success_count", match=MatchValue(value=old_success)),
                    FieldCondition(key="failure_count", match=MatchValue(value=old_failure)),
                    HasIdCondition(has_id=[point_id])
                ]
            )
        )
        if isinstance(res, dict) and res.get("updated") == 0:
            raise RuntimeError(f"Optimistic lock conflict for update_utility of {point_id}")
        elif hasattr(res, "updated") and res.updated == 0:
            raise RuntimeError(f"Optimistic lock conflict for update_utility of {point_id}")
            
        logger.debug("Optimistic updated utility for %s: %s", point_id, updates)
        return new_utility

    def update_on_retrieval(self, point_id: str, success: bool) -> Dict[str, Any]:
        """
        Convenience method:  perform all Layer B updates in one call.

        Updates recency, frequency, and utility atomically.

        Returns a dict of the new values.
        """
        mem = self.get_memory(point_id)
        if mem is None:
            raise ValueError(f"Memory not found: {point_id}")

        now = time.time()
        payload = mem.payload

        # Update counts
        payload.frequency += 1
        if success:
            payload.success_count += 1
        else:
            payload.failure_count += 1
        new_utility = payload.compute_utility()

        updates = {
            "recency": now,
            "frequency": payload.frequency,
            "success_count": payload.success_count,
            "failure_count": payload.failure_count,
            "utility": new_utility,
        }
        self.patch_payload(point_id, updates)
        return updates

    # ═══════════════════════════════════════════════════════════════════
    #  5.  MEMORY DELETION
    # ═══════════════════════════════════════════════════════════════════

    def delete_memory(self, point_id: str) -> None:
        """Permanently delete a single memory by its UUID."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=[point_id]),
        )
        logger.debug("Deleted memory: %s", point_id)

    def delete_memories_batch(self, point_ids: List[str]) -> None:
        """Permanently delete multiple memories."""
        if not point_ids:
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=point_ids),
        )
        logger.info("Batch-deleted %d memories.", len(point_ids))

    # ═══════════════════════════════════════════════════════════════════
    #  6.  DARS SCORE COMPUTATION
    # ═══════════════════════════════════════════════════════════════════

    def _compute_recency(
        self, last_access: float, current_time: Optional[float] = None
    ) -> float:
        """
        R = e^(−λ · Δt)   where Δt is in **hours**.

        Reference – DARS Specification §6.1 (Ebbinghaus decay).
        """
        if current_time is None:
            current_time = time.time()
        delta_hours = max(current_time - last_access, 0) / 3600.0
        return math.exp(-self.config.RECENCY_DECAY_LAMBDA * delta_hours)

    def _compute_frequency(self, access_count: int) -> float:
        """
        F = log(1 + f) / log(1 + F_CAP),  capped at 1.0.

        Reference – DARS Specification §6.2 (TF-IDF / Hebbian inspiration).
        """
        if self.config.FREQUENCY_CAP <= 0:
            return 0.0
        return min(
            math.log(1 + access_count) / math.log(1 + self.config.FREQUENCY_CAP),
            1.0,
        )

    def _compute_utility_score(
        self, success_count: int, failure_count: int
    ) -> float:
        """
        U = (success + 1) / (success + failure + 2).
        Laplacian smoothing instead of standard average to avoid dropping utility to 0 instantly.

        Reference – DARS Specification §6.3.
        """
        return (success_count + 1) / (success_count + failure_count + 2)

    def compute_dars_score(
        self,
        payload: Dict[str, Any],
        current_time: Optional[float] = None,
    ) -> float:
        """
        Compute the full DARS retention score.

            S = w_r·R + w_f·F + w_u·U + w_p·P

        Parameters
        ----------
        payload : dict
            Memory payload (must contain recency, frequency, success_count,
            failure_count, predictive).
        current_time : float, optional
            Reference time for recency calculation.

        Returns
        -------
        float
            DARS score in [0, 1].

        Reference – DARS Specification §8.
        """
        R = self._compute_recency(
            payload.get("recency", time.time()), current_time
        )
        F = self._compute_frequency(payload.get("frequency", 0))
        U = self._compute_utility_score(
            payload.get("success_count", 0),
            payload.get("failure_count", 0),
        )
        P = payload.get("predictive", self.config.DEFAULT_PREDICTIVE_VALUE)

        score = (
            self.weights.w_r * R
            + self.weights.w_f * F
            + self.weights.w_u * U
            + self.weights.w_p * P
        )
        return round(min(max(score, 0.0), 1.0), 6)

    # ═══════════════════════════════════════════════════════════════════
    #  7.  RETENTION CLASSIFICATION  (for Layer C)
    # ═══════════════════════════════════════════════════════════════════

    def classify_memory(self, dars_score: float) -> str:
        """
        Map a DARS score to a retention action.

            S > 0.7          → "retain"
            0.3 < S ≤ 0.7   → "compress"
            S ≤ 0.3          → "delete"

        Reference – DARS Specification §9 (Retention Policy).
        """
        if dars_score > self.config.THRESHOLD_RETAIN:
            return "retain"
        elif dars_score > self.config.THRESHOLD_COMPRESS:
            return "compress"
        else:
            return "delete"

    def triage_all_memories(
        self, limit: int = 500
    ) -> List[RetentionDecision]:
        """
        Scan the entire collection and classify each memory.

        This is the entry-point for Layer C's maintenance cycle.

        Returns
        -------
        list of RetentionDecision
            One decision per memory, sorted by DARS score ascending
            (worst memories first).
        """
        now = time.time()
        decisions: List[RetentionDecision] = []

        for chunk_points, _next_offset in self.get_all_memories(limit=limit, with_vectors=False, scroll_yield=True):
            for mem in chunk_points:
                score = self.compute_dars_score(mem.payload.to_dict(), current_time=now)
                action = self.classify_memory(score)
                decisions.append(
                    RetentionDecision(
                        action=action,
                        dars_score=score,
                        point_id=mem.point_id,
                        text_preview=mem.payload.text_content[:80],
                    )
                )

        decisions.sort(key=lambda d: d.dars_score)
        return decisions

    # ═══════════════════════════════════════════════════════════════════
    #  8.  UTILITY HELPERS
    # ═══════════════════════════════════════════════════════════════════

    def count_memories(self) -> int:
        """Return the total number of memory points in the collection."""
        info = self.client.get_collection(self.collection_name)
        return info.points_count

    def health_check(self) -> Dict[str, Any]:
        """
        Quick diagnostic:  connection status + collection stats.

        Returns a dict with connection and collection health info.
        """
        try:
            collections = self.client.get_collections()
            col_names = [c.name for c in collections.collections]
            exists = self.collection_name in col_names
            info = {}
            if exists:
                info = self.get_collection_info()
            return {
                "connected": True,
                "collection_exists": exists,
                "collection_info": info,
                "total_collections": len(col_names),
            }
        except Exception as e:
            return {"connected": False, "error": str(e)}
