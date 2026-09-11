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

Tech Stack:  Qdrant (cloud/server or local) · Python qdrant-client · all-MiniLM-L6-v2

Backends
--------
``QDRANT_URL`` selects a Qdrant server or Qdrant Cloud (approximate HNSW search).
``location=":memory:"`` (or ``QDRANT_LOCATION``) selects the in-process local
store, and a folder path selects on-disk local storage; local mode performs
exact nearest-neighbour search.
"""

from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HasIdCondition,
    IsEmptyCondition,
    MatchValue,
    PayloadField,
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

RANK_MODES = ("similarity", "rrf", "wrrf", "blend")
_UPSERT_BATCH = 256


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
    weights : DARSWeights, optional
        Per-vault DARS weight vector (ablations, sensitivity analysis).
        Defaults to the weights in ``config``.
    location : str, optional
        ``":memory:"`` for an in-process local store or a folder path for an
        on-disk local store.  Falls back to ``config.QDRANT_LOCATION``, then to
        ``config.QDRANT_URL``.  Each ``":memory:"`` vault is its own store.

    Example
    -------
    >>> vault = MemoryVault(location=":memory:")
    >>> vault.initialize_collection()
    >>> pid = vault.store_memory("The client prefers Python 3.12")
    >>> results = vault.search_and_rerank("What language does the client use?")
    """

    def __init__(
        self,
        config: Optional[DARSConfig] = None,
        collection_name: Optional[str] = None,
        *,
        weights: Optional[DARSWeights] = None,
        location: Optional[str] = None,
    ):
        self.config = config or DARSConfig()
        self.collection_name = collection_name or self.config.COLLECTION_NAME

        # ── Qdrant client ──────────────────────────────────────────────
        loc = location if location is not None else (self.config.QDRANT_LOCATION or None)
        if loc:
            if loc == ":memory:":
                self.client = QdrantClient(location=":memory:")
                self.backend = "local-memory"
            else:
                self.client = QdrantClient(path=loc)
                self.backend = "local-path"
            where = loc
        elif self.config.QDRANT_URL:
            kwargs = {}
            if not (
                self.config.QDRANT_URL.startswith("localhost")
                or self.config.QDRANT_URL.startswith("127.0.0.1")
            ):
                kwargs["prefer_grpc"] = False
            self.client = QdrantClient(
                url=self.config.QDRANT_URL,
                api_key=self.config.QDRANT_API_KEY,
                timeout=60.0,
                **kwargs,
            )
            self.backend = "remote"
            where = self.config.QDRANT_URL[:50] + "..."
        else:
            raise ValueError(
                "No vector store configured: set QDRANT_URL (server/cloud) or "
                "QDRANT_LOCATION (':memory:' or a folder path), or pass location=..."
            )

        # Serialises version-checked writes on in-process (local) stores; see _conditional_patch.
        self._write_lock = threading.RLock()

        # ── Embedding engine (lazy-loaded) ─────────────────────────────
        self.embedder = EmbeddingEngine(self.config.EMBEDDING_MODEL)

        # ── DARS weight vector ─────────────────────────────────────────
        self.weights = weights or DARSWeights(
            w_r=self.config.WEIGHT_RECENCY,
            w_f=self.config.WEIGHT_FREQUENCY,
            w_u=self.config.WEIGHT_UTILITY,
            w_p=self.config.WEIGHT_PREDICTIVE,
        )
        if not self.weights.validate():
            raise ValueError(
                "DARS weights must sum to 1.0, got "
                f"{self.weights.w_r + self.weights.w_f + self.weights.w_u + self.weights.w_p}"
            )

        logger.info(
            "MemoryVault initialised  [collection=%s, backend=%s, where=%s]",
            self.collection_name,
            self.backend,
            where,
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

        # Verify or Create payload indices used by filters (even on existing)
        for field_name in ["frequency", "success_count", "failure_count", "version"]:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema="integer"
                )
            except Exception as e:
                # If the index already exists, it might throw, or just log info
                logger.debug("Payload index for %s might already exist: %s", field_name, e)

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

    def _initial_predictive(self, vector: List[float], predictive_value: Optional[float]) -> float:
        """Predictive value on creation: explicit value, else goal alignment, clamped to [0, 1]."""
        p_val = predictive_value
        if p_val is None:
            goal_vec = self.config.get_goal_vector()
            if goal_vec is not None:
                p_val = max(0.0, self.embedder.cosine_similarity(vector, goal_vec))
            else:
                p_val = self.config.DEFAULT_PREDICTIVE_VALUE
        return max(0.0, min(1.0, p_val))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=8), reraise=True)
    def store_memory(
        self,
        text: str,
        predictive_value: Optional[float] = None,
        source: str = "",
        tags: Optional[List[str]] = None,
        vector_override: Optional[List[float]] = None,
        *,
        sim_timestamp: Optional[float] = None,
        current_time: Optional[float] = None,
        mab_injection_boost: bool = False,
    ) -> str:
        """
        Store a new memory with initial DARS metadata.

        Parameters
        ----------
        text : str
            The factual text content of the memory.
        predictive_value : float, optional
            Initial p-score ∈ [0, 1].  Defaults to goal-vector alignment.
        source : str
            Origin label  ("user" | "agent" | "system").
        tags : list of str, optional
            Classification tags.
        vector_override : list of float, optional
            Pre-computed embedding vector (e.g. centroid from clustering).
            When provided, skips internal text encoding.
        sim_timestamp, current_time
            If set, used as ``recency`` and ``created_at`` (virtual clock);
            ``sim_timestamp`` takes precedence.
        mab_injection_boost
            When True, seeds ``success_count`` from ``DARSConfig.MAB_INJECTION_INITIAL_SUCCESS``
            for acquisition-phase Laplace-friendly utility (benchmark only).

        Returns
        -------
        str
            The UUID of the newly created point.

        Reference – DARS Specification §17 (Memory Creation):
            m_new = f(observation, action, outcome)
            Access count = 0,  Utility = neutral,  Predictive = estimated.
        """
        point_id = MemoryPoint.generate_id()
        ts = sim_timestamp if sim_timestamp is not None else current_time
        now = float(ts) if ts is not None else time.time()

        vector = vector_override if vector_override is not None else self.embedder.encode(text)
        p_val = self._initial_predictive(vector, predictive_value)

        inj_succ = 0
        if mab_injection_boost:
            inj_succ = max(0, int(self.config.MAB_INJECTION_INITIAL_SUCCESS))

        payload = MemoryPayload(
            text_content=text,
            success_count=inj_succ,
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
        current_time: Optional[float] = None,
    ) -> List[str]:
        """
        Batch-insert multiple memories (upserted in chunks of 256 points).

        Parameters
        ----------
        memories : list of dict
            Each dict must contain ``"text"``; optional keys:
            ``"predictive_value"``, ``"source"``, ``"tags"``,
            ``"vector"`` (pre-computed embedding), ``"timestamp"``
            (virtual clock for recency / created_at) and ``"extra"``
            (additional payload fields, e.g. experiment bookkeeping).
        current_time : float, optional
            Default timestamp for memories without their own ``"timestamp"``.

        Returns
        -------
        list of str
            UUIDs of all created points.
        """
        if not memories:
            return []
        missing = [i for i, m in enumerate(memories) if m.get("vector") is None]
        encoded = self.embedder.encode_batch([memories[i]["text"] for i in missing]) if missing else []
        vectors: List[Optional[List[float]]] = [m.get("vector") for m in memories]
        for i, vec in zip(missing, encoded):
            vectors[i] = vec

        default_ts = float(current_time) if current_time is not None else time.time()
        point_ids: List[str] = []
        points: List[PointStruct] = []

        for mem, vec in zip(memories, vectors):
            pid = MemoryPoint.generate_id()
            point_ids.append(pid)
            ts = float(mem["timestamp"]) if mem.get("timestamp") is not None else default_ts
            payload = MemoryPayload(
                text_content=mem["text"],
                predictive=self._initial_predictive(vec, mem.get("predictive_value")),
                recency=ts,
                created_at=ts,
                source=mem.get("source", ""),
                tags=mem.get("tags", []),
            ).to_dict()
            if mem.get("extra"):
                payload.update(mem["extra"])
            points.append(PointStruct(id=pid, vector=vec, payload=payload))

        for start in range(0, len(points), _UPSERT_BATCH):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[start:start + _UPSERT_BATCH],
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
        query_vector: Optional[List[float]] = None,
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
        query_vector : list of float, optional
            Pre-computed query embedding (skips encoding ``query_text``).

        Returns
        -------
        list of MemoryPoint
            Ranked by cosine similarity (descending).
        """
        if query_vector is None:
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
        *,
        rank_mode: Optional[str] = None,
        beta_sim: float = 1.0,
        beta_dars: float = 1.0,
        query_vector: Optional[List[float]] = None,
        return_components: bool = False,
    ) -> List[MemoryPoint]:
        """
        Two-stage retrieval  (Layer A pipeline, Stage 2).

        1.  **Semantic search** → fetch ``fetch_k`` candidates.
        2.  **Reranking** → one of ``RANK_MODES``.
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
            Blend factor for ``blend`` mode (default: config.RERANK_ALPHA).
        use_rrf : bool
            Legacy switch used when ``rank_mode`` is not given:
            True → ``rrf``, False → ``blend``.
        rrf_k : int
            RRF smoothing constant (default 60, standard in literature).
        current_time : float, optional
            Reference timestamp for recency calculation.  Defaults to
            ``time.time()``.  Pass a virtual-clock value during simulated
            training / evaluation.
        rank_mode : str, optional
            ``similarity`` – cosine similarity only (DARS computed but unused);
            ``rrf``        – Reciprocal Rank Fusion of the similarity and DARS
                             rankings (Cormack et al., 2009), equal votes;
            ``wrrf``       – weighted RRF: β_s/(k+r_sim) + β_d/(k+r_dars);
            ``blend``      – α·norm_sim + (1−α)·DARS with variance-aware min-max.
        beta_sim, beta_dars : float
            Vote weights for ``wrrf``.
        query_vector : list of float, optional
            Pre-computed query embedding.
        return_components : bool
            If True, attach ``components`` (R, F, U, P, S, sim, sim_rank,
            dars_rank) to each returned MemoryPoint.

        Returns
        -------
        list of MemoryPoint
            Top-N memories sorted by combined score (descending).
        """
        mode = rank_mode or ("rrf" if use_rrf else "blend")
        if mode not in RANK_MODES:
            raise ValueError(f"Unknown rank_mode {mode!r}; expected one of {RANK_MODES}")
        fetch_k = fetch_k or self.config.DEFAULT_FETCH_K
        top_n = top_n or self.config.DEFAULT_TOP_N
        alpha = alpha if alpha is not None else self.config.RERANK_ALPHA
        now = current_time if current_time is not None else time.time()

        candidates = self.semantic_search(query_text, top_k=fetch_k, query_vector=query_vector)

        if not candidates:
            return []

        comps: Dict[int, Dict[str, float]] = {}
        for mem in candidates:
            c = self.compute_components(mem.payload.to_dict(), current_time=now)
            mem.dars_score = self.score_from_components(c)
            c["S"] = mem.dars_score
            c["sim"] = mem.score if mem.score is not None else 0.0
            comps[id(mem)] = c

        sim_ranked = sorted(candidates, key=lambda m: m.score or 0.0, reverse=True)
        dars_ranked = sorted(candidates, key=lambda m: m.dars_score or 0.0, reverse=True)
        sim_rank = {id(m): rank for rank, m in enumerate(sim_ranked, 1)}
        dars_rank = {id(m): rank for rank, m in enumerate(dars_ranked, 1)}

        if mode == "similarity":
            for mem in candidates:
                mem.score = comps[id(mem)]["sim"]
        elif mode in ("rrf", "wrrf"):
            b_s, b_d = (1.0, 1.0) if mode == "rrf" else (float(beta_sim), float(beta_dars))
            for mem in candidates:
                mem.score = b_s / (rrf_k + sim_rank[id(mem)]) + b_d / (rrf_k + dars_rank[id(mem)])
        else:  # blend
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

        if return_components:
            for mem in candidates:
                c = comps[id(mem)]
                c["sim_rank"] = float(sim_rank[id(mem)])
                c["dars_rank"] = float(dars_rank[id(mem)])
                mem.components = c

        candidates.sort(key=lambda m: m.score or 0.0, reverse=True)
        return candidates[:top_n]

    def mean_dars_score_all_points(
        self,
        current_time: Optional[float] = None,
        scroll_limit: int = 256,
    ) -> Tuple[float, int]:
        """
        Mean DARS retention score S over all points in the collection.

        Used for MemoryAgentBench ``dars_mass_ratio`` (retrieved mean vs vault mean).
        """
        gen = self.get_all_memories(limit=scroll_limit, with_vectors=False, scroll_yield=True)
        vals: List[float] = []
        now = current_time if current_time is not None else time.time()
        for chunk, _ in gen:
            for p in chunk:
                vals.append(self.compute_dars_score(p.payload.to_dict(), current_time=now))
        if not vals:
            return 0.0, 0
        return float(sum(vals) / len(vals)), int(len(vals))

    # ═══════════════════════════════════════════════════════════════════
    #  4.  ATOMIC PAYLOAD UPDATES  (Core Layer D Capability)
    # ═══════════════════════════════════════════════════════════════════
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=8), reraise=True)
    def patch_payload(self, point_id: str, updates: Dict[str, Any]) -> None:
        """
        Unguarded payload patch – update metadata WITHOUT re-uploading the vector.

        Last writer wins; use the versioned update methods (``update_utility``,
        ``increment_frequency``, ``update_on_retrieval``) for counters that
        concurrent writers may touch.

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

    def _read_raw_payload(self, point_id: str) -> Optional[Dict[str, Any]]:
        """Return the stored payload dict of a point, or None if it does not exist."""
        records = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id],
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            return None
        return dict(records[0].payload or {})

    def _conditional_patch(
        self,
        point_id: str,
        raw_payload: Dict[str, Any],
        updates: Dict[str, Any],
        operation: str,
    ) -> None:
        """
        Optimistic concurrency control for read-modify-write updates.

        The write is guarded on the version read before computing ``updates``
        (or on the version field being absent for points created before
        versioning) and carries a fresh nonce.  Qdrant's ``set_payload`` does
        not report whether a filtered write matched, so the point is read back:
        the update counts as applied only if the stored version is exactly the
        expected version + 1 and the nonce is ours.  Anything else raises, so a
        concurrent writer can never cause a silently lost update.

        Local (in-process) stores evaluate payload filters by scanning every
        point, so for them the same check is done by re-reading the point's
        version under an in-process lock and writing by id: within one process
        this is equally atomic, and the conflict semantics are unchanged.
        """
        has_version = "version" in raw_payload
        expected = int(raw_payload.get("version", 0))
        nonce = uuid.uuid4().hex

        if self.backend != "remote":
            with self._write_lock:
                current = self._read_raw_payload(point_id)
                if current is None:
                    raise ValueError(f"Memory not found: {point_id}")
                current_version = int(current["version"]) if "version" in current else None
                if current_version != (expected if has_version else None):
                    raise RuntimeError(
                        f"Optimistic lock conflict for {operation} of {point_id} "
                        f"(expected version {expected if has_version else 'unset'}, "
                        f"found version {current.get('version')})"
                    )
                self.client.set_payload(
                    collection_name=self.collection_name,
                    payload={**updates, "version": expected + 1, "write_nonce": nonce},
                    points=[point_id],
                )
            return

        guard = (
            FieldCondition(key="version", match=MatchValue(value=expected))
            if has_version
            else IsEmptyCondition(is_empty=PayloadField(key="version"))
        )
        self.client.set_payload(
            collection_name=self.collection_name,
            payload={**updates, "version": expected + 1, "write_nonce": nonce},
            points=Filter(must=[HasIdCondition(has_id=[point_id]), guard]),
        )
        after = self._read_raw_payload(point_id)
        if after is None:
            raise ValueError(f"Memory not found: {point_id}")
        if int(after.get("version", -1)) != expected + 1 or after.get("write_nonce") != nonce:
            raise RuntimeError(
                f"Optimistic lock conflict for {operation} of {point_id} "
                f"(expected version {expected} -> {expected + 1}, "
                f"found version {after.get('version')})"
            )

    def update_recency(self, point_id: str, current_time: Optional[float] = None) -> float:
        """
        Touch a memory  –  set recency to the current (or virtual) time.

        Returns the new timestamp.

        Reference – DARS Specification §19:
            t_i ← current_time
        """
        now = time.time() if current_time is None else float(current_time)
        self.patch_payload(point_id, {"recency": now})
        return now

    def increment_frequency(self, point_id: str) -> int:
        """
        Increment the access counter by 1 (version-guarded).

        Reference – DARS Specification §19:
            a_i ← a_i + 1

        Returns the new frequency value.
        """
        raw = self._read_raw_payload(point_id)
        if raw is None:
            raise ValueError(f"Memory not found: {point_id}")

        old_freq = int(raw.get("frequency", 0))
        new_freq = old_freq + 1
        self._conditional_patch(point_id, raw, {"frequency": new_freq}, "incrementing frequency")

        logger.debug("Optimistic increment frequency for %s, %d -> %d", point_id, old_freq, new_freq)
        return new_freq

    def update_utility(self, point_id: str, success: bool) -> float:
        """
        Update utility after a success/failure signal from Layer B (version-guarded).

        Increments the appropriate counter and recomputes the Laplace-smoothed
            U = (success_count + 1) / (success_count + failure_count + 2)

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
        raw = self._read_raw_payload(point_id)
        if raw is None:
            raise ValueError(f"Memory not found: {point_id}")

        s = int(raw.get("success_count", 0))
        f = int(raw.get("failure_count", 0))
        if success:
            s += 1
        else:
            f += 1
        new_utility = self._compute_utility_score(s, f)

        updates = {
            "success_count": s,
            "failure_count": f,
            "utility": new_utility,
        }
        self._conditional_patch(point_id, raw, updates, "update_utility")

        logger.debug("Optimistic updated utility for %s: %s", point_id, updates)
        return new_utility

    def update_on_retrieval(
        self,
        point_id: str,
        success: bool,
        current_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Convenience method:  perform all Layer B updates in one guarded write.

        Updates recency, frequency, and utility together.

        Returns a dict of the new values.
        """
        raw = self._read_raw_payload(point_id)
        if raw is None:
            raise ValueError(f"Memory not found: {point_id}")

        now = time.time() if current_time is None else float(current_time)
        frequency = int(raw.get("frequency", 0)) + 1
        s = int(raw.get("success_count", 0))
        f = int(raw.get("failure_count", 0))
        if success:
            s += 1
        else:
            f += 1

        updates = {
            "recency": now,
            "frequency": frequency,
            "success_count": s,
            "failure_count": f,
            "utility": self._compute_utility_score(s, f),
        }
        self._conditional_patch(point_id, raw, updates, "update_on_retrieval")
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

    def compute_components(
        self,
        payload: Dict[str, Any],
        current_time: Optional[float] = None,
    ) -> Dict[str, float]:
        """Return the four DARS components ``{"R", "F", "U", "P"}`` for a payload."""
        return {
            "R": self._compute_recency(payload.get("recency", time.time()), current_time),
            "F": self._compute_frequency(payload.get("frequency", 0)),
            "U": self._compute_utility_score(
                payload.get("success_count", 0),
                payload.get("failure_count", 0),
            ),
            "P": float(payload.get("predictive", self.config.DEFAULT_PREDICTIVE_VALUE)),
        }

    def score_from_components(
        self,
        components: Dict[str, float],
        weights: Optional[DARSWeights] = None,
    ) -> float:
        """Weighted DARS score from precomputed components, clamped to [0, 1]."""
        w = weights or self.weights
        score = (
            w.w_r * components["R"]
            + w.w_f * components["F"]
            + w.w_u * components["U"]
            + w.w_p * components["P"]
        )
        return round(min(max(score, 0.0), 1.0), 6)

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
        return self.score_from_components(self.compute_components(payload, current_time))

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
        self, limit: int = 500, current_time: Optional[float] = None
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
        now = time.time() if current_time is None else float(current_time)
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
                "backend": self.backend,
                "collection_exists": exists,
                "collection_info": info,
                "total_collections": len(col_names),
            }
        except Exception as e:
            return {"connected": False, "backend": self.backend, "error": str(e)}
