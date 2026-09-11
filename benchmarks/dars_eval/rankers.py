"""
Retrieval methods compared in the revision, all over the same memory units.

``MemoryIndex`` stores units in a local ``MemoryVault`` (the system's own
storage and ranking code) — all at once (static protocols) or incrementally via
``add`` (multi-session streams) — plus a BM25 index over the stored units.
``rank`` returns unit indices in rank order for one method:

dars      ``MemoryVault.search_and_rerank`` with the method's rank mode
          (similarity | rrf | wrrf | blend) and weight vector.  Two-stage: the
          ``fetch_k`` nearest memories are reranked; when more slots are needed
          (large token budgets) they follow similarity order.
bm25      Okapi BM25 over lower-cased word tokens (rank_bm25).
recency   most recently stored-or-accessed memories first (LRU order; query-agnostic).
random    a seeded random permutation per query.

``rerank_candidates`` recomputes the fused order of already-scored candidates
under different weights (used for per-component rank-influence analysis); it
mirrors ``MemoryVault.search_and_rerank`` exactly.
"""

from __future__ import annotations

import re
import zlib
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from benchmarks.dars_eval.memory_units import MemoryUnit
from core.layer_d.schema import DARSWeights
from core.layer_d.storage import MemoryVault

TOKEN_RE = re.compile(r"[a-z0-9]+")
DEFAULT_WEIGHTS = (0.30, 0.20, 0.30, 0.20)
COMPONENTS = ("R", "F", "U", "P")


def bm25_tokens(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class Method:
    """One retrieval configuration (hashable, recorded verbatim in manifests)."""

    name: str
    kind: str = "dars"                    # dars | bm25 | recency | random
    rank_mode: str = "similarity"         # dars only: similarity | rrf | wrrf | blend
    weights: Tuple[float, float, float, float] = DEFAULT_WEIGHTS   # (w_r, w_f, w_u, w_p)
    fetch_k: int = 50
    rrf_k: int = 60
    beta_sim: float = 1.0
    beta_dars: float = 1.0
    alpha: float = 0.5
    seed: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MemoryIndex:
    """Memory units stored in a fresh local vault (all at once, or incrementally)."""

    def __init__(
        self,
        units: Sequence[MemoryUnit],
        name: str,
        *,
        vectors: Optional[np.ndarray] = None,
        predictive: Optional[Sequence[float]] = None,
        default_time: Optional[float] = None,
        embed_batch: int = 64,
        ingest_all: bool = True,
    ):
        self.units = list(units)
        self.vault = MemoryVault(collection_name=name, location=":memory:")
        self.vault.initialize_collection(recreate=True)
        if vectors is None:
            vectors = np.asarray(
                self.vault.embedder.encode_batch([u.text for u in self.units], batch_size=embed_batch),
                dtype=np.float32,
            )
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.predictive = None if predictive is None else [float(p) for p in predictive]
        self.default_time = default_time
        self.point_ids: Dict[int, str] = {}
        self.unit_of: Dict[str, int] = {}
        self.ingested: List[int] = []
        self._bm25 = None
        self._bm25_units: List[int] = []
        if ingest_all:
            self.add(range(len(self.units)))

    def add(self, indices: Iterable[int], timestamps: Optional[Sequence[float]] = None) -> List[str]:
        """Store units (by index) with their own timestamps, or ``timestamps`` if given."""
        indices = [int(i) for i in indices if int(i) not in self.point_ids]
        if not indices:
            return []
        records = []
        for j, i in enumerate(indices):
            unit = self.units[i]
            ts = timestamps[j] if timestamps is not None else unit.timestamp
            rec: Dict[str, Any] = {
                "text": unit.text,
                "vector": self.vectors[i].tolist(),
                "timestamp": ts if ts is not None else self.default_time,
                "extra": {"unit_index": i},
            }
            if self.predictive is not None:
                rec["predictive_value"] = self.predictive[i]
            records.append(rec)
        pids = self.vault.store_memories_batch(records, current_time=self.default_time)
        for i, pid in zip(indices, pids):
            self.point_ids[i] = pid
            self.unit_of[pid] = i
        self.ingested.extend(indices)
        self._bm25 = None
        return pids

    def __len__(self) -> int:
        return len(self.ingested)

    @property
    def bm25(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi

            self._bm25_units = list(self.ingested)
            self._bm25 = BM25Okapi([bm25_tokens(self.units[i].text) or ["_"] for i in self._bm25_units])
        return self._bm25

    def payload_recency(self) -> Dict[int, float]:
        """Current ``recency`` (last store or access time) of every stored unit."""
        out: Dict[int, float] = {}
        for chunk, _ in self.vault.get_all_memories(limit=1024, scroll_yield=True):
            for p in chunk:
                out[self.unit_of[p.point_id]] = float(p.payload.recency)
        return out


def _query_seed(seed: int, query_text: str) -> int:
    return (int(seed) * 1_000_003 + zlib.crc32(query_text.encode("utf-8"))) % (2 ** 32)


def rank(
    index: MemoryIndex,
    method: Method,
    query_text: str,
    query_vector: Optional[Sequence[float]] = None,
    limit: int = 100,
    current_time: Optional[float] = None,
) -> Tuple[List[int], List[Optional[Dict[str, float]]]]:
    """Unit indices in rank order (at most ``limit``) and per-unit score components."""
    n = len(index)
    if n == 0:
        return [], []
    limit = max(1, min(int(limit), n))

    if method.kind == "bm25":
        scores = np.asarray(index.bm25.get_scores(bm25_tokens(query_text)))
        order = np.argsort(-scores, kind="stable")[:limit]
        return [index._bm25_units[i] for i in order], [None] * len(order)

    if method.kind == "random":
        rng = np.random.default_rng(_query_seed(method.seed, query_text))
        order = rng.permutation(n)[:limit]
        return [index.ingested[i] for i in order], [None] * len(order)

    if method.kind == "recency":
        rec = index.payload_recency()
        units = sorted(index.ingested, key=lambda u: (-rec[u], u))
        return units[:limit], [None] * min(limit, len(units))

    if method.kind != "dars":
        raise ValueError(f"Unknown method kind {method.kind!r}")

    vault = index.vault
    vault.weights = DARSWeights(*method.weights)
    if not vault.weights.validate():
        raise ValueError(f"Weights of {method.name} do not sum to 1: {method.weights}")
    qv = list(query_vector) if query_vector is not None else vault.embedder.encode(query_text)
    fetch_k = max(1, min(int(method.fetch_k), n))
    fused = vault.search_and_rerank(
        query_text,
        fetch_k=fetch_k,
        top_n=fetch_k,
        rank_mode=method.rank_mode,
        rrf_k=method.rrf_k,
        beta_sim=method.beta_sim,
        beta_dars=method.beta_dars,
        alpha=method.alpha,
        current_time=current_time,
        query_vector=qv,
        return_components=True,
    )
    ids = [index.unit_of[m.point_id] for m in fused]
    comps: List[Optional[Dict[str, float]]] = [m.components for m in fused]
    if limit > len(ids):
        seen = set(ids)
        for m in vault.semantic_search(query_text, top_k=limit, query_vector=qv):
            u = index.unit_of[m.point_id]
            if u not in seen:
                ids.append(u)
                comps.append({"sim": float(m.score)} if m.score is not None else None)
                seen.add(u)
    return ids[:limit], comps[:limit]


def rerank_candidates(
    components: Sequence[Dict[str, float]],
    method: Method,
    weights: Optional[Tuple[float, float, float, float]] = None,
) -> List[int]:
    """
    Order (indices into ``components``) of already-scored fetch_k candidates under
    ``weights`` (default: the method's), replicating ``MemoryVault.search_and_rerank``:
    6-decimal rounding of S, stable sorts in similarity-rank order, the same fusion.
    ``components`` must be the candidates' R, F, U, P, sim and sim_rank entries.
    """
    w = weights or method.weights
    n = len(components)
    by_sim = sorted(range(n), key=lambda i: components[i]["sim_rank"])
    sims = [components[i]["sim"] for i in range(n)]
    dars = [
        round(min(max(sum(wi * components[i][c] for wi, c in zip(w, COMPONENTS)), 0.0), 1.0), 6)
        for i in range(n)
    ]
    dars_order = sorted(by_sim, key=lambda i: -dars[i])
    dars_rank = {i: r for r, i in enumerate(dars_order, 1)}
    sim_rank = {i: r for r, i in enumerate(by_sim, 1)}

    if method.rank_mode == "similarity":
        score = {i: sims[i] for i in range(n)}
    elif method.rank_mode in ("rrf", "wrrf"):
        b_s, b_d = (1.0, 1.0) if method.rank_mode == "rrf" else (method.beta_sim, method.beta_dars)
        score = {i: b_s / (method.rrf_k + sim_rank[i]) + b_d / (method.rrf_k + dars_rank[i]) for i in range(n)}
    elif method.rank_mode == "blend":
        lo, hi = min(sims), max(sims)
        if hi - lo < 0.05:
            score = {i: method.alpha + (1 - method.alpha) * dars[i] for i in range(n)}
        else:
            score = {i: method.alpha * (sims[i] - lo) / (hi - lo) + (1 - method.alpha) * dars[i] for i in range(n)}
    else:
        raise ValueError(f"Unknown rank_mode {method.rank_mode!r}")
    return sorted(by_sim, key=lambda i: -score[i])


def leave_one_out_weights(weights: Tuple[float, float, float, float]) -> Dict[str, Tuple[float, ...]]:
    """Weight vectors with one component removed and the rest renormalised."""
    out = {}
    for j, c in enumerate(COMPONENTS):
        rest = sum(w for i, w in enumerate(weights) if i != j)
        if rest <= 0:
            continue
        out[c] = tuple(0.0 if i == j else w / rest for i, w in enumerate(weights))
    return out
