"""
A deeper ``rank`` call must return the same prefix for every method kind.

E2 computes rank precedence (H2) from the full ranking and derives the memories
shown to the reader from its budget-cut prefix, so the shown set must not depend on
how deep the ranking was requested.  Uses the local vault and all-MiniLM-L6-v2.
"""

import pytest

from benchmarks.dars_eval.memory_units import MemoryUnit
from benchmarks.dars_eval.rankers import MemoryIndex, Method, rank

TEXTS = [
    "The Normans conquered England in 1066.",
    "Python 3.12 is the project's runtime.",
    "The client prefers Python for data pipelines.",
    "Rollo was the first ruler of Normandy.",
    "The deployment uses Qdrant as the vector store.",
    "Normandy is a region in northern France.",
    "The team migrated the pipeline from Java to Python.",
    "Unit tests run on every commit.",
]
NOW = 100_000.0

METHODS = [
    Method("similarity", rank_mode="similarity", fetch_k=3),
    Method("rrf", rank_mode="rrf", fetch_k=3),
    Method("wrrf", rank_mode="wrrf", fetch_k=3, beta_dars=0.5),
    Method("blend", rank_mode="blend", fetch_k=3, alpha=0.5),
    Method("bm25", kind="bm25"),
    Method("recency", kind="recency"),
    Method("random", kind="random", seed=3),
]


@pytest.fixture(scope="module")
def index():
    units = [MemoryUnit(t, timestamp=1_000.0 + 3_600.0 * i) for i, t in enumerate(TEXTS)]
    idx = MemoryIndex(units, "rank_prefix")
    idx.vault.update_utility(idx.point_ids[6], success=True)
    idx.vault.update_utility(idx.point_ids[1], success=False)
    idx.vault.increment_frequency(idx.point_ids[2])
    return idx


@pytest.mark.parametrize("method", METHODS, ids=lambda m: m.name)
def test_shallow_ranking_is_prefix_of_full_ranking(index, method):
    query = "Which project uses Python?"
    full, _ = rank(index, method, query, limit=len(index), current_time=NOW)
    assert sorted(full) == list(range(len(TEXTS)))
    for limit in (1, 4, 6):           # below, at and above fetch_k = 3
        short, _ = rank(index, method, query, limit=limit, current_time=NOW)
        assert short == full[:limit]
