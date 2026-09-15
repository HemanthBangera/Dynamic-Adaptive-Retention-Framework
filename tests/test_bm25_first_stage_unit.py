"""DARS over a BM25 first stage (local vault, no API)."""

import numpy as np
import pytest

from benchmarks.dars_eval.memory_units import MemoryUnit
from benchmarks.dars_eval.rankers import MemoryIndex, Method, bm25_tokens, rank

TEXTS = ["Alice moved to Paris in spring.", "Alice moved to Rome last year.", "Bob likes tennis and chess.",
         "Carol collects stamps.", "Alice visited Paris and Rome.", "Dan cooks pasta on Sundays."]
QUERY = "Where did Alice move?"


def _index(name, timestamps=None):
    ts = timestamps or [1_000.0] * len(TEXTS)
    units = [MemoryUnit(t, timestamp=s) for t, s in zip(TEXTS, ts)]
    return MemoryIndex(units, name, default_time=1_000.0)


def _bm25_order(idx):
    scores = np.asarray(idx.bm25.get_scores(bm25_tokens(QUERY)))
    return [idx._bm25_units[i] for i in np.argsort(-scores, kind="stable")]


def test_with_constant_lifecycle_signals_rrf_reproduces_bm25_order():
    idx = _index("bm25fs_const")
    m = Method("bm25_dars_rrf", rank_mode="rrf", fetch_k=4, weights=(1.0, 0.0, 0.0, 0.0), first_stage="bm25")
    ids, comps = rank(idx, m, QUERY, limit=len(TEXTS), current_time=1_000.0)
    assert ids == _bm25_order(idx)
    assert all("sim_rank" in c for c in comps[:4]) and all("sim_rank" not in c for c in comps[4:])


def test_recency_can_promote_a_newer_lexical_match():
    # the Rome memory is newer; with recency-only DARS weights it moves ahead of the older Paris memory
    idx = _index("bm25fs_recent", timestamps=[1_000.0, 90_000.0, 1_000.0, 1_000.0, 1_000.0, 1_000.0])
    base = _bm25_order(idx)
    m = Method("bm25_dars_blend", rank_mode="blend", fetch_k=6, alpha=0.2, weights=(1.0, 0.0, 0.0, 0.0),
               first_stage="bm25")
    ids, _ = rank(idx, m, QUERY, limit=6, current_time=100_000.0)
    assert ids.index(1) < ids.index(0) or base.index(1) < base.index(0)
    assert ids.index(1) <= base.index(1)


def test_unknown_first_stage_is_rejected():
    idx = _index("bm25fs_bad")
    with pytest.raises(ValueError):
        rank(idx, Method("x", rank_mode="rrf", first_stage="dense"), QUERY, limit=3, current_time=1_000.0)


def test_default_first_stage_is_unchanged_vector_search():
    idx = _index("bm25fs_default")
    a, _ = rank(idx, Method("dars_rrf_k15", rank_mode="rrf", fetch_k=15), QUERY, limit=6, current_time=1_000.0)
    b, _ = rank(idx, Method("dars_rrf_k15", rank_mode="rrf", fetch_k=15, first_stage="vector"), QUERY, limit=6,
                current_time=1_000.0)
    assert a == b
