"""run_stream.evict: DARS eviction with explicit weights, and the baseline policies (local vault)."""

import numpy as np

from benchmarks.dars_eval.memory_units import MemoryUnit
from benchmarks.dars_eval.rankers import MemoryIndex
from benchmarks.dars_eval.run_stream import evict

TEXTS = ["Alpha is the first fact.", "Beta is the second fact.", "Gamma is the third fact.",
         "Delta is the fourth fact."]
NOW = 50_000.0


def _index(name):
    units = [MemoryUnit(t, timestamp=1_000.0 + 3_600.0 * i) for i, t in enumerate(TEXTS)]
    return MemoryIndex(units, name)


def test_utility_only_eviction_removes_the_failed_memory():
    idx = _index("evict_utility")
    idx.vault.update_utility(idx.point_ids[0], success=True)      # U = 2/3
    idx.vault.update_utility(idx.point_ids[1], success=False)     # U = 1/3; the rest keep the 1/2 prior
    gone = evict(idx, 3, "dars", NOW, np.random.default_rng(0), weights=(0.0, 0.0, 1.0, 0.0))
    assert gone == [1]
    assert len(idx) == 3 and 1 not in idx.point_ids and 1 not in idx.ingested


def test_lru_eviction_is_unchanged_and_removes_the_oldest():
    idx = _index("evict_lru")
    gone = evict(idx, 2, "lru", NOW, np.random.default_rng(0))
    assert gone == [0, 1] and sorted(idx.ingested) == [2, 3]


def test_no_eviction_below_capacity():
    idx = _index("evict_none")
    assert evict(idx, 10, "dars", NOW, np.random.default_rng(0), weights=(1.0, 0.0, 0.0, 0.0)) == []
