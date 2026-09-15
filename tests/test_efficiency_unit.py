"""Unit tests for the E8 helpers: lazy budget fill and the disk compression cache."""

from benchmarks.dars_eval.retrieval_eval import cut_to_budget
from benchmarks.dars_eval.run_efficiency import ENC, CompressionCache, fill_budget

TEXTS = [
    "Rollo was the first ruler of Normandy.",
    "The Normans conquered England in 1066 after the Battle of Hastings.",
    "Normandy is a region in northern France with a long coastline.",
    "Python 3.12 is the project's runtime.",
]


def test_fill_budget_matches_cut_to_budget():
    ranked = [2, 0, 3, 1]
    unit_tokens = [len(ENC.encode(t)) for t in TEXTS]
    for budget in (5, 20, 40, 60, 1000):
        shown, used = fill_budget(ranked, lambda u: TEXTS[u], budget)
        assert shown == cut_to_budget(ranked, unit_tokens, budget)
        assert used <= budget


def test_compressed_texts_fit_more_memories():
    ranked = [0, 1, 2, 3]
    full, _ = fill_budget(ranked, lambda u: TEXTS[u], 40)
    short, _ = fill_budget(ranked, lambda u: TEXTS[u].split()[0], 40)
    assert len(short) > len(full)


def test_compression_cache_persists_and_avoids_recomputation(tmp_path):
    calls = []

    def fn(text):
        calls.append(text)
        return text.upper()

    cache = CompressionCache("test_upper", fn, root=tmp_path)
    assert cache("abc") == "ABC" and cache("abc") == "ABC"
    assert calls == ["abc"] and cache.misses == 1
    reopened = CompressionCache("test_upper", fn, root=tmp_path)
    assert reopened("abc") == "ABC" and calls == ["abc"] and reopened.misses == 0
