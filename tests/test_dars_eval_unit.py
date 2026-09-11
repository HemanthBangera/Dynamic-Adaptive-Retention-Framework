"""Unit tests for the revision evaluation harness (stats, memory units, labels, metrics, rankers)."""

import math

import numpy as np
import pytest

from benchmarks.dars_eval import stats
from benchmarks.dars_eval.labels import fc_gold_fact, fc_labels, normalize_answer, ruler_evidence
from benchmarks.dars_eval.memory_units import (
    MemoryUnit,
    WindowGuard,
    fact_units,
    parse_chat_time,
    parse_facts,
    parse_lme_sessions,
    parse_ruler_documents,
)
from benchmarks.dars_eval.retrieval_eval import cut_to_budget, evidence_metrics, precedence
from benchmarks.dars_eval.rankers import MemoryIndex, Method, rank
from benchmarks.dars_eval.reader import MABReader, build_prompt
from benchmarks.dars_eval.splits import question_splits
from benchmarks.dars_eval.datasets import retrieval_query


# ═══════════════════════════════════════════════════════════════════════════════
#  Statistics
# ═══════════════════════════════════════════════════════════════════════════════

class TestStats:

    def test_cohen_kappa_known_values(self):
        assert stats.cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
        # po = 0.5, pe = 0.5 → kappa 0
        assert stats.cohen_kappa([1, 0, 1, 0], [1, 1, 0, 0]) == pytest.approx(0.0)

    def test_holm_matches_hand_computation(self):
        adj = stats.holm([0.01, 0.04, 0.03])
        assert adj == pytest.approx([0.03, 0.06, 0.06])

    def test_mcnemar_exact(self):
        a = [True] * 8 + [False] * 2
        b = [False] * 8 + [True] * 2
        res = stats.mcnemar_exact(a, b)
        assert res["a_only"] == 8 and res["b_only"] == 2
        assert res["p_value"] == pytest.approx(2 * sum(math.comb(10, i) for i in range(3)) / 2 ** 10)

    def test_auroc_matches_rank_formula(self):
        rng = np.random.default_rng(0)
        labels = rng.random(200) < 0.4
        scores = rng.normal(size=200) + labels * 1.0
        pos, neg = scores[labels], scores[~labels]
        manual = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
        res = stats.auroc_delong(scores, labels)
        assert res["auc"] == pytest.approx(manual)
        assert res["ci_lo"] < res["auc"] < res["ci_hi"]

    def test_paired_delong_identical_scores(self):
        labels = [1, 0, 1, 0, 1, 1, 0, 0]
        s = [0.9, 0.1, 0.8, 0.3, 0.7, 0.6, 0.2, 0.4]
        res = stats.auroc_delong_paired(s, s, labels)
        assert res["diff"] == 0.0 and res["p_value"] == 1.0

    def test_cluster_bootstrap_ci_contains_mean(self):
        rng = np.random.default_rng(1)
        values = rng.random(300)
        clusters = np.repeat(np.arange(5), 60)
        est = stats.cluster_bootstrap_mean(values, clusters, n_boot=2000, seed=3)
        assert est.lo <= est.mean <= est.hi and est.n == 300

    def test_paired_bootstrap_detects_clear_difference(self):
        a = np.ones(100)
        b = np.zeros(100)
        res = stats.paired_bootstrap_diff(a, b, n_boot=500)
        assert res["diff"] == 1.0 and res["p_value"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Parsing, window guard, labels
# ═══════════════════════════════════════════════════════════════════════════════

class TestParsingAndLabels:

    def test_parse_chat_time(self):
        t = parse_chat_time("Chat Time: 2023/05/21 (Sun) 16:10")
        assert (t.year, t.month, t.day, t.hour, t.minute) == (2023, 5, 21, 16, 10)

    def test_parse_lme_sessions(self):
        ctx = repr(["Chat Time: 2023/01/01 (Sun) 10:00", [{"role": "user", "content": "hi"}],
                    "Chat Time: 2023/01/02 (Mon) 11:30", [{"role": "assistant", "content": "yo"}]])
        sessions = parse_lme_sessions(ctx)
        assert len(sessions) == 2 and sessions[1][1][0]["content"] == "yo"

    def test_parse_ruler_documents(self):
        docs = parse_ruler_documents("Document 1:\nAlpha text.\n\nDocument 2:\nBeta text.")
        assert docs == {1: "Alpha text.", 2: "Beta text."}

    def test_fact_units_keep_serial_and_order_time(self):
        units = fact_units("Here is a list of facts:\n0. A is B.\n1. A is C.", t0=100.0, seconds_per_serial=10.0)
        assert [u.text for u in units] == ["0. A is B.", "1. A is C."]
        assert [u.timestamp for u in units] == [100.0, 110.0]

    def test_fc_gold_is_newest_matching_subject(self):
        facts = parse_facts(
            "1. goaltender is associated with the sport of ice hockey.\n"
            "2. pesäpallo was created in the country of Finland.\n"
            "3. goaltender is associated with the sport of pesäpallo.\n"
        )
        lab = fc_gold_fact(facts, "Which sport is goaltender associated with?", ["pesäpallo"], multi_hop=False)
        assert lab["gold_serial"] == 3 and lab["conflict_serials"] == [1] and lab["gold_is_newest"]

    def test_fc_answer_ending_with_period(self):
        facts = parse_facts("7. Pedro Pierluisi worked in the city of Washington, D.C..")
        lab = fc_gold_fact(facts, "Which city did Pedro Pierluisi work in?", ["Washington, D.C."], multi_hop=False)
        assert lab is not None and lab["gold_serial"] == 7

    def test_fc_labels_unlabelled_question(self):
        labs = fc_labels("1. X is Y.", ["Who is Z?"], [["nobody"]], multi_hop=False)
        assert labs == [None]

    def test_ruler_evidence_prefers_answer_bearing_units(self):
        units = [
            MemoryUnit("Document 5: Normandy is a region.", meta={"doc_id": 5}),
            MemoryUnit("Document 5: It is located in France.", meta={"doc_id": 5}),
            MemoryUnit("Document 6: Unrelated.", meta={"doc_id": 6}),
        ]
        ev = ruler_evidence(units, {0: [5]}, [["France"]], n_questions=1)
        assert ev == [[[1]]]
        assert normalize_answer("The France!") == "france"

    def test_window_guard_splits_long_text(self):
        guard = WindowGuard()
        long_text = " ".join(["memory"] * 1000) + ". " + "Short end."
        parts = guard.split(long_text)
        assert len(parts) > 1 and all(guard.fits(p) for p in parts)
        prefixed = guard.split_with_prefix("[2023/01/01 10:00] User:", long_text)
        assert all(p.startswith("[2023/01/01 10:00] User:") and guard.fits(p) for p in prefixed)


# ═══════════════════════════════════════════════════════════════════════════════
#  Retrieval metrics and rankers
# ═══════════════════════════════════════════════════════════════════════════════

class TestRetrievalMetrics:

    def test_cut_to_budget(self):
        assert cut_to_budget([0, 1, 2], [10, 10, 10], budget=31, overhead=5) == [0, 1]

    def test_evidence_metrics(self):
        m = evidence_metrics([7, 3, 9], [[3], [4, 9]])
        assert m["group_recall"] == 1.0 and m["all_covered"] == 1.0 and m["mrr"] == 0.5
        assert evidence_metrics([1], []) is None

    def test_precedence(self):
        assert precedence([5, 2], gold=5, older_versions=[2]) == 1.0
        assert precedence([2, 5], gold=5, older_versions=[2]) == 0.0
        assert precedence([2], gold=5, older_versions=[2]) == 0.0
        assert precedence([5], gold=5, older_versions=[]) is None


class TestRankers:

    TEXTS = [
        "The Normans settled in Normandy in France.",
        "Rollo was the Norse leader of the Normans.",
        "Photosynthesis converts light into chemical energy.",
        "The Duchy of Normandy was founded in 911.",
    ]

    def _index(self, name):
        units = [MemoryUnit(t, timestamp=1000.0 + i) for i, t in enumerate(self.TEXTS)]
        return MemoryIndex(units, name)

    def test_similarity_bm25_recency_random(self):
        idx = self._index("rank_basic")
        query = "Who led the Normans?"
        expected = [idx.unit_of[m.point_id] for m in idx.vault.semantic_search(query, top_k=4)]
        sim, comps = rank(idx, Method("sim"), query, limit=4)
        assert sim == expected and comps[0]["sim_rank"] == 1.0
        bm25, _ = rank(idx, Method("bm25", kind="bm25"), "Norse leader", limit=2)
        assert bm25[0] == 1
        rec, _ = rank(idx, Method("recency", kind="recency"), "anything", limit=4)
        assert rec == [3, 2, 1, 0]
        r1, _ = rank(idx, Method("rand", kind="random", seed=7), "q", limit=4)
        r2, _ = rank(idx, Method("rand", kind="random", seed=7), "q", limit=4)
        assert r1 == r2 and sorted(r1) == [0, 1, 2, 3]

    def test_two_stage_extends_in_similarity_order(self):
        idx = self._index("rank_two_stage")
        sim, _ = rank(idx, Method("sim", fetch_k=4), "Normandy France", limit=4)
        short, _ = rank(idx, Method("sim-k2", fetch_k=2), "Normandy France", limit=4)
        assert short == sim

    def test_dars_weights_are_applied_per_method(self):
        idx = self._index("rank_weights")
        rank(idx, Method("recency-only-dars", rank_mode="rrf", weights=(1.0, 0.0, 0.0, 0.0)), "q", limit=2)
        assert idx.vault.weights.w_r == 1.0
        with pytest.raises(ValueError, match="do not sum to 1"):
            rank(idx, Method("bad", weights=(0.5, 0.5, 0.5, 0.5)), "q", limit=2)


# ═══════════════════════════════════════════════════════════════════════════════
#  Splits, retrieval queries and the reader
# ═══════════════════════════════════════════════════════════════════════════════

class TestSplitsQueriesReader:

    def test_question_splits_deterministic_30_percent(self):
        a = question_splits("ruler_qa1_197K", 0, 100)
        b = question_splits("ruler_qa1_197K", 0, 100)
        assert a == b and a.count("dev") == 30 and a.count("test") == 70
        assert question_splits("ruler_qa1_197K", 1, 100) != a

    def test_eventqa_query_drops_boilerplate_and_fits_window(self):
        guard = WindowGuard()
        events = "\n".join(f"{i}. Event number {i} happened in the long story." for i in range(1, 80))
        q = (
            "These are the events that have already occurred:\n\n" + events +
            "\n\n\nBelow is a list of possible subsequent events:\n\n['Option A happens.', 'Option B happens.']"
            "\n\n Your task is to choose from the above events which event happens next based on the book excerpt."
            " In your response to me, only include the answer without anything else."
        )
        out = retrieval_query("eventqa_65536", q, guard)
        assert guard.fits(out)
        assert "Option B happens." in out and "Your task is to choose" not in out

    def test_short_query_unchanged(self):
        guard = WindowGuard()
        assert retrieval_query("ruler_qa1_197K", "In what country is Normandy located?", guard) == \
            "In what country is Normandy located?"

    def test_prompt_format_matches_mab_rag(self):
        assert build_prompt(["a", "b"], "Q?") == "Memory 1:\na\nMemory 2:\nb\nQ?"
        assert build_prompt([], "Q?") == "Q?"

    @pytest.mark.asyncio
    async def test_reader_scores_with_mab_metrics(self):
        class FakeTransport:
            model = "fake"
            temperature = 0.0

            async def complete(self, prompt, system=None, seed=None, max_tokens=None):
                assert prompt.startswith("Memory 1:") and max_tokens == 50
                return {"text": "France", "usage": {"prompt_tokens": 5, "completion_tokens": 1},
                        "cached": False, "cache_key": "k", "finish_reason": "stop"}

        reader = MABReader(FakeTransport(), "ruler_qa1_197K", "Accurate_Retrieval")
        res = await reader.answer(["Normandy is in France."], "Question: where?", ["France"])
        assert res["metrics"]["substring_exact_match"] == 1.0 and res["metrics"]["exact_match"] == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Incremental index and offline re-ranking
# ═══════════════════════════════════════════════════════════════════════════════

class TestIncrementalAndRerank:

    TEXTS = [
        "Alpha project uses Python for data pipelines.",
        "Beta project deadline is in March.",
        "Gamma team prefers PostgreSQL databases.",
        "Python notebooks are used for analysis in Alpha.",
        "The cafeteria serves lunch at noon.",
        "Delta project migrated to Rust last year.",
    ]

    def test_incremental_add_only_ranks_stored_units(self):
        from benchmarks.dars_eval.rankers import MemoryIndex, Method, rank
        units = [MemoryUnit(t, timestamp=100.0 * i) for i, t in enumerate(self.TEXTS)]
        idx = MemoryIndex(units, "incr_add", ingest_all=False)
        assert len(idx) == 0 and rank(idx, Method("sim"), "Python", limit=3) == ([], [])
        idx.add([0, 1, 2])
        got, _ = rank(idx, Method("sim"), "Python", limit=6)
        assert set(got) == {0, 1, 2}
        bm, _ = rank(idx, Method("bm25", kind="bm25"), "Python", limit=6)
        assert set(bm) == {0, 1, 2} and bm[0] == 0
        idx.add([3])
        bm2, _ = rank(idx, Method("bm25", kind="bm25"), "Python notebooks", limit=6)
        assert bm2[0] == 3

    def test_recency_ranks_by_last_access(self):
        from benchmarks.dars_eval.rankers import MemoryIndex, Method, rank
        units = [MemoryUnit(t, timestamp=100.0 * i) for i, t in enumerate(self.TEXTS[:3])]
        idx = MemoryIndex(units, "incr_recency")
        assert rank(idx, Method("rec", kind="recency"), "q", limit=3)[0] == [2, 1, 0]
        idx.vault.update_recency(idx.point_ids[0], current_time=1_000.0)
        assert rank(idx, Method("rec", kind="recency"), "q", limit=3)[0] == [0, 2, 1]

    @pytest.mark.parametrize("mode", ["similarity", "rrf", "wrrf", "blend"])
    def test_rerank_candidates_reproduces_vault_order(self, mode):
        from benchmarks.dars_eval.rankers import MemoryIndex, Method, rank, rerank_candidates
        units = [MemoryUnit(t, timestamp=1_000.0 + 3_600.0 * i) for i, t in enumerate(self.TEXTS)]
        idx = MemoryIndex(units, f"rerank_eq_{mode}")
        idx.vault.update_utility(idx.point_ids[4], success=True)
        idx.vault.update_utility(idx.point_ids[1], success=False)
        idx.vault.increment_frequency(idx.point_ids[5])
        method = Method("m", rank_mode=mode, fetch_k=6, beta_dars=0.5, alpha=0.5)
        ranked, comps = rank(idx, method, "Which project uses Python?", limit=6, current_time=50_000.0)
        order = rerank_candidates(comps, method)
        assert [ranked[i] for i in order] == ranked


# ═══════════════════════════════════════════════════════════════════════════════
#  E3 tuning machinery and offline fusion
# ═══════════════════════════════════════════════════════════════════════════════

class TestTuning:

    def test_simplex_grid(self):
        from benchmarks.dars_eval.tuning import simplex_grid
        grid = simplex_grid(0.1)
        assert len(grid) == 286
        assert all(abs(sum(w) - 1.0) < 1e-9 for w in grid)
        assert (0.3, 0.2, 0.3, 0.2) in [tuple(round(x, 10) for x in w) for w in grid]

    def test_auc_rows_matches_delong_auc(self):
        from benchmarks.dars_eval.tuning import auc_rows
        rng = np.random.default_rng(5)
        y = rng.random(120) < 0.3
        s = rng.normal(size=(3, 120)) + y * np.array([[0.0], [1.0], [2.0]])
        rows = auc_rows(s, y)
        for i in range(3):
            assert rows[i] == pytest.approx(stats.auroc_delong(s[i], y)["auc"])

    def test_time_to_threshold_closed_form(self):
        from benchmarks.dars_eval.tuning import time_to_threshold
        w = (0.3, 0.2, 0.3, 0.2)
        t = time_to_threshold(0.3, w, P=0.05, lam=0.005)
        rest = 0.3 * 0.5 + 0.2 * 0.05
        assert 0.3 * math.exp(-0.005 * t) + rest == pytest.approx(0.3)
        assert time_to_threshold(0.7, w, P=0.05, lam=0.005) == 0.0        # fresh neutral memory never retains
        assert time_to_threshold(0.1, w, P=0.05, lam=0.005) == math.inf    # floor 0.16 > 0.1

    def test_dirichlet_around_mean(self):
        from benchmarks.dars_eval.tuning import dirichlet_around
        d = dirichlet_around((0.3, 0.2, 0.3, 0.2), cv=0.2, draws=4000, seed=1)
        assert np.allclose(d.mean(axis=0), [0.3, 0.2, 0.3, 0.2], atol=0.01)
        assert np.allclose(d.sum(axis=1), 1.0)

    @pytest.mark.parametrize("mode,param", [("rrf", 0.0), ("wrrf", 0.5), ("blend", 0.5)])
    def test_fused_order_matches_rerank_candidates(self, mode, param):
        from benchmarks.dars_eval.rankers import Method, rerank_candidates
        from benchmarks.dars_eval.run_alfworld import _fused_order
        rng = np.random.default_rng(11)
        comps = []
        for i in range(8):
            comps.append({"R": float(rng.random()), "F": float(rng.random()), "U": float(rng.random()),
                          "P": float(rng.random()), "sim": float(rng.random())})
        order_sim = sorted(range(8), key=lambda i: -comps[i]["sim"])
        for r, i in enumerate(order_sim, 1):
            comps[i]["sim_rank"] = float(r)
        w = (0.3, 0.2, 0.3, 0.2)
        method = Method("m", rank_mode=mode, beta_dars=param if mode == "wrrf" else 1.0,
                        alpha=param if mode == "blend" else 0.5)
        expected = rerank_candidates(comps, method, weights=w)
        sim = np.array([c["sim"] for c in comps])
        S = np.array([sum(wi * c[k] for wi, k in zip(w, ("R", "F", "U", "P"))) for c in comps])
        assert list(_fused_order(sim, S, mode, param)) == expected
