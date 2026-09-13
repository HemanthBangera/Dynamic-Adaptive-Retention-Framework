"""Write-side components and prior-art retention baselines, checked on hand-made rows."""

import math

import pytest

from benchmarks.dars_eval.retention_signals import (
    dars_read_score, dars_write_score, generative_agents_score, memorybank_score, metadata_score,
    parse_rating, restated_sessions, write_component_matrix, write_components,
)

T_END = 1_700_000_000.0 + 2 * 86400 + 60 * 200


def row(**kw):
    base = {"dialogue": 1, "text": "i like tea", "created_session": 0, "mentions": 1, "mention_sessions": [0],
            "last_write_time": 1_700_000_000.0, "opportunities": 2, "t_end": T_END,
            "recency": 1_700_000_000.0, "frequency": 0,
            "components": {"R": 0.5, "F": 0.0, "U": 0.5, "P": 0.3}}
    base.update(kw)
    return base


def test_restatements_count_distinct_later_sessions_only():
    r = row(mention_sessions=[0, 0, 1, 1, 1, 2], mentions=6)
    assert restated_sessions(r) == 2


def test_write_components_follow_their_definitions():
    r = row(mention_sessions=[0, 1], mentions=2, last_write_time=T_END - 7200.0)
    c = write_components(r, lam=0.05)
    assert c["R"] == pytest.approx(math.exp(-0.05 * 2))
    assert c["F"] == pytest.approx(math.log(3) / math.log(51))
    assert c["U"] == pytest.approx((1 + 1) / (2 + 2))
    assert c["P"] == 0.3


def test_a_never_restated_fact_has_prior_utility_that_falls_with_opportunities():
    fresh = write_components(row(created_session=2, opportunities=0), lam=0.0)
    old = write_components(row(created_session=0, opportunities=2), lam=0.0)
    assert fresh["U"] == 0.5 and old["U"] == pytest.approx(0.25)


def test_rows_without_write_history_are_rejected():
    r = row()
    del r["mention_sessions"]
    with pytest.raises(KeyError):
        write_components(r, lam=0.01)


def test_matrix_matches_the_per_row_components():
    rows = [row(), row(mentions=3, mention_sessions=[0, 1, 2], last_write_time=T_END - 3600)]
    m = write_component_matrix(rows, lam=0.01)
    assert m.shape == (4, 2)
    assert m[2, 1] == pytest.approx(write_components(rows[1], 0.01)["U"])


def test_read_score_recomputes_recency_from_last_access():
    r = row(recency=T_END - 3600.0)
    s = dars_read_score([r], lam=0.1, weights=(1, 0, 0, 0))[0]
    assert s == pytest.approx(round(math.exp(-0.1), 6))


def test_write_score_uses_the_vault_clipping_and_rounding():
    s = dars_write_score([row()], lam=0.0, weights=(0.5, 0.0, 0.5, 0.0))[0]
    assert s == round(0.5 * 1.0 + 0.5 * 0.25, 6)


def test_generative_agents_normalises_within_each_dialogue():
    rows = [row(dialogue=1, text="a", recency=T_END), row(dialogue=1, text="b", recency=T_END - 36000),
            row(dialogue=2, text="c", recency=T_END), row(dialogue=2, text="d", recency=T_END)]
    imp = {"a": 2, "b": 9, "c": 5, "d": 5}
    s = generative_agents_score(rows, imp)
    assert s[0] == pytest.approx(1.0) and s[1] == pytest.approx(1.0)     # recent-but-mundane vs old-but-poignant
    assert s[2] == 0.0 and s[3] == 0.0                                   # no spread within dialogue 2


def test_memorybank_strength_slows_forgetting():
    weak = row(recency=T_END - 2 * 86400, frequency=0)
    strong = row(recency=T_END - 2 * 86400, frequency=3)
    w, s = memorybank_score([weak, strong])
    assert w == pytest.approx(math.exp(-2)) and s == pytest.approx(math.exp(-0.5))


def test_metadata_score_is_a_logistic_of_age_and_mentions():
    s = metadata_score([row(opportunities=1, mentions=3)], intercept=-1.0, coef=(0.5, 0.25))[0]
    assert s == pytest.approx(1 / (1 + math.exp(-(-1 + 0.5 + 0.75))))


@pytest.mark.parametrize("text,expected", [("7", 7), ("Rating: 10", 10), ("about 3/10", 3), ("", None), ("none", None)])
def test_rating_parser(text, expected):
    assert parse_rating(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("On a scale of 1 to 10, I would rate this a 4.", 4),
    ("I would rate this memory a 3 out of 10.", 3),
    ("Rating: 7/10", 7),
    ("I would rate this", None),
])
def test_verbose_rating_parser_ignores_the_scale(text, expected):
    from benchmarks.dars_eval.retention_signals import parse_rating_verbose
    assert parse_rating_verbose(text) == expected
