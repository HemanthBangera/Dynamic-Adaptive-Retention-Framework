"""Confirmatory MSC analysis: the family logic on synthetic facts, and the freeze guards."""

import hashlib
import json

import numpy as np
import pytest

from benchmarks.dars_eval import confirm_msc
from benchmarks.dars_eval.confirm_msc import FAMILY, all_scores, family

T0 = 1_700_000_000.0
T_END = T0 + 2 * 86400 + 60 * 200

CFG = {
    "read_h4": {"lambda": 0.05, "weights": [0.2, 0.0, 0.8, 0.0]},
    "read_h5": {"lambda": 0.0005, "weights": [0.3, 0.0, 0.7, 0.0]},
    "write_h4": {"lambda": 0.05, "weights": [0.2, 0.1, 0.5, 0.2]},
    "write_h5": {"lambda": 0.01, "weights": [0.4, 0.0, 0.3, 0.3]},
    "metadata_model": {"intercept": -1.0, "coef": [-2.9, 2.7]},
}


def synthetic(n_dialogues=80, per=12, seed=0):
    """Facts whose later restatement depends on write history, while retrieval statistics are noise."""
    rng = np.random.default_rng(seed)
    rows, importance = [], {}
    for d in range(n_dialogues):
        for i in range(per):
            created = int(rng.integers(0, 3))
            restated = [s for s in range(created + 1, 3) if rng.random() < 0.5]
            mentions = 1 + len(restated)
            last_write = T0 + (max([created] + restated)) * 86400
            p_needed = 0.15 + 0.35 * (created == 2) + 0.25 * len(restated)
            text = f"d{d} fact {i}"
            importance[text] = int(rng.integers(1, 11))
            succ, fail = int(rng.integers(0, 4)), int(rng.integers(0, 4))
            y = bool(rng.random() < min(p_needed, 0.95))
            rows.append({
                "dialogue": f"validation:{d}", "split": "validation", "text": text, "created_session": created,
                "mentions": mentions, "mention_sessions": [created] + restated, "last_write_time": last_write,
                "opportunities": 2 - created, "label_session": 3, "t_end": T_END,
                "recency": T0 + float(rng.random()) * 2 * 86400, "created_at": T0 + created * 86400 + i,
                "frequency": succ + fail, "success": succ, "failure": fail,
                "components": {"R": 0.5, "F": 0.2, "U": (succ + 1) / (succ + fail + 2), "P": float(rng.random())},
                "labels": {"lex_0.5": y, "lex_0.6": y, "lex_0.7": y, "embed_0.8": y},
            })
    return rows, importance


def test_family_has_eleven_comparisons_and_holm_is_applied():
    rows, imp = synthetic()
    scores = all_scores(rows, CFG, imp)
    res = family(rows, scores, n_boot_auc=200, n_boot=500)
    assert [r["id"] for r in res] == [f[0] for f in FAMILY] and len(res) == 11
    for r in res:
        assert r["holm_p"] >= r["p_value"] - 1e-12
        assert r["result"] in ("confirmed", "not confirmed", "significant, opposite direction")


def test_write_side_utility_beats_retrieval_utility_when_needs_follow_writes():
    rows, imp = synthetic(n_dialogues=150)
    scores = all_scores(rows, CFG, imp)
    c5 = next(r for r in family(rows, scores, n_boot_auc=300, n_boot=500) if r["id"] == "C5")
    assert c5["diff"] > 0 and c5["result"] == "confirmed"


def test_a_significant_effect_against_the_prediction_is_labelled_as_such():
    rows, imp = synthetic(n_dialogues=150)
    scores = all_scores(rows, CFG, imp)
    scores["read_h4"], scores["recency_lru"] = scores["recency_lru"], scores["write_h4"]   # force C1a the wrong way
    c1a = next(r for r in family(rows, scores, n_boot_auc=300, n_boot=500) if r["id"] == "C1a")
    assert c1a["diff"] < 0 and c1a["result"] == "significant, opposite direction"


def _setup(tmp_path, frozen=True, tamper=False, cfg_in_text=True):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(CFG), encoding="utf-8")
    cfg_hash = hashlib.sha256(cfg.read_bytes()).hexdigest()
    add = tmp_path / "addendum.md"
    add.write_text(("**Status: FROZEN**" if frozen else "**Status: DRAFT**") + "\n"
                   + (f"config {cfg_hash}\n" if cfg_in_text else ""), encoding="utf-8")
    (tmp_path / "addendum.md.sha256").write_text(hashlib.sha256(add.read_bytes()).hexdigest() + "\n", encoding="utf-8")
    if tamper:
        add.write_text(add.read_text(encoding="utf-8") + "edited\n", encoding="utf-8")
    return ["confirm", "--config", str(cfg), "--addendum", str(add), "--runs", str(tmp_path / "none"),
            "--importance", str(tmp_path / "none.jsonl"), "--out", str(tmp_path / "out")]


@pytest.mark.parametrize("kw,msg", [({"frozen": False}, "not frozen"), ({"tamper": True}, "SHA-256"),
                                    ({"cfg_in_text": False}, "config")])
def test_confirm_refuses_to_touch_data_unless_the_addendum_is_frozen_and_intact(tmp_path, kw, msg):
    with pytest.raises(SystemExit) as exc:
        confirm_msc.main(_setup(tmp_path, **kw))
    assert msg in str(exc.value)
