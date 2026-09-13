"""Importance ratings: one call per distinct text, the Park et al. prompt, neutral fallback when unparseable."""

import asyncio
import json

from benchmarks.dars_eval.importance import NEUTRAL_RATING, distinct_texts, load_ratings, rate


class _FakeTransport:
    def __init__(self, replies):
        self.replies = replies
        self.prompts = []

    async def complete(self, prompt, system=None, *, seed=None, max_tokens=None):
        self.prompts.append(prompt)
        return {"text": self.replies[len(self.prompts) - 1]}


def test_distinct_texts_are_rated_once_in_first_seen_order(tmp_path):
    f1, f2 = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    f1.write_text("\n".join(json.dumps({"text": t}) for t in ["x", "y", "x"]) + "\n", encoding="utf-8")
    f2.write_text(json.dumps({"text": "z"}) + "\n" + json.dumps({"text": "y"}) + "\n", encoding="utf-8")
    assert distinct_texts([f1, f2]) == ["x", "y", "z"]


def test_rating_uses_the_generative_agents_prompt_and_parses_replies(tmp_path):
    t = _FakeTransport(["8", "Rating: 2", "no idea"])
    recs = asyncio.run(rate(["i got married", "i brush my teeth", "hmm"], t))
    assert [r["rating"] for r in recs] == [8, 2, None]
    assert all("rate the likely poignancy" in p and "Memory: " in p for p in t.prompts)
    out = tmp_path / "imp.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    assert load_ratings(out) == {"i got married": 8, "i brush my teeth": 2, "hmm": NEUTRAL_RATING}
