"""Concurrent duplicate requests share one API call, so a run cannot hold two answers to one request."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import openai
import pytest

from core.llm_transport import LLMCallError, OpenAITransport

SECRET = "sk-test-secret-value-1234567890"


class CountingCompletions:
    """Returns a different answer to every call, so a duplicate call is visible in the result."""

    def __init__(self, delay=0.05, error=None):
        self.calls = 0
        self.delay = delay
        self.error = error

    async def create(self, **request):
        self.calls += 1
        n = self.calls                 # captured before awaiting, so concurrent calls differ
        await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=f"answer #{n}"),
                                     finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
            model="gpt-4o-mini-2024-07-18",
            system_fingerprint="fp_test",
        )


@pytest.fixture
def fake_openai(monkeypatch):
    def install(completions):
        class FakeAsyncOpenAI:
            def __init__(self, api_key=None, timeout=None, max_retries=None):
                self.chat = SimpleNamespace(completions=completions)

        monkeypatch.setattr(openai, "AsyncOpenAI", FakeAsyncOpenAI)
        return completions

    return install


def _transport(tmp_path, **kw):
    return OpenAITransport("gpt-4o-mini-2024-07-18", api_key=SECRET, cache_dir=tmp_path, **kw)


@pytest.mark.asyncio
async def test_concurrent_duplicates_make_one_call_and_agree(fake_openai, tmp_path):
    completions = fake_openai(CountingCompletions())
    t = _transport(tmp_path)
    results = await asyncio.gather(*(t.complete("same question") for _ in range(5)))
    assert completions.calls == 1
    assert {r["text"] for r in results} == {"answer #1"}
    assert t.ledger.calls == 1 and t.ledger.coalesced == 4
    assert t.ledger.as_dict()["coalesced"] == 4


@pytest.mark.asyncio
async def test_distinct_requests_still_run_concurrently(fake_openai, tmp_path):
    completions = fake_openai(CountingCompletions())
    t = _transport(tmp_path)
    results = await asyncio.gather(*(t.complete(f"q{i}") for i in range(4)))
    assert completions.calls == 4
    assert len({r["text"] for r in results}) == 4
    assert t.ledger.coalesced == 0


@pytest.mark.asyncio
async def test_a_later_duplicate_is_served_from_the_cache(fake_openai, tmp_path):
    completions = fake_openai(CountingCompletions(delay=0.0))
    t = _transport(tmp_path)
    first = await t.complete("q")
    second = await t.complete("q")
    assert completions.calls == 1
    assert second["cached"] is True and second["text"] == first["text"]
    assert t.ledger.cache_hits == 1 and t.ledger.coalesced == 0


@pytest.mark.asyncio
async def test_failure_reaches_every_waiter_and_is_not_cached(fake_openai, tmp_path):
    fake_openai(CountingCompletions(error=openai.OpenAIError("boom")))
    t = _transport(tmp_path)
    done = await asyncio.gather(*(t.complete("q") for _ in range(3)), return_exceptions=True)
    assert all(isinstance(e, LLMCallError) for e in done)
    assert not any(tmp_path.rglob("*.json"))
    # the in-flight entry is released, so a later attempt is free to try again
    assert all(not pending for _loop, pending in t._loop_inflight)
