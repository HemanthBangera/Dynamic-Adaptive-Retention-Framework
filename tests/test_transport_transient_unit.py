"""A dropped connection or a 5xx is retried, so a brief outage cannot abort a multi-day run."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import openai
import pytest

import core.llm_transport as lt
from core.llm_transport import LLMCallError, OpenAITransport

SECRET = "sk-test-secret-value-1234567890"
REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def _connection_error():
    return openai.APIConnectionError(request=REQUEST)


def _server_error():
    response = httpx.Response(500, request=REQUEST)
    return openai.InternalServerError("boom", response=response, body=None)


class FlakyCompletions:
    """Raises the given errors in order, then answers."""

    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = 0

    async def create(self, **request):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
            model="gpt-4o-mini-2024-07-18",
            system_fingerprint="fp_test",
        )


@pytest.fixture
def fake_openai(monkeypatch):
    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(lt.asyncio, "sleep", no_sleep)          # keep the backoff instant in tests

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
async def test_connection_drop_is_retried_then_succeeds(fake_openai, tmp_path):
    completions = fake_openai(FlakyCompletions([_connection_error(), _connection_error()]))
    t = _transport(tmp_path)
    out = await t.complete("q")
    assert out["text"] == "ok" and completions.calls == 3
    assert t.ledger.transient_retries == 2 and t.ledger.failures == 0
    assert t.ledger.as_dict()["transient_retries"] == 2


@pytest.mark.asyncio
async def test_server_error_is_retried(fake_openai, tmp_path):
    completions = fake_openai(FlakyCompletions([_server_error()]))
    t = _transport(tmp_path)
    assert (await t.complete("q"))["text"] == "ok"
    assert completions.calls == 2 and t.ledger.transient_retries == 1


@pytest.mark.asyncio
async def test_outage_longer_than_the_budget_still_fails_loud(fake_openai, tmp_path):
    completions = fake_openai(FlakyCompletions([_connection_error() for _ in range(10)]))
    t = _transport(tmp_path, transient_retries=2)
    with pytest.raises(LLMCallError, match="APIConnectionError"):
        await t.complete("q")
    assert completions.calls == 3            # the first try plus two retries
    assert t.ledger.failures == 1
    assert not any(tmp_path.rglob("*.json"))


@pytest.mark.asyncio
async def test_a_real_error_is_still_fatal_at_once(fake_openai, tmp_path):
    completions = fake_openai(FlakyCompletions([openai.OpenAIError("bad request")]))
    t = _transport(tmp_path)
    with pytest.raises(LLMCallError):
        await t.complete("q")
    assert completions.calls == 1 and t.ledger.transient_retries == 0


@pytest.mark.asyncio
async def test_retry_budget_comes_from_the_environment(fake_openai, monkeypatch, tmp_path):
    monkeypatch.setenv("DARS_OPENAI_TRANSIENT_RETRIES", "1")
    completions = fake_openai(FlakyCompletions([_connection_error() for _ in range(4)]))
    t = _transport(tmp_path)
    assert t.transient_retries == 1
    with pytest.raises(LLMCallError):
        await t.complete("q")
    assert completions.calls == 2
    await asyncio.sleep(0)
