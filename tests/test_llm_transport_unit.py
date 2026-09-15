"""Unit tests for the OpenAI transport (fake client, no network)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import openai
import pytest

import core.llm_transport as lt
from config.settings import DARSConfig
from core.llm_transport import (
    LLMCacheMiss,
    LLMCallError,
    OpenAITransport,
    get_default_transport,
    price_for,
)

SECRET = "sk-test-secret-value-1234567890"


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    async def create(self, **request):
        self.requests.append(request)
        outcome = self.outcomes.pop(0) if self.outcomes else "ok"
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=" Answer: 42 "), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
            model="gpt-4o-mini-2024-07-18",
            system_fingerprint="fp_test",
        )


@pytest.fixture
def fake_openai(monkeypatch):
    holder = {}

    class FakeAsyncOpenAI:
        def __init__(self, api_key=None, timeout=None, max_retries=None):
            assert api_key == SECRET
            self.chat = SimpleNamespace(completions=holder["completions"])

        async def close(self):
            pass

    def install(outcomes=()):
        holder["completions"] = FakeCompletions(outcomes)
        monkeypatch.setattr(openai, "AsyncOpenAI", FakeAsyncOpenAI)
        return holder["completions"]

    return install


def _transport(tmp_path, **kw):
    return OpenAITransport("gpt-4o-mini-2024-07-18", api_key=SECRET, cache_dir=tmp_path, **kw)


@pytest.mark.asyncio
async def test_cache_hit_avoids_second_api_call(fake_openai, tmp_path):
    completions = fake_openai()
    t = _transport(tmp_path)
    first = await t.complete("What is 6*7?", system="Be terse.")
    second = await t.complete("What is 6*7?", system="Be terse.")
    assert first["text"] == second["text"] == "Answer: 42"
    assert first["cached"] is False and second["cached"] is True
    assert len(completions.requests) == 1
    assert t.ledger.calls == 1 and t.ledger.cache_hits == 1
    assert t.ledger.cost_usd == pytest.approx((10 * 0.15 + 2 * 0.60) / 1e6)


@pytest.mark.asyncio
async def test_request_is_deterministic_by_default(fake_openai, tmp_path):
    completions = fake_openai()
    await _transport(tmp_path).complete("q")
    req = completions.requests[0]
    assert req["temperature"] == 0.0 and req["seed"] == 0
    assert req["messages"] == [{"role": "user", "content": "q"}]


@pytest.mark.asyncio
async def test_different_seeds_are_cached_separately(fake_openai, tmp_path):
    completions = fake_openai()
    t = _transport(tmp_path)
    await t.complete("q", seed=1)
    await t.complete("q", seed=2)
    assert len(completions.requests) == 2


@pytest.mark.asyncio
async def test_failure_is_loud_and_not_cached(fake_openai, tmp_path):
    fake_openai(outcomes=[openai.OpenAIError("boom")])
    t = _transport(tmp_path)
    with pytest.raises(LLMCallError, match="failed after retries"):
        await t.complete("q")
    assert t.ledger.failures == 1
    assert not any(tmp_path.rglob("*.json"))


@pytest.mark.asyncio
async def test_offline_mode_raises_on_cache_miss(tmp_path):
    t = _transport(tmp_path, offline=True)
    with pytest.raises(LLMCacheMiss):
        await t.complete("never asked")


@pytest.mark.asyncio
async def test_missing_key_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(lt, "load_openai_api_key", lambda env_file=None: None)
    t = OpenAITransport("gpt-4o-mini", cache_dir=tmp_path)
    with pytest.raises(LLMCallError, match="No OpenAI API key"):
        await t.complete("q")


@pytest.mark.asyncio
async def test_key_never_in_repr_or_cache(fake_openai, tmp_path):
    fake_openai()
    t = _transport(tmp_path)
    await t.complete("q")
    assert SECRET not in repr(t)
    for f in tmp_path.rglob("*.json"):
        assert SECRET not in f.read_text(encoding="utf-8")
        assert json.loads(f.read_text(encoding="utf-8"))["request"]["model"] == t.model


@pytest.mark.asyncio
async def test_generate_text_interface(fake_openai, tmp_path):
    fake_openai()
    text, idx = await _transport(tmp_path).generate_text("q")
    assert text == "Answer: 42" and idx == 0


def test_price_for_snapshots():
    assert price_for("gpt-4o-mini-2024-07-18") == (0.15, 0.60)
    assert price_for("gpt-4.1-nano-2025-04-14") == (0.10, 0.40)
    assert price_for("unknown-model") is None


def test_default_transport_follows_provider(monkeypatch):
    monkeypatch.setattr(DARSConfig, "LLM_PROVIDER", "gemini")
    assert get_default_transport("aux") is None
    monkeypatch.setattr(DARSConfig, "LLM_PROVIDER", "openai")
    aux = get_default_transport("aux")
    reader = get_default_transport("reader")
    assert aux.model == DARSConfig.OPENAI_AUX_MODEL
    assert reader.model == DARSConfig.OPENAI_READER_MODEL
    assert get_default_transport("judge") is aux
