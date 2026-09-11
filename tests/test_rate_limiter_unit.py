"""Unit tests for the OpenAI transport's sliding-window rate limiter (no network)."""

import asyncio
import time

import pytest

from core.llm_transport import RateLimiter, _suggested_wait, estimate_tokens


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_tokens_per_minute_window():
    clock = FakeClock()
    lim = RateLimiter(tpm=100, rpm=10, clock=clock)
    assert lim.try_acquire(60) == 0.0
    clock.t = 10.0
    assert lim.try_acquire(30) == 0.0
    clock.t = 20.0
    # 60 + 30 + 30 > 100: wait until the first reservation leaves the window (t = 60)
    assert lim.try_acquire(30) == pytest.approx(40.0)
    clock.t = 60.0
    assert lim.try_acquire(30) == 0.0


def test_requests_per_minute_window():
    clock = FakeClock()
    lim = RateLimiter(tpm=10_000, rpm=2, clock=clock)
    assert lim.try_acquire(1) == 0.0
    clock.t = 5.0
    assert lim.try_acquire(1) == 0.0
    assert lim.try_acquire(1) == pytest.approx(55.0)


def test_oversized_request_is_capped_not_blocked_forever():
    clock = FakeClock()
    lim = RateLimiter(tpm=100, rpm=10, clock=clock)
    assert lim.try_acquire(500) == 0.0
    assert lim.try_acquire(1) == pytest.approx(60.0)


def test_acquire_waits_for_capacity():
    lim = RateLimiter(tpm=10, rpm=10, window=0.3)

    async def go():
        await lim.acquire(10)
        t0 = time.monotonic()
        await lim.acquire(5)
        return time.monotonic() - t0

    assert asyncio.run(go()) >= 0.2


def test_suggested_wait_parsing():
    assert _suggested_wait(Exception("Please try again in 1.549s. Visit ...")) == pytest.approx(1.549)
    assert _suggested_wait(Exception("try again in 250ms")) == pytest.approx(0.25)
    assert _suggested_wait(Exception("no hint here")) is None


def test_estimate_tokens_includes_max_tokens():
    req = {"model": "gpt-4o-mini-2024-07-18",
           "messages": [{"role": "user", "content": "hello world"}], "max_tokens": 50}
    assert 50 < estimate_tokens(req) < 70
