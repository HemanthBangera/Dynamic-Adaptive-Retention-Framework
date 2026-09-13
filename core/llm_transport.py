"""
Provider-agnostic LLM transport for DARS (OpenAI Chat Completions).

Same call shape as ``GovernedGeminiTransport.generate_text(prompt) -> (text, key_index)``,
so Layers A/B/C and the benchmark reader can use either provider, plus:

* an optional system message,
* deterministic request settings (temperature 0 and an explicit seed by default),
* a content-addressed disk cache keyed by the full request (model, messages,
  temperature, seed, max_tokens): re-running an experiment replays identical
  outputs at no cost, and repeats with different seeds are cached separately,
* token and cost accounting per model,
* client-side rate limiting: a sliding 60-second window per model that reserves
  the estimated tokens of each request (prompt + max_tokens) and caps requests per
  minute below the account limits (``DARS_OPENAI_TPM``, default 150,000;
  ``DARS_OPENAI_RPM``, default 400).  The window is per process: run one
  LLM-heavy job at a time, or lower the limits per process,
* rate-limit (HTTP 429) retries that follow the server's "try again in" hint;
  an exhausted quota fails at once,
* fail-loud errors: a call that still fails after all retries raises
  ``LLMCallError``; nothing is silently replaced by a placeholder answer,
* an offline mode (``DARS_LLM_OFFLINE=1``) that serves only cached responses.

The API key is read from ``OPEN_AI_API_KEY`` / ``OPENAI_API_KEY`` (environment,
``DARS_ENV_FILE``, the repo ``.env`` or the parent folder's ``.env``) and is
never logged or included in cache records.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import os
import random
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "benchmark_runs" / "_llm_cache"
KEY_ENV_NAMES = ("OPEN_AI_API_KEY", "OPENAI_API_KEY")

# USD per 1M tokens (input, output), OpenAI standard pricing (checked September 2026).
PRICES_PER_MTOK: Dict[str, Tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
}


class LLMCallError(RuntimeError):
    """An LLM request failed after all retries (or could not be sent)."""


class LLMCacheMiss(LLMCallError):
    """Offline mode was requested and the response is not in the cache."""


def price_for(model: str) -> Optional[Tuple[float, float]]:
    """Per-1M-token (input, output) price for a model id or dated snapshot."""
    for base, price in PRICES_PER_MTOK.items():
        if model == base or model.startswith(base + "-20"):
            return price
    return None


def load_openai_api_key(env_file: Optional[str] = None) -> Optional[str]:
    """Find an OpenAI key without logging it: environment first, then .env files."""
    for name in KEY_ENV_NAMES:
        value = os.getenv(name, "").strip()
        if value:
            return value

    candidates: List[Path] = []
    if env_file:
        candidates.append(Path(env_file))
    if os.getenv("DARS_ENV_FILE"):
        candidates.append(Path(os.environ["DARS_ENV_FILE"]))
    candidates += [PROJECT_ROOT / ".env", PROJECT_ROOT.parent / ".env"]

    from dotenv import dotenv_values

    for path in candidates:
        if not path.is_file():
            continue
        values = dotenv_values(path)
        for name in KEY_ENV_NAMES:
            value = (values.get(name) or "").strip().strip('"').strip("'")
            if value:
                return value
    return None


# ── Rate limiting ────────────────────────────────────────────────────────


class RateLimiter:
    """Sliding-window limiter on tokens and requests per ``window`` seconds (thread-safe)."""

    def __init__(self, tpm: int, rpm: int, window: float = 60.0,
                 clock: Callable[[], float] = time.monotonic):
        self.tpm = int(tpm)
        self.rpm = int(rpm)
        self.window = float(window)
        self.clock = clock
        self._events: Deque[Tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def try_acquire(self, tokens: int) -> float:
        """Reserve ``tokens`` now and return 0.0, or return the seconds to wait before retrying.

        A request larger than the whole budget is capped to it, so it waits for an
        empty window instead of blocking forever.
        """
        tokens = max(0, min(int(tokens), self.tpm))
        with self._lock:
            now = self.clock()
            while self._events and now - self._events[0][0] >= self.window:
                self._events.popleft()
            used = sum(t for _, t in self._events)
            if len(self._events) < self.rpm and used + tokens <= self.tpm:
                self._events.append((now, tokens))
                return 0.0
            remaining_n, remaining_t = len(self._events), used
            for ts, t in self._events:
                remaining_n -= 1
                remaining_t -= t
                if remaining_n < self.rpm and remaining_t + tokens <= self.tpm:
                    return max(ts + self.window - now, 1e-3)
            return self.window

    async def acquire(self, tokens: int) -> None:
        while True:
            wait = self.try_acquire(tokens)
            if wait == 0.0:
                return
            await asyncio.sleep(wait)


_COLLECT_LOCK = threading.Lock()
_COLLECT_SEEN: Dict[str, set] = {}


def _collect_target(model: str) -> Optional[str]:
    """Path to collect this model's requests into, or None to call the API normally.

    ``DARS_LLM_COLLECT`` names the file; ``DARS_LLM_COLLECT_MODEL`` optionally restricts
    collection to one model, so an experiment can batch its reader calls while its
    (unrestricted) auxiliary model still runs live.
    """
    path = os.getenv("DARS_LLM_COLLECT", "").strip()
    if not path:
        return None
    only = os.getenv("DARS_LLM_COLLECT_MODEL", "").strip()
    return path if (not only or only == model) else None


def _append_collect(path: str, key: str, request: Dict[str, Any]) -> None:
    """Append one Batch API line, skipping keys already written in this process."""
    line = json.dumps({"custom_id": key, "method": "POST", "url": "/v1/chat/completions",
                       "body": request}, ensure_ascii=False)
    with _COLLECT_LOCK:
        seen = _COLLECT_SEEN.setdefault(path, set())
        if key in seen:
            return
        seen.add(key)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


_LIMITERS: Dict[str, RateLimiter] = {}
_LIMITERS_LOCK = threading.Lock()


def rate_limiter_for(model: str) -> RateLimiter:
    """The per-process limiter shared by every transport of ``model``."""
    with _LIMITERS_LOCK:
        limiter = _LIMITERS.get(model)
        if limiter is None:
            limiter = RateLimiter(tpm=int(os.getenv("DARS_OPENAI_TPM", "150000")),
                                  rpm=int(os.getenv("DARS_OPENAI_RPM", "400")))
            _LIMITERS[model] = limiter
        return limiter


@functools.lru_cache(maxsize=8)
def _encoding(model: str):
    import tiktoken

    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def estimate_tokens(request: Dict[str, Any]) -> int:
    """Upper-bound token estimate of a request for rate limiting: prompt + max_tokens."""
    try:
        enc = _encoding(request["model"])
        prompt = sum(len(enc.encode(m["content"])) + 4 for m in request["messages"])
    except ImportError:
        prompt = sum(len(m["content"]) // 3 + 4 for m in request["messages"])
    return prompt + int(request.get("max_tokens") or 0)


_RETRY_IN = re.compile(r"try again in (\d+(?:\.\d+)?)\s*(ms|s)\b", re.IGNORECASE)


def _suggested_wait(exc: BaseException) -> Optional[float]:
    """Seconds suggested by a 429 message ("Please try again in 1.549s"), if any."""
    m = _RETRY_IN.search(str(exc))
    if not m:
        return None
    value = float(m.group(1))
    return value / 1000.0 if m.group(2).lower() == "ms" else value


@dataclass
class UsageLedger:
    """Running totals of API usage for one transport (or shared across several)."""

    calls: int = 0
    cache_hits: int = 0
    failures: int = 0
    rate_limit_retries: int = 0
    transient_retries: int = 0
    coalesced: int = 0
    collected: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    by_model: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def record(self, model: str, prompt_tokens: int, completion_tokens: int, cost: float) -> None:
        self.calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.cost_usd += cost
        m = self.by_model.setdefault(
            model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
        )
        m["calls"] += 1
        m["prompt_tokens"] += prompt_tokens
        m["completion_tokens"] += completion_tokens
        m["cost_usd"] += cost

    def as_dict(self) -> Dict[str, Any]:
        return {
            "api_calls": self.calls,
            "cache_hits": self.cache_hits,
            "api_failures": self.failures,
            "rate_limit_retries": self.rate_limit_retries,
            "transient_retries": self.transient_retries,
            "coalesced": self.coalesced,
            "collected": self.collected,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "by_model": self.by_model,
        }


class OpenAITransport:
    """Cached, deterministic, rate-limited, fail-loud OpenAI Chat Completions transport."""

    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        env_file: Optional[str] = None,
        temperature: float = 0.0,
        seed: Optional[int] = 0,
        max_tokens: int = 256,
        timeout: float = 120.0,
        max_retries: int = 8,
        rate_limit_retries: int = 8,
        transient_retries: Optional[int] = None,
        max_concurrency: int = 8,
        cache_dir: Optional[Path] = DEFAULT_CACHE_DIR,
        use_cache: bool = True,
        offline: Optional[bool] = None,
        ledger: Optional[UsageLedger] = None,
    ):
        self.model = model
        self._api_key = api_key or load_openai_api_key(env_file)
        self.temperature = float(temperature)
        self.seed = seed
        self.max_tokens = int(max_tokens)
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.rate_limit_retries = int(rate_limit_retries)
        # Connection drops and 5xx are retried beyond the SDK's own retries, so a brief
        # network outage cannot abort a run that takes days.
        self.transient_retries = int(os.getenv("DARS_OPENAI_TRANSIENT_RETRIES", "8")
                                     if transient_retries is None else transient_retries)
        self.max_concurrency = int(max_concurrency)
        if cache_dir is DEFAULT_CACHE_DIR and os.getenv("DARS_LLM_CACHE_DIR"):
            cache_dir = Path(os.environ["DARS_LLM_CACHE_DIR"])
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.use_cache = bool(use_cache) and self.cache_dir is not None
        if offline is None:
            offline = os.getenv("DARS_LLM_OFFLINE", "").lower() in ("1", "true", "yes")
        self.offline = bool(offline)
        self.ledger = ledger or UsageLedger()
        self._loop_clients: List[Tuple[asyncio.AbstractEventLoop, Any, asyncio.Semaphore]] = []
        self._loop_inflight: List[Tuple[asyncio.AbstractEventLoop, Dict[str, "asyncio.Future"]]] = []

    def __repr__(self) -> str:
        return (
            f"OpenAITransport(model={self.model!r}, "
            f"key={'set' if self._api_key else 'missing'}, cache={self.use_cache})"
        )

    @property
    def has_credentials(self) -> bool:
        return bool(self._api_key)

    # ── Requests and cache ─────────────────────────────────────────────

    def build_request(
        self,
        prompt: str,
        system: Optional[str] = None,
        *,
        seed: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        return {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "seed": self.seed if seed is None else seed,
            "max_tokens": self.max_tokens if max_tokens is None else int(max_tokens),
        }

    @staticmethod
    def cache_key(request: Dict[str, Any]) -> str:
        blob = json.dumps(request, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def _cache_path(self, key: str) -> Path:
        assert self.cache_dir is not None
        return self.cache_dir / key[:2] / f"{key}.json"

    def _cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._cache_path(key)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring unreadable cache entry %s: %s", path.name, exc)
            return None

    def _cache_put(self, key: str, record: Dict[str, Any]) -> None:
        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)

    def _cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        price = price_for(self.model)
        if price is None:
            return 0.0
        return (prompt_tokens * price[0] + completion_tokens * price[1]) / 1_000_000

    # ── Client per event loop ──────────────────────────────────────────

    def _client_for_loop(self) -> Tuple[Any, asyncio.Semaphore]:
        loop = asyncio.get_running_loop()
        self._loop_clients = [e for e in self._loop_clients if not e[0].is_closed()]
        for entry_loop, client, sem in self._loop_clients:
            if entry_loop is loop:
                return client, sem
        if not self._api_key:
            raise LLMCallError(
                "No OpenAI API key found (set OPEN_AI_API_KEY, or DARS_ENV_FILE to a .env holding it)."
            )
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self._api_key, timeout=self.timeout, max_retries=self.max_retries)
        sem = asyncio.Semaphore(self.max_concurrency)
        self._loop_clients.append((loop, client, sem))
        return client, sem

    def _inflight_for_loop(self) -> Dict[str, "asyncio.Future"]:
        """This loop's in-flight requests, keyed by cache key.

        Concurrent duplicate requests share one API call, so a run can never hold two
        different answers to the same request while the cache keeps only one of them.
        """
        loop = asyncio.get_running_loop()
        self._loop_inflight = [e for e in self._loop_inflight if not e[0].is_closed()]
        for entry_loop, pending in self._loop_inflight:
            if entry_loop is loop:
                return pending
        pending: Dict[str, "asyncio.Future"] = {}
        self._loop_inflight.append((loop, pending))
        return pending

    # ── Public API ─────────────────────────────────────────────────────

    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        *,
        seed: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run (or replay from cache) one chat completion; returns the full record."""
        request = self.build_request(prompt, system, seed=seed, max_tokens=max_tokens)
        key = self.cache_key(request)

        if self.use_cache:
            hit = self._cache_get(key)
            if hit is not None:
                self.ledger.cache_hits += 1
                return {**hit, "cached": True, "cache_key": key}
        if self.offline:
            raise LLMCacheMiss(f"Offline mode: no cached response for request {key[:12]}")

        collect = _collect_target(self.model)
        if collect is not None:
            # Collect mode: write the exact request out for the Batch API instead of calling
            # the synchronous endpoint, and hand back a placeholder. The caller's outputs are
            # meaningless in this pass, so a collect run must write to a scratch directory;
            # the real numbers come from replaying the stage once the responses are cached.
            _append_collect(collect, key, request)
            self.ledger.collected += 1
            return {"request": request, "text": "", "finish_reason": "collected",
                    "model": self.model, "system_fingerprint": None,
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0}, "cost_usd": 0.0,
                    "created": time.time(), "cached": False, "collected": True, "cache_key": key}

        pending = self._inflight_for_loop()
        waiting = pending.get(key)
        if waiting is not None:              # identical request already in flight: share its answer
            record = await waiting
            self.ledger.coalesced += 1
            return {**record, "cached": True, "coalesced": True, "cache_key": key}

        client, sem = self._client_for_loop()
        from openai import APIConnectionError, InternalServerError, OpenAIError, RateLimitError

        limiter = rate_limiter_for(self.model)
        estimate = estimate_tokens(request)
        future = asyncio.get_running_loop().create_future()
        future.add_done_callback(lambda f: f.cancelled() or f.exception())   # never "never retrieved"
        pending[key] = future
        attempt = 0
        transient = 0
        try:
            while True:
                try:
                    async with sem:
                        await limiter.acquire(estimate)
                        resp = await client.chat.completions.create(**request)
                    break
                except RateLimitError as exc:
                    attempt += 1
                    if "insufficient_quota" in str(exc) or attempt > self.rate_limit_retries:
                        self.ledger.failures += 1
                        raise LLMCallError(
                            f"OpenAI call failed after retries ({type(exc).__name__}): {exc}"
                        ) from exc
                    self.ledger.rate_limit_retries += 1
                    wait = max(_suggested_wait(exc) or 0.0, min(60.0, 2.0 ** attempt)) + random.uniform(0.0, 1.0)
                    logger.warning("Rate limited on %s; retry %d/%d in %.1fs",
                                   self.model, attempt, self.rate_limit_retries, wait)
                    await asyncio.sleep(wait)
                except (APIConnectionError, InternalServerError) as exc:
                    transient += 1               # a dropped connection or a 5xx: wait and retry
                    if transient > self.transient_retries:
                        self.ledger.failures += 1
                        raise LLMCallError(
                            f"OpenAI call failed after retries ({type(exc).__name__}): {exc}"
                        ) from exc
                    self.ledger.transient_retries += 1
                    wait = min(60.0, 2.0 ** transient) + random.uniform(0.0, 1.0)
                    logger.warning("Transient error on %s (%s); retry %d/%d in %.1fs",
                                   self.model, type(exc).__name__, transient, self.transient_retries, wait)
                    await asyncio.sleep(wait)
                except OpenAIError as exc:
                    self.ledger.failures += 1
                    raise LLMCallError(
                        f"OpenAI call failed after retries ({type(exc).__name__}): {exc}"
                    ) from exc

            choice = resp.choices[0]
            usage = resp.usage
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            cost = self._cost(prompt_tokens, completion_tokens)
            record = {
                "request": request,
                "text": (choice.message.content or "").strip(),
                "finish_reason": choice.finish_reason,
                "model": resp.model,
                "system_fingerprint": getattr(resp, "system_fingerprint", None),
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
                "cost_usd": cost,
                "created": time.time(),
            }
            self.ledger.record(self.model, prompt_tokens, completion_tokens, cost)
            if self.use_cache:
                self._cache_put(key, record)     # cache before releasing, so late duplicates hit it
            if not future.done():
                future.set_result(record)
        except BaseException as exc:
            if not future.done():
                future.set_exception(exc)
            raise
        finally:
            pending.pop(key, None)
        return {**record, "cached": False, "cache_key": key}

    async def generate_text(
        self,
        prompt_text: str,
        system: Optional[str] = None,
        *,
        seed: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, int]:
        """Interface-compatible with ``GovernedGeminiTransport.generate_text``."""
        out = await self.complete(prompt_text, system=system, seed=seed, max_tokens=max_tokens)
        return out["text"], 0

    async def aclose(self) -> None:
        """Close the client bound to the running event loop, if any."""
        loop = asyncio.get_running_loop()
        keep = []
        for entry_loop, client, sem in self._loop_clients:
            if entry_loop is loop:
                await client.close()
            elif not entry_loop.is_closed():
                keep.append((entry_loop, client, sem))
        self._loop_clients = keep


# ── Default provider resolution for Layer A/B/C components ──────────────

_DEFAULT_TRANSPORTS: Dict[str, OpenAITransport] = {}


def resolve_provider() -> str:
    """``"openai"``, ``"gemini"`` or ``"none"`` (see ``DARSConfig.LLM_PROVIDER``)."""
    from config.settings import DARSConfig

    provider = (DARSConfig.LLM_PROVIDER or "").strip().lower()
    if provider in ("openai", "gemini"):
        return provider
    if DARSConfig.GEMINI_API_KEY:
        return "gemini"
    if load_openai_api_key():
        return "openai"
    return "none"


def get_default_transport(role: str = "aux") -> Optional[OpenAITransport]:
    """
    Shared OpenAI transport for a component role, or None when the legacy
    Gemini path (the component's own HTTP client) should be used.

    ``role="reader"`` uses ``DARSConfig.OPENAI_READER_MODEL``; every other
    role (judge, compressor, reformulator) uses ``DARSConfig.OPENAI_AUX_MODEL``.
    """
    if resolve_provider() != "openai":
        return None
    from config.settings import DARSConfig

    model = DARSConfig.OPENAI_READER_MODEL if role == "reader" else DARSConfig.OPENAI_AUX_MODEL
    transport = _DEFAULT_TRANSPORTS.get(model)
    if transport is None:
        transport = OpenAITransport(model)
        _DEFAULT_TRANSPORTS[model] = transport
    return transport
