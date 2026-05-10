"""
Shared async Gemini HTTP transport: key rotation on 429, global min-interval pacing.

Used by MemoryAgentBench driver and optionally by Layer A/B/C Gemini callers.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import List, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)

# Google AI Studio / Generative Language API client keys (same as curl -H 'X-goog-api-key: …').
# Scan whole-file so rough dumps work: "Api key : AIzaSy…", "Apu key : …", labels on other lines.
_GEMINI_API_KEY_RE = re.compile(r"\bAIzaSy[A-Za-z0-9_-]{29,200}\b")


def extract_gemini_api_keys_from_text(raw: str) -> List[str]:
    """
    Return all plausible Gemini API keys in document order, deduplicated.

    Works for plain one-key-per-line files and messy exports (prefix/suffix text on the same line).
    """
    out: List[str] = []
    seen = set()
    for m in _GEMINI_API_KEY_RE.finditer(raw):
        k = m.group(0)
        if len(k) > 256:
            continue
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def line_looks_like_gemini_api_key(s: str) -> bool:
    """
    True if the line is plausibly a Google AI Studio / Generative Language API key.

    Exported key dumps often include labels ("API Key", "Project name", "projects/…")
    on their own lines; those must NOT be sent as X-goog-api-key (Google returns 400).
    """
    t = s.strip()
    if len(t) < 35 or len(t) > 256:
        return False
    # Typical keys: AIzaSy… (39 chars common); allow a little length slack.
    if t.startswith("AIzaSy") and t.isascii() and " " not in t:
        return True
    return False


def parse_gemini_keys_file(path: str) -> List[str]:
    """
    Load Gemini API keys from a file.

    Accepts plain one-key-per-line files, GCP console exports, and informal lines such as
    ``Api key : AIzaSy…`` (embedded key). Only substrings matching ``AIzaSy…`` are used.
    """
    from pathlib import Path

    raw = Path(path).read_text(encoding="utf-8")
    keys = extract_gemini_api_keys_from_text(raw)
    non_empty_lines = sum(1 for ln in raw.splitlines() if ln.strip() and not ln.lstrip().startswith("#"))
    if non_empty_lines > len(keys):
        logger.info(
            "parse_gemini_keys_file: extracted %d key(s) from %s (%d non-empty non-comment line(s); rest is labels/metadata or non-matching text)",
            len(keys),
            path,
            non_empty_lines,
        )
    elif keys:
        logger.debug("parse_gemini_keys_file: extracted %d key(s) from %s", len(keys), path)
    return keys


def collect_gemini_keys_from_env_and_file(keys_file: Optional[str] = None) -> List[str]:
    """Merge keys: optional file first, then GEMINI_API_KEYS, then GEMINI_API_KEY (deduped)."""
    import os

    keys: List[str] = []
    if keys_file:
        keys.extend(parse_gemini_keys_file(keys_file))
    bulk = os.getenv("GEMINI_API_KEYS", "").strip()
    if bulk:
        keys.extend(k.strip() for k in bulk.split(",") if k.strip())
    single = os.getenv("GEMINI_API_KEY", "").strip()
    if single and single not in keys:
        keys.append(single)
    # de-dupe preserving order
    seen = set()
    out: List[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def redact_key_suffix(key: str, n: int = 4) -> str:
    if len(key) <= n:
        return "****"
    return f"....{key[-n:]}"


def _is_quota_or_billing_block(http_body: str) -> bool:
    """True when 429 is billing/day quota (retries will not help until keys or plan change)."""
    b = http_body.lower()
    return any(
        s in b
        for s in (
            "quota",
            "billing",
            "resource_exhausted",
            "exceeded your current quota",
            "generate_requests_per_day",
        )
    )


class GovernedGeminiTransport:
    """
    One aiohttp session, min spacing between posts, rotate API key on 429/403.
    """

    def __init__(
        self,
        *,
        keys: List[str],
        model: str,
        timeout: float,
        min_interval_s: float,
        max_retries: int,
    ):
        if not keys:
            raise ValueError("GovernedGeminiTransport requires at least one API key")
        self._keys = list(keys)
        self._idx = 0
        self.model = model
        self.timeout = timeout
        self.min_interval_s = float(min_interval_s)
        self.max_retries = int(max_retries)
        self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        self._gate = asyncio.Lock()
        self._last_post_mono: float = 0.0
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()

    def _rotate(self) -> None:
        self._idx = (self._idx + 1) % len(self._keys)
        logger.info("Gemini key rotation -> index=%d suffix=%s", self._idx, redact_key_suffix(self._keys[self._idx]))

    def current_key_index(self) -> int:
        return self._idx % len(self._keys)

    async def _ensure_session(self) -> aiohttp.ClientSession:
        async with self._session_lock:
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession()
            return self._session

    async def aclose(self) -> None:
        async with self._session_lock:
            if self._session and not self._session.closed:
                await self._session.close()
                self._session = None

    async def _pace_before_request(self) -> None:
        """Enforce min wall time between the start of successive generateContent calls."""
        if self.min_interval_s <= 0:
            return
        async with self._gate:
            now = time.monotonic()
            wait = self.min_interval_s - (now - self._last_post_mono)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_post_mono = time.monotonic()

    async def generate_text(self, prompt_text: str) -> Tuple[Optional[str], int]:
        """
        POST generateContent; on 429/403 rotate key before backoff.

        Returns (text_or_none, key_index_used_on_successful_200).
        """
        session = await self._ensure_session()
        last_err: Optional[BaseException] = None
        key_at_success = 0
        quota_key_rotations = 0

        for attempt in range(1 + self.max_retries):
            await self._pace_before_request()
            key = self._keys[self._idx % len(self._keys)]
            ki = self._idx % len(self._keys)
            payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
            headers = {
                "Content-Type": "application/json",
                "X-goog-api-key": key,
            }
            try:
                async with session.post(
                    self.endpoint,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                    headers=headers,
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        cands = data.get("candidates", [])
                        if cands:
                            parts = cands[0].get("content", {}).get("parts", [])
                            if parts:
                                key_at_success = ki
                                return (parts[0].get("text") or "").strip(), ki
                        return None, ki
                    if resp.status in (429, 403):
                        body = await resp.text()
                        logger.warning(
                            "Gemini HTTP %s (attempt %d) key_index=%d suffix=%s body=%s",
                            resp.status,
                            attempt,
                            ki,
                            redact_key_suffix(key),
                            body[:120],
                        )
                        if resp.status == 429 and _is_quota_or_billing_block(body):
                            quota_key_rotations += 1
                            if len(self._keys) <= 1 or quota_key_rotations >= len(self._keys):
                                logger.error(
                                    "Gemini quota/billing exhaustion after %d key(s). "
                                    "Use another key/model, enable billing, or wait for quota reset.",
                                    quota_key_rotations,
                                )
                                return None, ki
                            logger.warning("Quota message on key_index=%d; rotating to next key.", ki)
                            self._rotate()
                            await asyncio.sleep(0.5)
                            continue
                        self._rotate()
                        last_err = aiohttp.ClientResponseError(
                            resp.request_info, resp.history, status=resp.status, message="Rate limited or forbidden"
                        )
                        base = 5.0 if resp.status == 429 else 2.0
                        delay = min(90.0, base * (1.7**attempt))
                        await asyncio.sleep(delay)
                        continue
                    if resp.status >= 500:
                        last_err = aiohttp.ClientResponseError(
                            resp.request_info, resp.history, status=resp.status, message="Server error"
                        )
                        delay = min(90.0, 1.5 * (1.7**attempt))
                        await asyncio.sleep(delay)
                        continue
                    body = await resp.text()
                    logger.error("Gemini error %s: %s", resp.status, body[:200])
                    return None, ki
            except asyncio.TimeoutError as e:
                last_err = e
                await asyncio.sleep(1.0 * (attempt + 1))
            except aiohttp.ClientError as e:
                last_err = e
                await asyncio.sleep(1.0 * (attempt + 1))

        logger.warning("GovernedGeminiTransport exhausted: %s", last_err)
        return None, key_at_success
