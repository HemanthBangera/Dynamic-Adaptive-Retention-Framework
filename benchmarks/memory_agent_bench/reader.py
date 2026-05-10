"""
Gemini reader for MemoryAgentBench-style QA (Mem0-like memory bullets).

Output should include `Answer:` so vendored `parse_output` can extract it.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import aiohttp

from config.settings import DARSConfig

logger = logging.getLogger(__name__)


class GeminiBenchmarkReader:
    def __init__(self, timeout: Optional[float] = None, max_retries: Optional[int] = None):
        self.timeout = timeout or DARSConfig.GEMINI_TIMEOUT
        self.max_retries = (
            int(max_retries) if max_retries is not None else int(DARSConfig.GEMINI_MAX_RETRIES)
        )
        self.api_key = DARSConfig.GEMINI_API_KEY
        self.model = DARSConfig.GEMINI_MODEL
        self.endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )

    async def _call_once(self, prompt_text: str) -> Optional[str]:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is required for MemoryAgentBench reader.")

        payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": self.api_key,
        }
        async with aiohttp.ClientSession() as session:
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
                            return (parts[0].get("text") or "").strip()
                elif resp.status == 429:
                    raise aiohttp.ClientResponseError(
                        resp.request_info, resp.history, status=429, message="Rate limited"
                    )
                elif resp.status >= 500:
                    raise aiohttp.ClientResponseError(
                        resp.request_info, resp.history, status=resp.status, message="Server error"
                    )
                else:
                    body = await resp.text()
                    logger.error("Gemini reader error %s: %s", resp.status, body[:300])
        return None

    async def answer_with_memories_bullets(
        self,
        formatted_user_query: str,
        memory_bullets: str,
    ) -> str:
        """
        Mem0-style: system lists memories; user message is the benchmark query block.
        """
        system = (
            "You are a helpful AI. Answer using ONLY the retrieved memory bullets below. "
            "If the answer is not in the memories, give your best concise guess. "
            "Start your reply with the line `Answer:` followed by the answer text."
        )
        user = f"--- Retrieved memories ---\n{memory_bullets}\n\n--- Task ---\n{formatted_user_query}"
        prompt = f"{system}\n\n{user}"
        last_err: Optional[BaseException] = None
        for attempt in range(1 + self.max_retries):
            try:
                out = await self._call_once(prompt)
                if out:
                    return out
            except aiohttp.ClientResponseError as e:
                last_err = e
                if e.status in (429, 500, 502, 503):
                    # Capped exponential backoff (uncapped 2**N caused multi-hour waits on 429).
                    base = 5.0 if e.status == 429 else 1.5
                    delay = min(90.0, base * (1.7**attempt))
                    await asyncio.sleep(delay)
                    continue
                raise
            except asyncio.TimeoutError as e:
                last_err = e
                await asyncio.sleep(1.0 * (attempt + 1))
                continue
        logger.warning("Gemini reader failed after retries: %s", last_err)
        return "Answer: "

    async def answer_with_gateway_xml(self, gateway_xml_prompt: str) -> str:
        """Path A: entire Layer A XML string is the user task."""
        system = (
            "You are the assistant. Use the XML memory stream to answer the current user query. "
            "Start your reply with the line `Answer:` followed by the answer text."
        )
        prompt = f"{system}\n\n{gateway_xml_prompt}"
        for attempt in range(1 + self.max_retries):
            try:
                out = await self._call_once(prompt)
                if out:
                    return out
            except aiohttp.ClientResponseError as e:
                if e.status in (429, 500, 502, 503):
                    base = 5.0 if e.status == 429 else 1.5
                    delay = min(90.0, base * (1.7**attempt))
                    await asyncio.sleep(delay)
                    continue
                raise
            await asyncio.sleep(1.0 * (attempt + 1))
        return "Answer: "
