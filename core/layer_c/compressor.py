import aiohttp
import asyncio
import logging
from typing import Optional
from config.settings import DARSConfig
from core.layer_d.storage import MemoryVault

logger = logging.getLogger(__name__)


class SemanticCompressor:
    """
    The Distiller — uses Gemini to compress memory content into dense factual summaries.

    Preserves the original embedding vector in Qdrant (shadow indexing) so
    compressed memories remain discoverable via their original semantic meaning.
    """

    def __init__(self, timeout: float = None, vault: Optional[MemoryVault] = None):
        if vault is None:
            raise TypeError("SemanticCompressor requires an explicit vault instance")
        self.timeout = timeout or DARSConfig.GEMINI_TIMEOUT
        self.max_retries = DARSConfig.GEMINI_MAX_RETRIES
        self.api_key = DARSConfig.GEMINI_API_KEY
        self.model = DARSConfig.GEMINI_MODEL
        self.endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        self.vault = vault

    async def _call_gemini(self, prompt_text: str) -> str | None:
        """Make a single Gemini REST call. Returns extracted text or None."""
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
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            return parts[0].get("text", "").strip()
                elif resp.status == 429:
                    raise aiohttp.ClientResponseError(
                        resp.request_info, resp.history,
                        status=429, message="Rate limited",
                    )
                elif resp.status >= 500:
                    raise aiohttp.ClientResponseError(
                        resp.request_info, resp.history,
                        status=resp.status, message="Server error",
                    )
                else:
                    resp_text = await resp.text()
                    logger.error("Gemini compressor error %d: %s", resp.status, resp_text[:200])
        return None

    async def compress_memory(self, point_id: str, text_content: str) -> bool:
        """
        Compress memory text via Gemini and atomically patch the payload.

        Returns True on success, False on compression failure.
        Raises RuntimeError if no valid API key is configured.
        """
        if not text_content:
            logger.warning("Empty text content for point %s. Skipping compression.", point_id)
            return False

        if not self.api_key:
            raise RuntimeError("Gemini API key is required for Semantic Compressor.")

        prompt = (
            "Summarize the following memory into a single, dense, factual bullet point.\n"
            f"CONTENT: {text_content}\n"
            "REQUIREMENTS:\n"
            "Preserve all proper nouns, technical terms, and specific dates.\n"
            "Strip conversational filler and social context.\n"
            "Output ONLY the summarized bullet point."
        )

        compressed_text = None
        last_error = None

        for attempt in range(1 + self.max_retries):
            try:
                result = await self._call_gemini(prompt)
                if result:
                    compressed_text = result
                    break
                logger.warning("Empty compression response (attempt %d).", attempt + 1)
            except asyncio.TimeoutError:
                logger.warning("Gemini compressor timeout (attempt %d).", attempt + 1)
                last_error = "timeout"
            except aiohttp.ClientResponseError as e:
                if e.status in (429, 503) and attempt < self.max_retries:
                    await asyncio.sleep(2 ** attempt)
                last_error = str(e)
            except Exception as e:
                logger.error("Compressor error (attempt %d): %s", attempt + 1, e)
                last_error = str(e)

        if not compressed_text:
            logger.error("Compression failed after retries (%s) for point %s.", last_error, point_id)
            return False

        updates = {
            "text_content": compressed_text,
            "original_text_backup": text_content,
            "is_compressed": True,
        }

        loop = asyncio.get_running_loop()

        try:
            pt = self.vault.get_memory(point_id)
            if pt and hasattr(pt.payload, "tags") and pt.payload.tags:
                if "system:high_priority_distillation" in pt.payload.tags:
                    updates["tags"] = [
                        t for t in pt.payload.tags if t != "system:high_priority_distillation"
                    ]
                    logger.info("dars_high_priority_distillation_complete for memory ID %s", point_id)
        except Exception:
            pass

        await loop.run_in_executor(None, self.vault.patch_payload, point_id, updates)
        logger.info("Compressed memory %s: %d -> %d chars.", point_id, len(text_content), len(compressed_text))
        return True
