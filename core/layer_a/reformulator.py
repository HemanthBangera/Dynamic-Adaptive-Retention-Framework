import asyncio
import aiohttp
import logging
from config.settings import DARSConfig

logger = logging.getLogger(__name__)


class QueryReformulator:
    """
    Expands underspecified queries into descriptive search vectors via Gemini API.

    Fail-open design: any API failure, timeout, or invalid response falls back
    to the raw user query so the pipeline never blocks.
    """

    def __init__(self, timeout: float = None):
        self.timeout = timeout or DARSConfig.GEMINI_TIMEOUT
        self.max_retries = DARSConfig.GEMINI_MAX_RETRIES
        self.max_expansion_chars = DARSConfig.GEMINI_MAX_EXPANSION_CHARS
        self.api_key = DARSConfig.GEMINI_API_KEY
        self.model = DARSConfig.GEMINI_MODEL
        self.endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )

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
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            return parts[0].get("text", "").strip()
                elif response.status == 429:
                    logger.warning("Gemini API rate-limited (429). Will retry.")
                    raise aiohttp.ClientResponseError(
                        response.request_info, response.history,
                        status=429, message="Rate limited",
                    )
                elif response.status >= 500:
                    resp_text = await response.text()
                    logger.warning("Gemini server error %d: %s", response.status, resp_text[:200])
                    raise aiohttp.ClientResponseError(
                        response.request_info, response.history,
                        status=response.status, message="Server error",
                    )
                else:
                    resp_text = await response.text()
                    logger.error("Gemini API error %d: %s", response.status, resp_text[:300])
        return None

    async def reformulate_query(self, raw_query: str) -> str:
        """
        Transforms underspecified queries into descriptive search vectors.
        Falls back to the raw query on timeout, failure, or empty response.
        """
        if not self.api_key:
            logger.warning("No Gemini API key found. Falling back to raw query.")
            return raw_query

        prompt = (
            f"Analyze the User Input: '{raw_query}'.\n"
            f"FACT EXTRACTION: Identify and list any new facts or preferences.\n"
            f"SEMANTIC EXPANSION: Generate 3-5 technical synonyms, related entities, "
            f"and keyword variations of the underlying inquiry.\n"
            f"OUTPUT: Return a single concise string (under {self.max_expansion_chars} characters) "
            f"containing the expanded keywords. Do NOT rephrase or interpret the user's intent. "
            f"Do NOT apologize. Keep entity names (e.g., 'Pista') untouched."
        )

        last_error = None
        for attempt in range(1 + self.max_retries):
            try:
                result = await self._call_gemini(prompt)

                if not result:
                    logger.warning("Empty expansion from Gemini (attempt %d). Retrying.", attempt + 1)
                    continue

                if len(result) > self.max_expansion_chars:
                    result = result[:self.max_expansion_chars]
                    logger.info("Expansion truncated to %d chars.", self.max_expansion_chars)

                logger.info(
                    "Query reformulated: '%s' -> '%s' (%d chars)",
                    raw_query[:50], result[:50], len(result),
                )
                return result

            except asyncio.TimeoutError:
                logger.warning(
                    "Gemini timeout (attempt %d/%d, %.1fs) for query: '%s'.",
                    attempt + 1, 1 + self.max_retries, self.timeout, raw_query[:50],
                )
                last_error = "timeout"
            except aiohttp.ClientResponseError as e:
                if e.status in (429, 503) and attempt < self.max_retries:
                    wait = 2 ** attempt
                    logger.info("Retryable %d, backing off %ds.", e.status, wait)
                    await asyncio.sleep(wait)
                last_error = str(e)
            except Exception as e:
                logger.error("Gemini reformulation error (attempt %d): %s", attempt + 1, e)
                last_error = str(e)

        logger.warning("All Gemini attempts exhausted (%s). Falling back to raw query.", last_error)
        return raw_query
