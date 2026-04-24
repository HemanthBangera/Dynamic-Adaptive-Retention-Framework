import aiohttp
import asyncio
import logging
from config.settings import DARSConfig

logger = logging.getLogger(__name__)


class SuccessEvaluator:
    """
    The Judge — evaluates whether retrieved memories helped the agent answer correctly.

    Uses Gemini API for binary YES/NO judgment. Returns NEUTRAL on any ambiguity
    or failure to prevent poisoning the learning loop.
    """

    def __init__(self, timeout: float = None):
        self.timeout = timeout or DARSConfig.GEMINI_TIMEOUT
        self.max_retries = DARSConfig.GEMINI_MAX_RETRIES
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
                    logger.error("Gemini evaluator API error %d: %s", resp.status, resp_text[:200])
        return None

    async def evaluate_success(self, query: str, response: str, memories: str) -> str:
        """
        Uses Gemini to judge if the memories were helpful.
        Returns 'YES', 'NO', or 'NEUTRAL' on failure/uncertainty.

        Raises RuntimeError when no valid API key is configured,
        preventing silent no-op learning.
        """
        if not self.api_key:
            raise RuntimeError("Gemini API key is required for Success Evaluator.")

        prompt = (
            "You are the DARS Success Evaluator.\n"
            f"USER QUERY: {query}\n"
            f"AGENT RESPONSE: {response}\n"
            f"RETRIEVED MEMORIES: {memories}\n"
            "EVALUATION TASK: Did the provided memories actually help the agent "
            "answer the user query accurately?\n"
            "Respond ONLY with 'YES' or 'NO'. No explanation."
        )

        last_error = None
        for attempt in range(1 + self.max_retries):
            try:
                raw = await self._call_gemini(prompt)

                if raw is None:
                    logger.warning("Empty evaluator response (attempt %d).", attempt + 1)
                    continue

                verdict = raw.upper().strip()
                if verdict.startswith("YES"):
                    return "YES"
                if verdict.startswith("NO"):
                    return "NO"

                logger.warning(
                    "Judge returned non-binary output: '%s'. Defaulting to NEUTRAL.", verdict[:60],
                )
                return "NEUTRAL"

            except asyncio.TimeoutError:
                logger.warning("Gemini evaluator timeout (attempt %d).", attempt + 1)
                last_error = "timeout"
            except aiohttp.ClientResponseError as e:
                if e.status in (429, 503) and attempt < self.max_retries:
                    await asyncio.sleep(2 ** attempt)
                last_error = str(e)
            except Exception as e:
                logger.error("Evaluator error (attempt %d): %s", attempt + 1, e)
                last_error = str(e)

        logger.warning("Evaluator exhausted retries (%s). Returning NEUTRAL.", last_error)
        return "NEUTRAL"
