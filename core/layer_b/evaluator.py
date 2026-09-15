import aiohttp
import asyncio
import logging
import re
from typing import TYPE_CHECKING, Optional

from config.settings import DARSConfig
from core.llm_transport import get_default_transport

if TYPE_CHECKING:
    from core.gemini_transport import GovernedGeminiTransport

logger = logging.getLogger(__name__)

# The judge prompt.  Kept at module level so the E5 reliability study
# (benchmarks/dars_eval/run_judge.py) validates exactly the prompt the system uses.
PROMPT_TEMPLATE = (
    "You are the DARS Success Evaluator.\n"
    "USER QUERY: {query}\n"
    "AGENT RESPONSE: {response}\n"
    "RETRIEVED MEMORIES: {memories}\n"
    "EVALUATION TASK: Did the provided memories actually help the agent "
    "answer the user query accurately?\n"
    "Respond ONLY with 'YES' or 'NO'. No explanation."
)

# A verdict is a whole-word YES or NO at the start of the reply (optionally quoted).
# "NOT SURE", "NONE" or "NOPE" are not verdicts, so they become NEUTRAL.
_VERDICT = re.compile(r"^\s*['\"`*]*(YES|NO)\b")


def build_judge_prompt(query: str, response: str, memories: str) -> str:
    return PROMPT_TEMPLATE.format(query=query, response=response, memories=memories)


def parse_verdict(raw: Optional[str]) -> str:
    """'YES', 'NO', or 'NEUTRAL' for anything that is not a clear binary verdict."""
    m = _VERDICT.match((raw or "").upper())
    return m.group(1) if m else "NEUTRAL"


class SuccessEvaluator:
    """
    The Judge — evaluates whether retrieved memories helped the agent answer correctly.

    Uses an LLM (injected transport, default OpenAI transport, or Gemini REST)
    for a binary YES/NO judgment. Returns NEUTRAL on any ambiguity or failure
    to prevent poisoning the learning loop.
    """

    def __init__(self, timeout: float = None, transport: Optional["GovernedGeminiTransport"] = None):
        self.transport = transport if transport is not None else get_default_transport("aux")
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
        if self.transport is not None:
            text, _ki = await self.transport.generate_text(prompt_text)
            return text

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
        Uses the LLM judge to decide whether the memories were helpful.
        Returns 'YES', 'NO', or 'NEUTRAL' on failure/uncertainty.

        Raises RuntimeError when no valid API key is configured,
        preventing silent no-op learning.
        """
        if not self.api_key and self.transport is None:
            raise RuntimeError("An LLM API key is required for Success Evaluator (Gemini or OpenAI).")

        prompt = build_judge_prompt(query, response, memories)

        last_error = None
        for attempt in range(1 + self.max_retries):
            try:
                raw = await self._call_gemini(prompt)

                if raw is None:
                    logger.warning("Empty evaluator response (attempt %d).", attempt + 1)
                    continue

                verdict = parse_verdict(raw)
                if verdict == "NEUTRAL":
                    logger.warning(
                        "Judge returned non-binary output: '%s'. Defaulting to NEUTRAL.", raw.strip()[:60],
                    )
                return verdict

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
