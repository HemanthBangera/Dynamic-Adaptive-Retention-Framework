import aiohttp
import asyncio
import logging
from config.settings import DARSConfig

logger = logging.getLogger(__name__)

class SuccessEvaluator:
    """The Judge - Evaluates if retrieved memories helped answer the query."""
    
    def __init__(self, timeout: float = 3.0):
        self.timeout = timeout
        self.api_key = DARSConfig.GEMINI_API_KEY
        self.model = DARSConfig.GEMINI_MODEL
        self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

    async def evaluate_success(self, query: str, response: str, memories: str) -> str:
        """
        Uses Gemini to judge if the memories were helpful.
        Returns 'YES', 'NO', or 'NEUTRAL' on failure/uncertainty.
        """
        if not self.api_key or self.api_key == "dummy_key":
            logger.warning("No Gemini API key for SuccessEvaluator. Raising RuntimeError.")
            raise RuntimeError("Gemini API key is required for Success Evaluator.")
            
        prompt = (
            "You are the DARS Success Evaluator.\n"
            f"USER QUERY: {query}\n"
            f"AGENT RESPONSE: {response}\n"
            f"RETRIEVED MEMORIES: {memories}\n"
            "EVALUATION TASK: Did the provided memories actually help the agent answer the user query accurately?\n"
            "Respond ONLY with 'YES' or 'NO'. No explanation."
        )

        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.endpoint,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                    headers={"Content-Type": "application/json"}
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        candidates = data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            if parts:
                                result = parts[0].get("text", "").strip().upper()
                                if result in ['YES', 'NO']:
                                    return result
                                else:
                                    logger.warning(f"Judge returned non-binary output: '{result}'. Defaulting to NEUTRAL.")
                                    return "NEUTRAL"
                    else:
                        logger.error(f"Gemini API Error {resp.status} during evaluation.")
                        return "NEUTRAL"
        except Exception as e:
            logger.error(f"Error during evaluate_success: {e}. Defaulting to NEUTRAL.")
            return "NEUTRAL"
