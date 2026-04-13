import asyncio
import aiohttp
import json
import logging
from config.settings import DARSConfig

logger = logging.getLogger(__name__)

class QueryReformulator:
    """Expands underspecified queries into descriptive search vectors."""

    def __init__(self, timeout: float = 3.0):
        self.timeout = timeout
        self.api_key = DARSConfig.GEMINI_API_KEY
        self.model = DARSConfig.GEMINI_MODEL
        # Use v1beta or appropriate endpoint for Gemini REST API
        self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

    async def reformulate_query(self, raw_query: str) -> str:
        """
        Transforms underspecified queries into descriptive search vectors.
        Falls back to the raw query on timeout or failure.
        """
        if not self.api_key:
            logger.warning("No Gemini API key found. Falling back to raw query.")
            return raw_query

        prompt = (
            f"Analyze the User Input: '{raw_query}'.\n"
            f"FACT EXTRACTION: Identify and list any new facts or preferences (e.g., 'I like ice cream').\n"
            f"SEMANTIC EXPANSION: Generate 3-5 technical synonyms, related entities, and keyword variations of the underlying inquiry.\n"
            f"OUTPUT: Return a single string containing the expanded keywords. Do NOT rephrase or interpret the user's intent. Do NOT apologize. Keep entity names (e.g., 'Pista') untouched."
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
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        candidates = data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            if parts:
                                result = parts[0].get("text", "").strip()
                                
                                if not result:
                                    logger.warning("Empty expansion string. Falling back to raw_query.")
                                    return raw_query
                                
                                # Length difference check to prevent wildly hallucinated expansions
                                len_diff_ratio = abs(len(result) - len(raw_query)) / max(1, len(raw_query))
                                # Only enforce 50% length variance check on queries longer than 20 characters to avoid dropping valid short-query expansions
                                if len(raw_query) > 20 and len_diff_ratio > 0.5:
                                    logger.warning(f"Expanded query length differed by >50% ({len_diff_ratio:.2f}). Falling back to raw_query.")
                                    return raw_query
                                    
                                return result
                    else:
                        resp_text = await response.text()
                        logger.error(f"Gemini API Error {response.status}: {resp_text}")
        except asyncio.TimeoutError:
            logger.warning(f"Timeout of {self.timeout}s reached during Gemini query reformulation for query: '{raw_query}'. Failing open to standard execution.")
        except Exception as e:
            logger.error(f"Error during query reformulation for query: '{raw_query}'. Error: {e}. Failing open to standard execution.")

        # Fallback to the original raw query
        return raw_query
