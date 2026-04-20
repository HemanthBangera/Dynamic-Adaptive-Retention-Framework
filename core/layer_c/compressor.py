import aiohttp
import asyncio
import logging
from typing import Optional
from config.settings import DARSConfig
from core.layer_d.storage import MemoryVault

logger = logging.getLogger(__name__)

class SemanticCompressor:
    """The Distiller - Uses Gemini 2.5 Flash to distill memory content."""

    def __init__(self, timeout: float = 5.0, vault: Optional[MemoryVault] = None):
        self.timeout = timeout
        self.api_key = DARSConfig.GEMINI_API_KEY
        self.model = DARSConfig.GEMINI_MODEL
        self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        self.vault = vault or MemoryVault()

    async def compress_memory(self, point_id: str, text_content: str) -> bool:
        """Compress memory and replace payload atomic property."""
        if not text_content:
            logger.warning(f"Empty text content for point {point_id}. Skipping compression.")
            return False

        if not self.api_key or self.api_key == "dummy_key":
            logger.warning("No Gemini API key found. Raising RuntimeError to prevent silent failure.")
            raise RuntimeError("Gemini API key is required for Semantic Compressor.")

        prompt = (
            "Summarize the following memory into a single, dense, factual bullet point.\n"
            f"CONTENT: {text_content}\n"
            "REQUIREMENTS:\n"
            "Preserve all proper nouns, technical terms, and specific dates.\n"
            "Strip conversational filler and social context.\n"
            "Output ONLY the summarized bullet point."
        )

        payload = {"contents": [{"parts": [{"text": prompt}]}]}

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
                                compressed_text = parts[0].get("text", "").strip()
                            else:
                                return False
                        else:
                            return False
                    else:
                        resp_text = await resp.text()
                        logger.error(f"Gemini API Error {resp.status} during compression: {resp_text}")
                        return False
        except Exception as e:
            logger.error(f"Error during compress_memory: {e}. Skipping.")
            return False

        # Perform the atomic payload patch using explicit Qdrant rules (no original vector, keeping shadow structure)
        updates = {
            "text_content": compressed_text,
            "original_text_backup": text_content,
            "is_compressed": True
        }
        
        loop = asyncio.get_running_loop()
        
        # Remove the high priority tag
        try:
            pt = self.vault.get_memory(point_id)
            if pt and hasattr(pt.payload, 'tags') and "system:high_priority_distillation" in pt.payload.tags:
                new_tags = [t for t in pt.payload.tags if t != "system:high_priority_distillation"]
                updates["tags"] = new_tags
                logger.info(f"dars_high_priority_distillation_complete for memory ID {point_id}")
        except Exception:
            pass
            
        await loop.run_in_executor(None, self.vault.patch_payload, point_id, updates)
        
        return True
