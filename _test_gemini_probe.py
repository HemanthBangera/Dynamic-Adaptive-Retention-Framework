"""Quick Gemini API probe — run directly to verify connectivity."""
import aiohttp
import asyncio
from config.settings import DARSConfig

async def probe():
    cfg = DARSConfig()
    print(f"Key: {cfg.GEMINI_API_KEY[:12]}...")
    print(f"Model: {cfg.GEMINI_MODEL}")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{cfg.GEMINI_MODEL}:generateContent"
    )
    payload = {"contents": [{"parts": [{"text": "Reply OK"}]}]}
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": cfg.GEMINI_API_KEY,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url, json=payload,
            timeout=aiohttp.ClientTimeout(total=15),
            headers=headers,
        ) as resp:
            print(f"Status: {resp.status}")
            data = await resp.text()
            print(f"Response: {data[:600]}")

if __name__ == "__main__":
    asyncio.run(probe())
