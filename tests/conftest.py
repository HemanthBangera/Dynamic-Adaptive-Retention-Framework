"""
DARS Test Infrastructure — Real API Fixtures
=============================================
All fixtures connect to real Qdrant and Gemini.  No mocks.

Tests use gemini-2.5-flash-lite (15 RPM / 1000 RPD free tier) to avoid
rate-limit failures that plague gemini-2.5-flash (10 RPM / 250 RPD).
"""

import os
import time
import uuid
import pytest
from config.settings import DARSConfig
from core.layer_d.storage import MemoryVault
from core.layer_d.schema import MemoryPayload, MemoryPoint
from core.layer_d.embedding import EmbeddingEngine

# Use the cheapest model for tests to avoid rate-limit storms.
TEST_GEMINI_MODEL = "gemini-2.5-flash-lite"
os.environ["GEMINI_MODEL"] = TEST_GEMINI_MODEL
DARSConfig.GEMINI_MODEL = TEST_GEMINI_MODEL

TEST_COLLECTION = f"dars_test_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="session")
def vault():
    """Session-scoped MemoryVault pointing at a disposable test collection."""
    v = MemoryVault(collection_name=TEST_COLLECTION)
    v.initialize_collection(recreate=True)
    yield v
    try:
        v.delete_collection()
    except Exception:
        pass


@pytest.fixture(scope="session")
def embedder():
    return EmbeddingEngine()


@pytest.fixture(scope="session")
def config():
    return DARSConfig()


@pytest.fixture(autouse=True)
def _clean_collection(vault, request):
    """Wipe all points before each test unless marked preserve_data."""
    if "preserve_data" in [m.name for m in request.node.iter_markers()]:
        return
    try:
        vault.initialize_collection(recreate=True)
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _gemini_rate_limit_pause(request):
    """4-second pause after every Gemini test to stay within 15 RPM."""
    yield
    markers = [m.name for m in request.node.iter_markers()]
    if "asyncio" in markers:
        time.sleep(4)


def _seed_memory(vault: MemoryVault, text: str, **overrides) -> str:
    """Insert a memory and patch any payload overrides. Returns point_id."""
    pid = vault.store_memory(text, source="test")
    if overrides:
        vault.patch_payload(pid, overrides)
    return pid


# ── Gemini reachability probe ────────────────────────────────────────────────

def _gemini_reachable() -> bool:
    cfg = DARSConfig()
    if not cfg.GEMINI_API_KEY:
        return False

    import urllib.request
    import urllib.error
    import json as _json

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{TEST_GEMINI_MODEL}:generateContent"
    )
    body = _json.dumps({"contents": [{"parts": [{"text": "Reply OK"}]}]}).encode()
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={
            "Content-Type": "application/json",
            "X-goog-api-key": cfg.GEMINI_API_KEY,
        },
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return resp.status == 200
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                import time as _t
                _t.sleep(3 * (attempt + 1))
                continue
            return False
        except Exception:
            return False
    return True  # 429/503 after retries means API is reachable, just busy


requires_gemini = pytest.mark.skipif(
    not _gemini_reachable(),
    reason="Gemini API unreachable or key invalid — skipping live LLM tests.",
)
