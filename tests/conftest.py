"""
DARS Test Infrastructure
========================
Vector store: an in-process local Qdrant store by default, so the suite runs
without cloud credentials.  Set ``DARS_TEST_BACKEND=remote`` (with
``QDRANT_URL`` / ``QDRANT_API_KEY``) to run the same tests against Qdrant Cloud.

LLM: live LLM tests use whichever provider is configured (see
``core.llm_transport.resolve_provider``): OpenAI (``gpt-4.1-nano`` for the
judge / compressor / reformulator) or Gemini (``gemini-2.5-flash-lite``).
Responses are cached under ``benchmark_runs/_llm_cache/tests`` so re-runs are
deterministic and free.  No network call happens at import time.
"""

import os
import time
import uuid
from pathlib import Path

import pytest

from config.settings import DARSConfig
from core.llm_transport import resolve_provider
from core.layer_d.storage import MemoryVault
from core.layer_d.schema import MemoryPayload, MemoryPoint
from core.layer_d.embedding import EmbeddingEngine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "DARS_LLM_CACHE_DIR", str(PROJECT_ROOT / "benchmark_runs" / "_llm_cache" / "tests")
)

LLM_PROVIDER = resolve_provider()

if LLM_PROVIDER == "gemini":
    # Use the cheapest Gemini model for tests to avoid rate-limit storms.
    TEST_GEMINI_MODEL = "gemini-2.5-flash-lite"
    os.environ["GEMINI_MODEL"] = TEST_GEMINI_MODEL
    DARSConfig.GEMINI_MODEL = TEST_GEMINI_MODEL

_REMOTE = os.getenv("DARS_TEST_BACKEND", "").lower() == "remote" and bool(DARSConfig.QDRANT_URL)
TEST_LOCATION = None if _REMOTE else ":memory:"
TEST_COLLECTION = f"dars_test_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="session")
def vault():
    """Session-scoped MemoryVault on a disposable collection (local store by default)."""
    v = MemoryVault(collection_name=TEST_COLLECTION, location=TEST_LOCATION)
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
    """Gemini free tier only: 4-second pause after async tests to stay within 15 RPM."""
    yield
    if LLM_PROVIDER != "gemini":
        return
    markers = [m.name for m in request.node.iter_markers()]
    if "asyncio" in markers:
        time.sleep(4)


def _seed_memory(vault: MemoryVault, text: str, **overrides) -> str:
    """Insert a memory and patch any payload overrides. Returns point_id."""
    pid = vault.store_memory(text, source="test")
    if overrides:
        vault.patch_payload(pid, overrides)
    return pid


# Live-LLM tests run only when a provider is configured (decided by key presence).
requires_llm = pytest.mark.skipif(
    LLM_PROVIDER == "none",
    reason="No LLM credentials configured (OpenAI or Gemini) — skipping live LLM tests.",
)
# Backwards-compatible name used by the existing test modules.
requires_gemini = requires_llm
