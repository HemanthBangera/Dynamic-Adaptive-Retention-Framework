"""
Run provenance: everything needed to say exactly which code, data and models
produced a result.  Recorded in every run manifest.
"""

from __future__ import annotations

import importlib.metadata as md
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PACKAGES = (
    "qdrant-client",
    "sentence-transformers",
    "transformers",
    "torch",
    "numpy",
    "openai",
    "tiktoken",
    "nltk",
    "datasets",
    "rank-bm25",
    "rouge-score",
)

HF_REPOS = {
    "memoryagentbench": ("datasets", "ai-hyz/MemoryAgentBench"),
    "squad_v2": ("datasets", "rajpurkar/squad_v2"),
    "hotpot_qa": ("datasets", "hotpotqa/hotpot_qa"),
    "alfworld": ("datasets", "awawa-agi/alfworld-raw"),
    "msc": ("datasets", "nayohan/multi_session_chat"),
    "embedding_model": ("models", "sentence-transformers/all-MiniLM-L6-v2"),
}


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=20, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def git_state() -> Dict[str, Any]:
    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def hf_snapshot(kind: str, repo_id: str) -> Optional[str]:
    """Commit hash of the locally cached Hugging Face snapshot (refs/main), if present."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        return None
    folder = Path(HF_HUB_CACHE) / f"{kind}--{repo_id.replace('/', '--')}" / "refs" / "main"
    try:
        return folder.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def package_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for name in PACKAGES:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = None
    return versions


def dars_settings() -> Dict[str, Any]:
    from config.settings import DARSConfig as C

    return {
        "weights": {
            "w_r": C.WEIGHT_RECENCY,
            "w_f": C.WEIGHT_FREQUENCY,
            "w_u": C.WEIGHT_UTILITY,
            "w_p": C.WEIGHT_PREDICTIVE,
        },
        "recency_decay_lambda_per_hour": C.RECENCY_DECAY_LAMBDA,
        "frequency_cap": C.FREQUENCY_CAP,
        "threshold_retain": C.THRESHOLD_RETAIN,
        "threshold_compress": C.THRESHOLD_COMPRESS,
        "grace_period_seconds": C.GRACE_PERIOD_SECONDS,
        "embedding_model": C.EMBEDDING_MODEL,
        "goal_description": C._resolve_goal_description(),
        "openai_reader_model": C.OPENAI_READER_MODEL,
        "openai_aux_model": C.OPENAI_AUX_MODEL,
    }


def embedding_cache() -> Optional[Dict[str, Any]]:
    """Namespace and hit/miss counts of the embedding cache (None when DARS_EMBED_CACHE is unset)."""
    if not os.getenv("DARS_EMBED_CACHE", "").strip():
        return None
    from core.layer_d.embedding import EmbeddingEngine

    return EmbeddingEngine().cache_stats()


PREREGISTRATION = PROJECT_ROOT / "experiments" / "preregistration.md"


def preregistration_sha256() -> Optional[str]:
    """SHA-256 of the pre-registration as it stood when the run was made."""
    import hashlib

    try:
        return hashlib.sha256(PREREGISTRATION.read_bytes()).hexdigest()
    except OSError:
        return None


def collect_provenance() -> Dict[str, Any]:
    return {
        "preregistration_sha256": preregistration_sha256(),
        "embedding_cache": embedding_cache(),
        "created_unix": time.time(),
        "git": git_state(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": package_versions(),
        "hf_snapshots": {k: hf_snapshot(kind, repo) for k, (kind, repo) in HF_REPOS.items()},
        "dars_settings": dars_settings(),
    }
