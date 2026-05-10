"""Run manifest for reproducible MemoryAgentBench × DARS experiments."""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from config.settings import DARSConfig


def build_manifest(
    *,
    split: str,
    source: str,
    chunk_size: int,
    max_test_samples: Optional[int],
    seed: int,
    hf_revision: str,
    path_mode: str,
    fetch_k: int,
    top_n: int,
    alpha: float,
    baseline: str,
    tiktoken_model: str,
    max_qa: Optional[int],
    gemini_sleep_s: float,
    gemini_max_retries: int,
    upstream_mab_commit: str = "main",
) -> Dict[str, Any]:
    return {
        "benchmark": "MemoryAgentBench",
        "hf_dataset": "ai-hyz/MemoryAgentBench",
        "hf_revision": hf_revision,
        "upstream_mab_eval_utils_pin": upstream_mab_commit,
        "split": split,
        "metadata_source": source,
        "chunk_size_tokens": chunk_size,
        "tiktoken_model": tiktoken_model,
        "max_test_samples": max_test_samples,
        "seed": seed,
        "path_mode": path_mode,
        "dars_fetch_k": fetch_k,
        "dars_top_n": top_n,
        "dars_rerank_alpha": alpha,
        "baseline_mode": baseline,
        "max_qa_per_context": max_qa,
        "gemini_inter_qa_sleep_s": gemini_sleep_s,
        "gemini_max_retries": gemini_max_retries,
        "gemini_model": DARSConfig.GEMINI_MODEL,
        "embedding_model": DARSConfig.EMBEDDING_MODEL,
        "goal_description_set": bool(DARSConfig.GOAL_DESCRIPTION),
        "training_group": DARSConfig.TRAINING_GROUP,
        "python": sys.version,
        "platform": platform.platform(),
        "created_unix": time.time(),
    }


def write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
