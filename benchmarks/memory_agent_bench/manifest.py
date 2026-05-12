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
    min_context_tokens: Optional[int] = None,
    max_context_tokens: Optional[int] = None,
    load_stats: Optional[Dict[str, int]] = None,
    gemini_min_interval_s: float = 0.0,
    gemini_keys_file: Optional[str] = None,
    vault_recreate: bool = True,
    keep_collection: bool = False,
    run_label: Optional[str] = None,
    audit_jsonl: Optional[str] = None,
    chunk_overlap_tokens: int = 0,
    narrative_profile: bool = False,
    failure_detail_jsonl: Optional[str] = None,
    tombstone_sim_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    m: Dict[str, Any] = {
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
        "gemini_min_interval_s": gemini_min_interval_s,
        "gemini_keys_file": gemini_keys_file,
        "min_context_tokens": min_context_tokens,
        "max_context_tokens": max_context_tokens,
        "load_stats": load_stats or {},
        "vault_recreate": vault_recreate,
        "keep_collection": keep_collection,
        "run_label": run_label,
        "audit_jsonl": audit_jsonl,
        "chunk_overlap_tokens": chunk_overlap_tokens,
        "narrative_profile_applied": narrative_profile,
        "failure_detail_jsonl": failure_detail_jsonl,
        "tombstone_sim_threshold": tombstone_sim_threshold,
        "mab_expand_neighbor_chunks": DARSConfig.MAB_EXPAND_NEIGHBOR_CHUNKS,
        "mab_dual_query_retrieval": DARSConfig.MAB_DUAL_QUERY_RETRIEVAL,
        "mab_tombstone_sim_threshold": DARSConfig.MAB_TOMBSTONE_SIM_THRESHOLD,
        "mab_use_virtual_time": DARSConfig.MAB_USE_VIRTUAL_TIME,
        "mab_virtual_time_step_s": DARSConfig.MAB_VIRTUAL_TIME_STEP_S,
        "mab_injection_initial_success": DARSConfig.MAB_INJECTION_INITIAL_SUCCESS,
        "gemini_model": DARSConfig.GEMINI_MODEL,
        "embedding_model": DARSConfig.EMBEDDING_MODEL,
        "goal_description_set": bool(DARSConfig.GOAL_DESCRIPTION),
        "training_group": DARSConfig.TRAINING_GROUP,
        "python": sys.version,
        "platform": platform.platform(),
        "created_unix": time.time(),
    }
    return m


def write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
