"""
Orchestrate one MemoryAgentBench context episode against DARS (Qdrant + Layer A + reader).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import hashlib
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Tuple

from config.settings import DARSConfig
from core.layer_a.gateway import CognitiveGateway
from core.layer_a.reranker import DARSReranker
from core.layer_d.storage import MemoryVault
from third_party.memoryagentbench_eval import count_tokens, metrics_summarization

from benchmarks.memory_agent_bench.chunking import chunk_context
from benchmarks.memory_agent_bench.qa_builder import build_qa_pairs, min_context_chars
from benchmarks.memory_agent_bench.reader import GeminiBenchmarkReader

logger = logging.getLogger(__name__)


def _sanitize_collection_segment(s: str, max_len: int = 32) -> str:
    t = re.sub(r"[^a-zA-Z0-9_-]+", "_", s.lower()).strip("_")
    return (t[:max_len] or "x").rstrip("_")


def _collection_name(split: str, source: str, context: str) -> str:
    """Unique per process run (avoids Qdrant 404 if two benchmarks share same context hash)."""
    h = hashlib.sha256(context.encode("utf-8", errors="ignore")).hexdigest()[:8]
    run = uuid.uuid4().hex[:8]
    base = f"dars_mab_{_sanitize_collection_segment(split)}_{_sanitize_collection_segment(source)}_{h}_{run}"
    return base[:63]


def _memories_to_bullets(memories) -> str:
    lines = []
    for m in memories:
        text = getattr(m.payload, "text_content", "") or ""
        lines.append(f"- {text}")
    return "\n".join(lines) if lines else "(no memories retrieved)"


@dataclass
class EpisodeResult:
    metrics: DefaultDict[str, List[Any]] = field(default_factory=lambda: defaultdict(list))
    rows: List[Dict[str, Any]] = field(default_factory=list)


async def run_single_sample(
    *,
    row: Dict[str, Any],
    split_name: str,
    source_filter: str,
    chunk_size: int,
    tiktoken_model: str,
    path_mode: str,
    fetch_k: int,
    top_n: int,
    alpha: float,
    baseline: str,
    max_qa: Optional[int],
    gemini_sleep_s: float,
    dataset_config: Dict[str, Any],
    reader: GeminiBenchmarkReader,
    gateway_factory: Callable[[MemoryVault], CognitiveGateway],
    vault_factory: Callable[[str], MemoryVault],
) -> EpisodeResult:
    out = EpisodeResult()
    context = row.get("context") or ""
    if len(context) < min_context_chars():
        logger.warning("skip_short_context len=%d", len(context))
        return out

    cname = _collection_name(split_name, source_filter, context)
    vault = vault_factory(cname)
    vault.initialize_collection(recreate=True)

    t0 = time.time()
    if baseline != "empty":
        chunks = chunk_context(context, chunk_size=chunk_size, tiktoken_model=tiktoken_model)
        for i, chunk in enumerate(chunks):
            vault.store_memory(
                chunk,
                source=f"mab:{split_name}:{source_filter}",
                tags=["mab", split_name, source_filter, f"chunk:{i}"],
            )
    mem_time = time.time() - t0

    reranker = DARSReranker(vault=vault)
    gateway = gateway_factory(vault)

    qa_list = build_qa_pairs(row, sub_dataset=source_filter)
    if max_qa is not None and max_qa > 0:
        qa_list = qa_list[:max_qa]

    for qi, (formatted_query, answer, qa_pair_id) in enumerate(qa_list):
        q_start = time.time()
        if path_mode.lower() == "a":
            xml_prompt = await gateway.process_query(formatted_query)
            raw = await reader.answer_with_gateway_xml(xml_prompt)
            mem_ctx = xml_prompt
        else:
            loop = asyncio.get_running_loop()
            memories = await loop.run_in_executor(
                None,
                lambda: reranker.rerank(
                    query=formatted_query,
                    fetch_k=fetch_k,
                    top_n=top_n,
                    alpha=alpha,
                ),
            )
            bullets = _memories_to_bullets(memories)
            raw = await reader.answer_with_memories_bullets(formatted_query, bullets)
            mem_ctx = bullets

        q_time = time.time() - q_start
        inp_tokens = count_tokens(mem_ctx + "\n" + formatted_query, model_name=tiktoken_model)
        out_tokens = count_tokens(raw, model_name=tiktoken_model)

        payload = {
            "output": raw,
            "input_len": inp_tokens,
            "output_len": out_tokens,
            "memory_construction_time": mem_time if qi == 0 else 0.0,
            "query_time_len": q_time,
        }
        metrics_summarization(
            payload,
            formatted_query,
            answer,
            dataset_config,
            out.metrics,
            out.rows,
            query_id=qi,
            qa_pair_id=qa_pair_id,
        )
        if gemini_sleep_s > 0:
            await asyncio.sleep(gemini_sleep_s)

    try:
        vault.delete_collection()
    except Exception as exc:
        logger.warning("collection_cleanup_failed %s: %s", cname, exc)

    return out


def merge_episode_metrics(acc: DefaultDict[str, List], ep: EpisodeResult) -> None:
    for k, vs in ep.metrics.items():
        acc[k].extend(vs)


def summarize_metrics(acc: DefaultDict[str, List]) -> Dict[str, Any]:
    import numpy as np

    summary: Dict[str, Any] = {}
    for k, vs in acc.items():
        if not vs:
            continue
        arr = np.array(vs, dtype=float)
        summary[k] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "n": int(len(vs)),
        }
    return summary


def write_summary_md(path: Path, title: str, summary: Dict[str, Any], manifest: Dict[str, Any]) -> None:
    lines = [f"# {title}", "", "## Manifest", "", "```json", json.dumps(manifest, indent=2), "```", "", "## Metrics", ""]
    for k, v in sorted(summary.items()):
        lines.append(f"- **{k}**: mean={v['mean']:.4f} std={v['std']:.4f} (n={v['n']})")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
