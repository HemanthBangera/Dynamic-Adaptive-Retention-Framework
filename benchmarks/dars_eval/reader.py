"""
MemoryAgentBench-format reader used for every retrieval method.

Prompt, as in MemoryAgentBench's RAG agents:
    system : the MemoryAgentBench system message
    user   : "Memory 1:\\n<text>\\nMemory 2:\\n<text>…" + "\\n" + <rag_agent query template>
Memories are listed in rank order (best first).  Answers are scored with the
vendored MemoryAgentBench ``post_process`` (substring exact match, EM, F1, ROUGE).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from third_party.memoryagentbench_eval import post_process
from third_party.memoryagentbench_eval.templates import SYSTEM_MESSAGE, normalize_dataset_name

# Generation cap per dataset family (tokens).  Chosen for this study (the answer
# formats are short except summaries); recorded in every run manifest.
MAX_ANSWER_TOKENS = {
    "ruler_qa": 50,
    "eventqa": 100,
    "longmemeval": 100,
    "factconsolidation": 50,
    "in_context_learning": 20,
    "detective_qa": 100,
    "infbench_sum": 1200,
    "recsys_redial": 300,
}


def format_memories(memory_texts: Sequence[str]) -> str:
    return "\n".join(f"Memory {i + 1}:\n{t}" for i, t in enumerate(memory_texts))


def build_prompt(memory_texts: Sequence[str], formatted_query: str) -> str:
    if not memory_texts:
        return formatted_query
    return format_memories(memory_texts) + "\n" + formatted_query


def _json_safe(metrics: Dict[str, Any]) -> Dict[str, float]:
    return {k: float(v) for k, v in metrics.items() if isinstance(v, (bool, int, float))}


class MABReader:
    """Answers one benchmark query from a list of memory texts."""

    def __init__(self, transport, source: str, split_name: str,
                 max_answer_tokens: Optional[int] = None, seed: int = 0):
        self.transport = transport
        self.source = source
        self.split_name = split_name
        self.family = normalize_dataset_name(source)
        self.max_answer_tokens = int(max_answer_tokens or MAX_ANSWER_TOKENS.get(self.family, 100))
        self.seed = seed

    def settings(self) -> Dict[str, Any]:
        return {
            "model": getattr(self.transport, "model", None),
            "temperature": getattr(self.transport, "temperature", None),
            "max_answer_tokens": self.max_answer_tokens,
            "system_message": SYSTEM_MESSAGE,
            "memory_format": "Memory {i}:\\n{text} joined by \\n, then \\n + query",
        }

    async def answer(self, memory_texts: Sequence[str], formatted_query: str, gold: Any,
                     seed: Optional[int] = None) -> Dict[str, Any]:
        prompt = build_prompt(memory_texts, formatted_query)
        out = await self.transport.complete(
            prompt,
            system=SYSTEM_MESSAGE,
            seed=self.seed if seed is None else seed,
            max_tokens=self.max_answer_tokens,
        )
        metrics, extra = post_process(
            {"output": out["text"]}, gold, {"sub_dataset": self.source, "dataset": self.split_name}
        )
        return {
            "output": out["text"],
            "parsed_output": extra.get("parsed_output"),
            "metrics": _json_safe(metrics),
            "prompt_tokens": out["usage"]["prompt_tokens"],
            "completion_tokens": out["usage"]["completion_tokens"],
            "finish_reason": out.get("finish_reason"),
            "cached": out["cached"],
            "cache_key": out["cache_key"],
            "system_fingerprint": out.get("system_fingerprint"),
        }
