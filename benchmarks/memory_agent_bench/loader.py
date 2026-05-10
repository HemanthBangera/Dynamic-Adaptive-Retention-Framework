"""
Load ai-hyz/MemoryAgentBench with the same semantics as upstream
`utils/eval_data_utils.load_data_huggingface`.
"""

from __future__ import annotations

import random
import logging
from typing import Any, Dict, List, Optional

from datasets import DatasetDict, load_dataset

logger = logging.getLogger(__name__)

SUPPORTED_SPLITS = frozenset(
    {
        "Accurate_Retrieval",
        "Test_Time_Learning",
        "Long_Range_Understanding",
        "Conflict_Resolution",
    }
)


def _ensure_field_is_list(field_value: Any) -> List[Any]:
    if isinstance(field_value, list):
        return field_value
    if field_value:
        return [field_value]
    return []


def _process_single_sample_qa_lists(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Mirror MemoryAgentBench `_process_single_sample_qa_lists`."""
    metadata = sample.get("metadata") or {}
    metadata_fields = [
        "question_dates",
        "question_types",
        "question_ids",
        "previous_events",
        "qa_pair_ids",
        "demo",
    ]
    processed = dict(sample)
    processed.update(
        {
            "questions": _ensure_field_is_list(sample.get("questions")),
            "answers": _ensure_field_is_list(sample.get("answers")),
            "source": metadata.get("source", ""),
            **{
                field: _ensure_field_is_list(metadata.get(field, []))
                for field in metadata_fields
            },
        }
    )
    return processed


def _load_split_table(
    huggingface_dataset_name: str,
    split_name: str,
    revision: Optional[str],
) -> Dataset:
    """Load one split as a HuggingFace `Dataset` row table."""
    kwargs: Dict[str, Any] = {}
    if revision:
        kwargs["revision"] = revision
    try:
        return load_dataset(huggingface_dataset_name, split=split_name, **kwargs)
    except Exception as first_err:
        logger.debug("split= load failed (%s); trying DatasetDict access", first_err)
        dd = load_dataset(huggingface_dataset_name, **kwargs)
        if isinstance(dd, DatasetDict) and split_name in dd:
            return dd[split_name]
        raise


def load_mab_filtered(
    split_name: str,
    source_filter: str,
    max_test_samples: Optional[int] = None,
    seed: int = 42,
    revision: str = "main",
) -> List[Dict[str, Any]]:
    """
    Load HF rows for one competency split, filter `metadata.source == source_filter`,
    optionally subsample with deterministic seed, return list of processed dicts.
    """
    if split_name not in SUPPORTED_SPLITS:
        raise ValueError(
            f"Unknown split {split_name!r}. Expected one of {sorted(SUPPORTED_SPLITS)}"
        )

    raw = _load_split_table("ai-hyz/MemoryAgentBench", split_name, revision or None)
    original_length = len(raw)
    filtered = raw.filter(
        lambda sample: (sample.get("metadata") or {}).get("source", "") == source_filter
    )
    logger.info(
        "MAB load split=%s source=%s: %d rows (from %d)",
        split_name,
        source_filter,
        len(filtered),
        original_length,
    )
    if len(filtered) == 0:
        raise ValueError(
            f"No rows for split={split_name!r} with metadata.source={source_filter!r}. "
            "Run `python -m benchmarks.memory_agent_bench list-sources --split ...`."
        )

    n = len(filtered)
    if max_test_samples is not None and n > max_test_samples:
        rng = random.Random(seed)
        indices = list(range(n))
        rng.shuffle(indices)
        pick = sorted(indices[:max_test_samples])
        filtered = filtered.select(pick)
        logger.info("Subsampled to %d rows (seed=%s)", max_test_samples, seed)

    return [_process_single_sample_qa_lists(dict(row)) for row in filtered]


def list_sources_for_split(split_name: str, revision: str = "main") -> List[str]:
    """Distinct `metadata.source` values present in a split (for pilot YAML)."""
    raw = _load_split_table("ai-hyz/MemoryAgentBench", split_name, revision or None)
    seen = set()
    for row in raw:
        src = (row.get("metadata") or {}).get("source", "") or ""
        if src:
            seen.add(src)
    return sorted(seen)
