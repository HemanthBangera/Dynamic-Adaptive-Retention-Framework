"""
Group B – ALFWorld Dataset Loader
===================================
Downloads awawa-agi/alfworld-raw from HuggingFace, caches all three splits
locally, and prints a metadata report for research validation.

Splits:
  - train           (~3 553 tasks)
  - eval_in_distribution   (~140 tasks, same task_types as train)
  - eval_out_of_distribution (~134 tasks)

Run standalone:  python -m data.groupB.loader
"""

from __future__ import annotations

import json
import logging
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "cache"
DATASET_NAME = "awawa-agi/alfworld-raw"

SPLITS: List[Tuple[str, str]] = [
    ("train", "alfworld_train.json"),
    ("eval_in_distribution", "alfworld_eval_in.json"),
    ("eval_out_of_distribution", "alfworld_eval_out.json"),
]


def download_and_cache(force: bool = False) -> Dict[str, List[Dict[str, Any]]]:
    """Download all ALFWorld splits from HuggingFace and cache as JSON.

    Returns ``{split_name: [rows]}`` mapping.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    all_cached = all((CACHE_DIR / fname).exists() for _, fname in SPLITS)

    if all_cached and not force:
        logger.info("Loading cached ALFWorld dataset from %s", CACHE_DIR)
        result: Dict[str, List[Dict[str, Any]]] = {}
        for split_name, fname in SPLITS:
            with open(CACHE_DIR / fname, "r", encoding="utf-8") as f:
                result[split_name] = json.load(f)
        return result

    logger.info("Downloading %s from HuggingFace (all splits)...", DATASET_NAME)
    from datasets import load_dataset

    result = {}
    for split_name, fname in SPLITS:
        logger.info("  Downloading split: %s ...", split_name)
        ds = load_dataset(DATASET_NAME, split=split_name)
        rows = [dict(row) for row in ds]
        with open(CACHE_DIR / fname, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False)
        logger.info("  Cached %d rows to %s", len(rows), CACHE_DIR / fname)
        result[split_name] = rows

    return result


def group_by_task_type(
    rows: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Group rows into ``{task_type: [tasks]}``."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["task_type"]].append(row)
    return dict(grouped)


def print_metadata_report(
    data: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Analyse and print a full metadata report across all splits."""
    stats: Dict[str, Any] = {}

    print("\n" + "=" * 70)
    print("  ALFWORLD DATASET – METADATA REPORT")
    print("=" * 70)

    for split_name, rows in data.items():
        print(f"\n  --- Split: {split_name} ({len(rows):,} rows) ---")
        stats[f"{split_name}_rows"] = len(rows)

        type_counter = Counter(r["task_type"] for r in rows)
        for ttype, count in type_counter.most_common():
            print(f"    {ttype}: {count}")
        stats[f"{split_name}_task_types"] = dict(type_counter)

        gc_lengths = []
        for r in rows:
            gc = r.get("game_content", "")
            gc_lengths.append(len(gc) if isinstance(gc, str) else 0)

        if gc_lengths:
            print(f"    game_content length — "
                  f"min={min(gc_lengths):,}  "
                  f"max={max(gc_lengths):,}  "
                  f"mean={statistics.mean(gc_lengths):,.0f}  "
                  f"median={statistics.median(gc_lengths):,.0f}")
            stats[f"{split_name}_gc_len_mean"] = round(statistics.mean(gc_lengths))

    all_types: set = set()
    for rows in data.values():
        for r in rows:
            all_types.add(r["task_type"])
    print(f"\n  Unique task_types across all splits: {sorted(all_types)}")
    stats["all_task_types"] = sorted(all_types)

    if "train" in data and data["train"]:
        sample = data["train"][0]
        print(f"\n  --- Sample Row (train[0]) ---")
        print(f"    id:             {sample.get('id', 'N/A')}")
        print(f"    task_type:      {sample.get('task_type', 'N/A')}")
        print(f"    game_file_path: {str(sample.get('game_file_path', 'N/A'))[:80]}")
        gc = sample.get("game_content", "")
        print(f"    game_content:   ({len(gc):,} chars)  {gc[:200]}...")

    print("=" * 70 + "\n")
    return stats


def load_alfworld(
    force_download: bool = False,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Main entry: load all splits grouped by task_type.

    Returns ``{split_name: {task_type: [rows]}}``
    """
    raw = download_and_cache(force=force_download)
    return {split_name: group_by_task_type(rows) for split_name, rows in raw.items()}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    data = download_and_cache()
    print_metadata_report(data)
