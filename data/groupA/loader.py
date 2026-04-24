"""
Group A – MSC Dataset Loader
=============================
Downloads nayohan/multi_session_chat from HuggingFace, caches locally,
and prints a metadata report for research validation.

Run standalone:  python -m data.groupA.loader
"""

from __future__ import annotations

import json
import logging
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_FILE = CACHE_DIR / "msc_train.json"

DATASET_NAME = "nayohan/multi_session_chat"
SPLIT = "train"


def download_and_cache(force: bool = False) -> List[Dict[str, Any]]:
    """Download MSC train split from HuggingFace and cache as JSON."""
    if CACHE_FILE.exists() and not force:
        logger.info("Loading cached MSC dataset from %s", CACHE_FILE)
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    logger.info("Downloading %s (split=%s) from HuggingFace...", DATASET_NAME, SPLIT)
    from datasets import load_dataset

    ds = load_dataset(DATASET_NAME, split=SPLIT)
    rows = [dict(row) for row in ds]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    logger.info("Cached %d rows to %s", len(rows), CACHE_FILE)
    return rows


def group_by_dialogue(rows: List[Dict[str, Any]]) -> Dict[int, Dict[int, Dict[str, Any]]]:
    """Group rows into {dialoug_id: {session_id: row}}."""
    grouped: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[row["dialoug_id"]][row["session_id"]] = row
    return dict(grouped)


def print_metadata_report(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyse and print a full metadata report. Returns the stats dict."""
    dialogues = group_by_dialogue(rows)
    total_dialogues = len(dialogues)
    total_rows = len(rows)

    sessions_per_dialogue = [len(sessions) for sessions in dialogues.values()]
    complete_dialogues = sum(1 for s in sessions_per_dialogue if s == 4)

    persona1_counts: Dict[int, List[int]] = defaultdict(list)
    persona2_counts: Dict[int, List[int]] = defaultdict(list)
    dialogue_turn_counts: Dict[int, List[int]] = defaultdict(list)

    for did, sessions in dialogues.items():
        for sid, row in sessions.items():
            persona1_counts[sid].append(len(row.get("persona1", [])))
            persona2_counts[sid].append(len(row.get("persona2", [])))
            dialogue_turn_counts[sid].append(len(row.get("dialogue", [])))

    stats = {
        "total_rows": total_rows,
        "total_dialogues": total_dialogues,
        "complete_dialogues_4_sessions": complete_dialogues,
        "incomplete_dialogues": total_dialogues - complete_dialogues,
    }

    print("\n" + "=" * 70)
    print("  MSC DATASET – METADATA REPORT")
    print("=" * 70)
    print(f"  Total rows:                  {total_rows:,}")
    print(f"  Unique dialogue pairs:       {total_dialogues:,}")
    print(f"  Complete (4 sessions):       {complete_dialogues:,}")
    print(f"  Incomplete:                  {total_dialogues - complete_dialogues:,}")

    print("\n  --- Persona Growth Across Sessions ---")
    for sid in sorted(persona1_counts.keys()):
        p1 = persona1_counts[sid]
        p2 = persona2_counts[sid]
        turns = dialogue_turn_counts[sid]
        p1_avg = statistics.mean(p1) if p1 else 0
        p2_avg = statistics.mean(p2) if p2 else 0
        turns_avg = statistics.mean(turns) if turns else 0
        count = len(p1)
        print(
            f"  Session {sid}:  dialogues={count:,}  "
            f"persona1_avg={p1_avg:.1f}  persona2_avg={p2_avg:.1f}  "
            f"turns_avg={turns_avg:.1f}"
        )
        stats[f"session_{sid}_count"] = count
        stats[f"session_{sid}_p1_avg"] = round(p1_avg, 2)
        stats[f"session_{sid}_p2_avg"] = round(p2_avg, 2)
        stats[f"session_{sid}_turns_avg"] = round(turns_avg, 2)

    all_turns = []
    for sid_list in dialogue_turn_counts.values():
        all_turns.extend(sid_list)
    if all_turns:
        print(f"\n  --- Dialogue Turn Distribution ---")
        print(f"  Min turns:  {min(all_turns)}")
        print(f"  Max turns:  {max(all_turns)}")
        print(f"  Mean turns: {statistics.mean(all_turns):.1f}")
        print(f"  Median:     {statistics.median(all_turns):.1f}")

    print("=" * 70 + "\n")
    return stats


def load_msc(force_download: bool = False) -> Dict[int, Dict[int, Dict[str, Any]]]:
    """Main entry: load + group. Returns {dialoug_id: {session_id: row}}."""
    rows = download_and_cache(force=force_download)
    return group_by_dialogue(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    rows = download_and_cache()
    print_metadata_report(rows)
