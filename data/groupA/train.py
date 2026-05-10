"""
Group A – DARS Training Pipeline (MSC Temporal Learning)
=========================================================
Ingests sessions 0-2 into DARS for each dialogue, simulating
multi-session temporal learning with time jumps and feedback loops.

Run standalone:  python -m data.groupA.train [--dialogues N]
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from config.settings import DARSConfig
from core.layer_d.embedding import EmbeddingEngine
from core.layer_d.storage import MemoryVault
from data.groupA.extractor import (
    ProcessedDialogue,
    extract_all,
    FEEDBACK_MATCH_THRESHOLD,
)

logger = logging.getLogger(__name__)

INDEXING_WAIT = 0.5
SESSION_GAP_HOURS = 24
COLLECTION_PREFIX = "dars_msc_train"


def _cosine(a, b) -> float:
    a, b = np.array(a), np.array(b)
    d = np.dot(a, b)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(d / n) if n > 0 else 0.0


class GroupATrainer:
    """Runs the full Group A training pipeline on extracted dialogues."""

    def __init__(self, collection_name: Optional[str] = None):
        self.collection_name = collection_name or f"{COLLECTION_PREFIX}_{int(time.time())}"
        self.vault = MemoryVault(collection_name=self.collection_name)
        self.embedder = EmbeddingEngine()
        self.vault.initialize_collection(recreate=True)
        self._pid_map: Dict[str, str] = {}

    def _ingest_memories(
        self, memories: List[Dict[str, Any]], session_filter: Optional[int] = None
    ) -> List[str]:
        """Store persona-fact memories, optionally filtering by first_session."""
        pids = []
        for mem in memories:
            if session_filter is not None and mem["first_session"] != session_filter:
                continue
            if mem["text"] in self._pid_map:
                continue
            pid = self.vault.store_memory(
                text=mem["text"],
                source=mem["source"],
                tags=mem["tags"],
                vector_override=mem.get("centroid_embedding"),
            )
            self._pid_map[mem["text"]] = pid
            pids.append(pid)
        return pids

    def _simulate_time_jump(self, hours: float = SESSION_GAP_HOURS):
        """Shift all memories' recency backward to simulate passage of time."""
        shift = hours * 3600
        all_mems = self.vault.get_all_memories(limit=500)
        for mem in all_mems:
            new_recency = mem.payload.recency - shift
            self.vault.patch_payload(mem.point_id, {"recency": new_recency})

    def _run_feedback(
        self,
        interactions: List[Dict[str, Any]],
        session_id: int,
    ) -> Dict[str, int]:
        """Run strict-match feedback loop for dialogue turns in a session.

        Returns dict of {point_id: times_matched} for diagnostics.
        """
        match_counts: Dict[str, int] = {}
        session_turns = [t for t in interactions if t["session_id"] == session_id]

        all_mems = self.vault.get_all_memories(limit=500, with_vectors=False)
        if not all_mems:
            return match_counts

        for turn in session_turns:
            speaker = turn["speaker"]
            query_emb = self.embedder.encode(turn["text"])

            results = self.vault.semantic_search(turn["text"], top_k=10)

            for mem in results:
                if f"speaker:{speaker}" not in mem.payload.tags:
                    continue
                mem_emb = self.embedder.encode(mem.payload.text_content)
                sim = _cosine(query_emb, mem_emb)
                if sim < FEEDBACK_MATCH_THRESHOLD:
                    continue
                try:
                    self.vault.update_utility(mem.point_id, success=True)
                    self.vault.increment_frequency(mem.point_id)
                    self.vault.update_recency(mem.point_id)
                    match_counts[mem.point_id] = match_counts.get(mem.point_id, 0) + 1
                except Exception as exc:
                    logger.warning("Feedback update failed for %s: %s", mem.point_id, exc)

        return match_counts

    def train_dialogue(self, dialogue: ProcessedDialogue) -> Dict[str, Any]:
        """Run the full train pipeline for a single dialogue."""
        self.vault.initialize_collection(recreate=True)
        self._pid_map.clear()

        s0_pids = self._ingest_memories(dialogue.memories, session_filter=0)
        time.sleep(INDEXING_WAIT)

        self._simulate_time_jump(SESSION_GAP_HOURS)

        new_s1 = self._ingest_memories(dialogue.memories, session_filter=1)
        time.sleep(INDEXING_WAIT)
        s1_matches = self._run_feedback(dialogue.interactions, session_id=1)

        self._simulate_time_jump(SESSION_GAP_HOURS)

        new_s2 = self._ingest_memories(dialogue.memories, session_filter=2)
        time.sleep(INDEXING_WAIT)
        s2_matches = self._run_feedback(dialogue.interactions, session_id=2)

        final_memories = self.vault.get_all_memories(limit=500)
        scores: Dict[str, Dict[str, Any]] = {}
        for mem in final_memories:
            dars = self.vault.compute_dars_score(mem.payload.to_dict())
            scores[mem.point_id] = {
                "text": mem.payload.text_content[:80],
                "dars_score": dars,
                "frequency": mem.payload.frequency,
                "utility": mem.payload.utility,
                "predictive": mem.payload.predictive,
                "classification": self.vault.classify_memory(dars),
            }

        high_freq_scores = []
        low_freq_scores = []
        for mem_data in dialogue.memories:
            pid = self._pid_map.get(mem_data["text"])
            if pid and pid in scores:
                if mem_data["frequency_ground_truth"] >= 3:
                    high_freq_scores.append(scores[pid]["dars_score"])
                elif mem_data["frequency_ground_truth"] == 1:
                    low_freq_scores.append(scores[pid]["dars_score"])

        return {
            "dialogue_id": dialogue.dialogue_id,
            "total_memories": len(final_memories),
            "s0_ingested": len(s0_pids),
            "s1_new": len(new_s1),
            "s2_new": len(new_s2),
            "s1_feedback_matches": sum(s1_matches.values()),
            "s2_feedback_matches": sum(s2_matches.values()),
            "avg_dars_high_freq": np.mean(high_freq_scores) if high_freq_scores else None,
            "avg_dars_low_freq": np.mean(low_freq_scores) if low_freq_scores else None,
            "scores": scores,
        }

    def cleanup(self):
        try:
            self.vault.delete_collection()
        except Exception:
            pass


def run_training(
    max_dialogues: int = 50,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Run Group A training across multiple dialogues."""
    original_group = DARSConfig.TRAINING_GROUP
    DARSConfig.TRAINING_GROUP = "MSC"
    DARSConfig._goal_vector_cache = None

    try:
        logger.info("Extracting dialogues (max=%d)...", max_dialogues)
        dialogues = extract_all(max_dialogues=max_dialogues)
        if not dialogues:
            logger.error("No complete dialogues found.")
            return []

        trainer = GroupATrainer()
        results = []

        for i, d in enumerate(dialogues):
            logger.info(
                "Training dialogue %d/%d (id=%d)...",
                i + 1, len(dialogues), d.dialogue_id,
            )
            result = trainer.train_dialogue(d)
            results.append(result)

            if verbose:
                hf = result["avg_dars_high_freq"]
                lf = result["avg_dars_low_freq"]
                hf_str = f"{hf:.3f}" if hf is not None else "N/A"
                lf_str = f"{lf:.3f}" if lf is not None else "N/A"
                print(
                    f"  [{i+1}/{len(dialogues)}] dialogue={d.dialogue_id} "
                    f"mems={result['total_memories']} "
                    f"feedback_s1={result['s1_feedback_matches']} "
                    f"feedback_s2={result['s2_feedback_matches']} "
                    f"DARS(high)={hf_str} DARS(low)={lf_str}"
                )

        trainer.cleanup()
        _print_training_summary(results)
        return results

    finally:
        DARSConfig.TRAINING_GROUP = original_group
        DARSConfig._goal_vector_cache = None


def _print_training_summary(results: List[Dict[str, Any]]) -> None:
    high_scores = [r["avg_dars_high_freq"] for r in results if r["avg_dars_high_freq"] is not None]
    low_scores = [r["avg_dars_low_freq"] for r in results if r["avg_dars_low_freq"] is not None]

    print("\n" + "=" * 70)
    print("  GROUP A TRAINING SUMMARY")
    print("=" * 70)
    print(f"  Dialogues trained:           {len(results)}")
    print(f"  Dialogues with high-freq:    {len(high_scores)}")
    print(f"  Dialogues with low-freq:     {len(low_scores)}")

    if high_scores:
        print(f"\n  --- DARS Scores (High-Freq Facts, >=3 sessions) ---")
        print(f"  Mean:   {np.mean(high_scores):.4f}")
        print(f"  Median: {np.median(high_scores):.4f}")
        print(f"  Min:    {np.min(high_scores):.4f}")
        print(f"  Max:    {np.max(high_scores):.4f}")

    if low_scores:
        print(f"\n  --- DARS Scores (Low-Freq Facts, 1 session) ---")
        print(f"  Mean:   {np.mean(low_scores):.4f}")
        print(f"  Median: {np.median(low_scores):.4f}")
        print(f"  Min:    {np.min(low_scores):.4f}")
        print(f"  Max:    {np.max(low_scores):.4f}")

    if high_scores and low_scores:
        diff = np.mean(high_scores) - np.mean(low_scores)
        print(f"\n  --- Score Separation ---")
        print(f"  Mean(high) - Mean(low) = {diff:+.4f}")
        if diff > 0:
            print(f"  RESULT: High-freq facts score HIGHER (temporal learning works)")
        else:
            print(f"  RESULT: High-freq facts score LOWER (unexpected - investigate)")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Group A MSC Training")
    parser.add_argument("--dialogues", type=int, default=5, help="Number of dialogues to train")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_training(max_dialogues=args.dialogues)
