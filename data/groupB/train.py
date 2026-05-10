"""
Group B – DARS Training Pipeline (ALFWorld Strategic Learning)
================================================================
Trains DARS's Utility (U) and Predictive Value (P) components using
ALFWorld tasks, with:

  - **PDDL-grounded utility feedback** (goal-state analysis, not hardcoded)
  - **Contextual weight shifting** (U=0.40, P=0.30 > R=0.15, F=0.15)
  - **Negative feedback** via update_utility(success=False)
  - **Virtual clock** for time-jump simulation (zero DB-write overhead)
  - **Memory deduplication** (concepts/strategies stored once, frequency grows)

Run standalone:  python -m data.groupB.train [--per-type N]
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Set

import numpy as np

from config.settings import DARSConfig
from core.layer_d.embedding import EmbeddingEngine
from core.layer_d.storage import MemoryVault
from data.groupB.extractor import (
    ProcessedTask,
    extract_all,
    FEEDBACK_THRESHOLD,
)

logger = logging.getLogger(__name__)

INDEXING_WAIT = 0.5
TASK_GAP_HOURS = 24
COLLECTION_PREFIX = "dars_alfworld_train"

GROUPB_WEIGHTS = {
    "WEIGHT_RECENCY": 0.15,
    "WEIGHT_FREQUENCY": 0.15,
    "WEIGHT_UTILITY": 0.40,
    "WEIGHT_PREDICTIVE": 0.30,
}


def _cosine(a, b) -> float:
    a, b = np.array(a), np.array(b)
    d = np.dot(a, b)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(d / n) if n > 0 else 0.0


def _compute_relevance(
    memory_text: str,
    goal_objects: Set[str],
    goal_actions: Set[str],
    goal_description: str,
    embedder: EmbeddingEngine,
    threshold: float = FEEDBACK_THRESHOLD,
) -> bool:
    """PDDL-grounded relevance check for utility feedback.

    A memory is 'useful' if it is both:
      1. Semantically related to the goal (cosine >= threshold)
      2. References an object or action mentioned in the goal predicates

    This is generalizable: the same function works for ANY task type
    by reading objects/actions from the parsed PDDL goal.
    """
    mem_emb = embedder.encode(memory_text)
    goal_emb = embedder.encode(goal_description)
    sim = _cosine(mem_emb, goal_emb)

    if sim < threshold:
        return False

    text_lower = memory_text.lower()
    for obj in goal_objects:
        if obj.lower() in text_lower:
            return True
    for act in goal_actions:
        if act.lower() in text_lower:
            return True

    return False


class GroupBTrainer:
    """Runs the full Group B training pipeline on extracted ALFWorld tasks.

    Uses a virtual clock to simulate temporal decay without patching
    every memory's recency on each time jump (O(1) instead of O(n)).
    """

    def __init__(self, collection_name: Optional[str] = None):
        self.collection_name = (
            collection_name or f"{COLLECTION_PREFIX}_{int(time.time())}"
        )
        self.vault = MemoryVault(collection_name=self.collection_name)
        self.embedder = EmbeddingEngine()
        self.vault.initialize_collection(recreate=True)
        self._pid_map: Dict[str, str] = {}
        self._freq_cache: Dict[str, int] = {}
        self._sim_time: float = time.time()

    def _ingest_memories(
        self, memories: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """Store task memories with concept/strategy deduplication.

        Concepts and strategies are stored once; re-encounters boost frequency.
        Instances are always stored (they're task-specific via tags).
        """
        ingested: Dict[str, List[str]] = {
            "instance": [],
            "concept": [],
            "strategy": [],
            "goal": [],
        }

        for mem in memories:
            mem_type = mem["mem_type"]
            text = mem["text"]

            if mem_type in ("concept", "strategy"):
                if text in self._pid_map:
                    pid = self._pid_map[text]
                    try:
                        old_freq = self._freq_cache.get(pid, 1)
                        new_freq = old_freq + 1
                        self._freq_cache[pid] = new_freq
                        self.vault.patch_payload(pid, {
                            "frequency": new_freq,
                            "recency": self._sim_time,
                        })
                    except Exception as exc:
                        logger.debug(
                            "Freq/recency boost failed for %s: %s", pid, exc
                        )
                    ingested[mem_type].append(pid)
                    continue

            pid = self.vault.store_memory(
                text=text,
                source="alfworld",
                tags=mem.get("tags", []),
            )
            self.vault.patch_payload(pid, {"recency": self._sim_time})
            self._pid_map[text] = pid
            self._freq_cache[pid] = 1
            ingested[mem_type].append(pid)

        return ingested

    def _simulate_time_jump(self, hours: float = TASK_GAP_HOURS):
        """Advance the virtual clock.  Zero DB writes."""
        self._sim_time += hours * 3600

    def _run_feedback(
        self, task: ProcessedTask
    ) -> Dict[str, int]:
        """Run PDDL-grounded feedback loop for a task.

        Computes all updates locally from already-loaded payloads,
        then writes each memory's changes in a single patch_payload
        call (1 API write per memory, not 3-5).
        """
        feedback: Dict[str, int] = {}

        results = self.vault.semantic_search(
            task.goal_description, top_k=20
        )
        if not results:
            return feedback

        goal_emb = self.embedder.encode(task.goal_description)

        for mem in results:
            mem_emb = self.embedder.encode(mem.payload.text_content)
            sim = _cosine(mem_emb, goal_emb)

            is_relevant = False
            if sim >= FEEDBACK_THRESHOLD:
                text_lower = mem.payload.text_content.lower()
                for obj in task.goal_objects:
                    if obj.lower() in text_lower:
                        is_relevant = True
                        break
                if not is_relevant:
                    for act in task.goal_actions:
                        if act.lower() in text_lower:
                            is_relevant = True
                            break

            payload = mem.payload
            updates: Dict[str, Any] = {}

            if is_relevant:
                new_sc = payload.success_count + 1
                new_util = new_sc / (new_sc + payload.failure_count + 1)
                new_freq = payload.frequency + 1
                updates = {
                    "success_count": new_sc,
                    "utility": new_util,
                    "frequency": new_freq,
                    "recency": self._sim_time,
                }
                feedback[mem.point_id] = 1
            else:
                new_fc = payload.failure_count + 1
                new_util = payload.success_count / (
                    payload.success_count + new_fc + 1
                )
                updates = {
                    "failure_count": new_fc,
                    "utility": new_util,
                }
                feedback[mem.point_id] = -1

            try:
                self.vault.patch_payload(mem.point_id, updates)
            except Exception as exc:
                logger.warning(
                    "Feedback update failed for %s: %s", mem.point_id, exc
                )

        return feedback

    def train_task_sequence(
        self, tasks: List[ProcessedTask]
    ) -> Dict[str, Any]:
        """Train on a sequence of tasks (typically same task_type).

        Task 0: ingest only (cold start).
        Tasks 1..N: ingest + feedback loop, with time jumps between.
        """
        all_feedback: Dict[str, int] = {}
        total_ingested = {"instance": 0, "concept": 0, "strategy": 0, "goal": 0}

        for i, task in enumerate(tasks):
            ingested = self._ingest_memories(task.memories)
            for mt, pids in ingested.items():
                total_ingested[mt] += len(pids)
            time.sleep(INDEXING_WAIT)

            if i > 0:
                fb = self._run_feedback(task)
                for pid, signal in fb.items():
                    all_feedback[pid] = all_feedback.get(pid, 0) + signal

            if i < len(tasks) - 1:
                self._simulate_time_jump(TASK_GAP_HOURS)

        final_mems = self.vault.get_all_memories(limit=2000)
        scores_by_type: Dict[str, List[float]] = {
            "instance": [],
            "concept": [],
            "strategy": [],
            "goal": [],
        }
        for mem in final_mems:
            dars = self.vault.compute_dars_score(
                mem.payload.to_dict(), current_time=self._sim_time
            )
            for tag in mem.payload.tags:
                if tag.startswith("mem_type:"):
                    mt = tag.split(":")[1]
                    if mt in scores_by_type:
                        scores_by_type[mt].append(dars)
                    break

        return {
            "num_tasks": len(tasks),
            "total_memories": len(final_mems),
            "ingested": total_ingested,
            "feedback_positive": sum(1 for v in all_feedback.values() if v > 0),
            "feedback_negative": sum(1 for v in all_feedback.values() if v < 0),
            "scores_by_type": {
                mt: {
                    "mean": float(np.mean(s)) if s else None,
                    "std": float(np.std(s)) if s else None,
                    "count": len(s),
                }
                for mt, s in scores_by_type.items()
            },
        }

    @property
    def simulated_time(self) -> float:
        """Current virtual clock value for evaluation."""
        return self._sim_time

    def cleanup(self):
        try:
            self.vault.delete_collection()
        except Exception:
            pass


def run_training(
    max_per_type: int = 10,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run Group B training across all task types.

    Sets contextual weights (U=0.40, P=0.30) for the duration of training.
    """
    original_weights = {
        "WEIGHT_RECENCY": DARSConfig.WEIGHT_RECENCY,
        "WEIGHT_FREQUENCY": DARSConfig.WEIGHT_FREQUENCY,
        "WEIGHT_UTILITY": DARSConfig.WEIGHT_UTILITY,
        "WEIGHT_PREDICTIVE": DARSConfig.WEIGHT_PREDICTIVE,
    }
    original_group = DARSConfig.TRAINING_GROUP
    DARSConfig._goal_vector_cache = None

    try:
        for k, v in GROUPB_WEIGHTS.items():
            setattr(DARSConfig, k, v)
        DARSConfig.validate_and_normalize()
        DARSConfig.TRAINING_GROUP = "ALFWorld"
        DARSConfig._goal_vector_cache = None

        cache = __import__("pathlib").Path(__file__).parent / "cache" / "alfworld_train.json"
        with open(cache, "r", encoding="utf-8") as f:
            rows = json.load(f)

        logger.info(
            "Extracting tasks (max_per_type=%d) ...", max_per_type
        )
        extracted = extract_all(rows, max_per_type=max_per_type)

        trainer = GroupBTrainer()
        results_by_type: Dict[str, Dict[str, Any]] = {}

        for ttype, tasks in sorted(extracted.items()):
            logger.info(
                "Training task_type=%s (%d tasks)...", ttype, len(tasks)
            )
            result = trainer.train_task_sequence(tasks)
            results_by_type[ttype] = result

            if verbose:
                s = result["scores_by_type"]
                strat_mean = (
                    f"{s['strategy']['mean']:.3f}"
                    if s["strategy"]["mean"] is not None
                    else "N/A"
                )
                concept_mean = (
                    f"{s['concept']['mean']:.3f}"
                    if s["concept"]["mean"] is not None
                    else "N/A"
                )
                inst_mean = (
                    f"{s['instance']['mean']:.3f}"
                    if s["instance"]["mean"] is not None
                    else "N/A"
                )
                print(
                    f"  {ttype}: mems={result['total_memories']} "
                    f"fb+={result['feedback_positive']} "
                    f"fb-={result['feedback_negative']} "
                    f"DARS(strat)={strat_mean} "
                    f"DARS(concept)={concept_mean} "
                    f"DARS(instance)={inst_mean}"
                )

        _print_training_summary(results_by_type)

        return {
            "trainer": trainer,
            "results_by_type": results_by_type,
            "extracted": extracted,
        }

    finally:
        for k, v in original_weights.items():
            setattr(DARSConfig, k, v)
        DARSConfig.validate_and_normalize()
        DARSConfig.TRAINING_GROUP = original_group
        DARSConfig._goal_vector_cache = None


def _print_training_summary(
    results_by_type: Dict[str, Dict[str, Any]],
) -> None:
    all_strategy: List[float] = []
    all_concept: List[float] = []
    all_instance: List[float] = []
    all_goal: List[float] = []

    for result in results_by_type.values():
        sbt = result["scores_by_type"]
        if sbt["strategy"]["mean"] is not None:
            all_strategy.append(sbt["strategy"]["mean"])
        if sbt["concept"]["mean"] is not None:
            all_concept.append(sbt["concept"]["mean"])
        if sbt["instance"]["mean"] is not None:
            all_instance.append(sbt["instance"]["mean"])
        if sbt["goal"]["mean"] is not None:
            all_goal.append(sbt["goal"]["mean"])

    print("\n" + "=" * 70)
    print("  GROUP B TRAINING SUMMARY (Contextual Weights: U=0.40, P=0.30)")
    print("=" * 70)
    print(f"  Task types trained:  {len(results_by_type)}")
    total_mems = sum(r["total_memories"] for r in results_by_type.values())
    total_fb_pos = sum(r["feedback_positive"] for r in results_by_type.values())
    total_fb_neg = sum(r["feedback_negative"] for r in results_by_type.values())
    print(f"  Total memories:      {total_mems}")
    print(f"  Feedback signals:    +{total_fb_pos}  / -{total_fb_neg}")

    print(f"\n  --- Mean DARS by Memory Type (across all task types) ---")
    for label, scores in [
        ("Strategy", all_strategy),
        ("Concept", all_concept),
        ("Goal", all_goal),
        ("Instance", all_instance),
    ]:
        if scores:
            print(
                f"    {label:12s}: mean={np.mean(scores):.4f}  "
                f"std={np.std(scores):.4f}  n={len(scores)}"
            )
        else:
            print(f"    {label:12s}: no data")

    if all_strategy and all_instance:
        delta = np.mean(all_strategy) - np.mean(all_instance)
        print(f"\n  --- Utility Separation ---")
        print(f"    Mean(strategy) - Mean(instance) = {delta:+.4f}")
        if delta > 0:
            print(
                f"    RESULT: Strategies score HIGHER than instances "
                f"(utility learning works)"
            )
        else:
            print(
                f"    RESULT: Strategies score LOWER than instances "
                f"(unexpected - investigate)"
            )

    print("=" * 70 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Group B ALFWorld Training")
    parser.add_argument(
        "--per-type",
        type=int,
        default=10,
        help="Number of tasks per task_type",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_training(max_per_type=args.per_type)
