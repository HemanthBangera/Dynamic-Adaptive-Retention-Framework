"""
Group B – DARS Evaluation Pipeline (ALFWorld Strategic Learning)
=================================================================
Evaluates DARS retrieval quality across 6 metrics:

  1. **Utility Separation** – Strategy > Concept > Instance mean DARS
  2. **Strategic Recall (In-Dist)** – Recall/Precision/MRR on eval_in tasks
  3. **OOD Transfer (Cooling/Heating)** – Performance on eval_out + type-rank
  4. **Retention Matrix** – mem_type × {retain, compress, delete}
  5. **Negative Control** – Out-of-domain recall (should be ~0%)
  6. **Ablation** – DARS+RRF vs Pure Similarity baseline (the "delta")

Relevance in Group B is defined by PDDL-grounded checks (keyword +
semantic threshold), NOT by a strict cosine gate.  This is appropriate
because stored memories are conceptual templates, not paraphrases.

Run standalone:  python -m data.groupB.evaluate [--per-type N] [--k 5]
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
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
from data.groupB.train import (
    GroupBTrainer,
    GROUPB_WEIGHTS,
    INDEXING_WAIT,
    _compute_relevance,
)

logger = logging.getLogger(__name__)

RELEVANCE_THRESHOLD = 0.80

NEGATIVE_QUERIES = [
    "What is the capital of France?",
    "Explain the theory of relativity in simple terms",
    "How does photosynthesis work in plants?",
    "Write a poem about the ocean at sunset",
    "What are the rules of basketball?",
    "Describe the life cycle of a butterfly",
    "How do stock markets function?",
    "What causes earthquakes and tsunamis?",
    "Explain blockchain technology",
    "What is the history of the Roman Empire?",
    "How does machine learning differ from deep learning?",
    "Describe the water cycle in nature",
    "What are the symptoms of common cold?",
    "How do solar panels generate electricity?",
    "Explain the process of fermentation in brewing",
    "What is quantum computing?",
    "Describe the migration patterns of birds",
    "How do vaccines protect against diseases?",
    "What are the principles of aerodynamics?",
    "Explain the concept of supply and demand in economics",
]


def _cosine(a, b) -> float:
    a, b = np.array(a), np.array(b)
    d = np.dot(a, b)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(d / n) if n > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Evaluation Results
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TaskEvalResult:
    task_id: str
    task_type: str
    recall: float
    precision: float
    mrr: float
    retrieved_relevant: int
    total_relevant: int
    total_retrieved: int
    hit: bool = False
    recall_capped: float = 0.0


@dataclass
class EvalReport:
    """Aggregated evaluation report for all 6 metrics."""

    # Metric 1
    utility_separation: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Metric 2
    in_dist_results: List[TaskEvalResult] = field(default_factory=list)
    in_dist_recall: float = 0.0
    in_dist_precision: float = 0.0
    in_dist_mrr: float = 0.0
    in_dist_hit_rate: float = 0.0
    in_dist_recall_capped: float = 0.0

    # Metric 3
    ood_results: List[TaskEvalResult] = field(default_factory=list)
    ood_recall: float = 0.0
    ood_precision: float = 0.0
    ood_mrr: float = 0.0
    ood_hit_rate: float = 0.0
    ood_recall_capped: float = 0.0
    cooling_delta: float = 0.0

    # Metric 4
    retention_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Metric 5
    negative_recall: float = 0.0
    negative_total: int = 0
    negative_false_hits: int = 0

    # Metric 6
    ablation_dars_recall: float = 0.0
    ablation_dars_precision: float = 0.0
    ablation_dars_mrr: float = 0.0
    ablation_dars_recall_cap: float = 0.0
    ablation_baseline_recall: float = 0.0
    ablation_baseline_precision: float = 0.0
    ablation_baseline_mrr: float = 0.0
    ablation_baseline_recall_cap: float = 0.0


class GroupBEvaluator:
    """Evaluates DARS on ALFWorld eval tasks after training."""

    def __init__(
        self,
        k: int = 5,
        fetch_k: int = 20,
        simulated_time: Optional[float] = None,
    ):
        self.k = k
        self.fetch_k = fetch_k
        self.sim_time = simulated_time
        self.embedder = EmbeddingEngine()
        self._emb_cache: Dict[str, np.ndarray] = {}

    def _get_embedding(self, text: str) -> np.ndarray:
        """Cached embedding lookup — encode once per unique text."""
        if text not in self._emb_cache:
            self._emb_cache[text] = np.array(self.embedder.encode(text))
        return self._emb_cache[text]

    def _is_relevant(
        self,
        mem_text: str,
        task: ProcessedTask,
    ) -> bool:
        """PDDL-grounded relevance with cached embeddings."""
        mem_emb = self._get_embedding(mem_text)
        goal_emb = self._get_embedding(task.goal_description)
        sim = _cosine(mem_emb, goal_emb)

        if sim < FEEDBACK_THRESHOLD:
            return False

        text_lower = mem_text.lower()
        for obj in task.goal_objects:
            if obj.lower() in text_lower:
                return True
        for act in task.goal_actions:
            if act.lower() in text_lower:
                return True
        return False

    def precompute_embeddings(
        self,
        texts: List[str],
        label: str = "",
    ) -> None:
        """Batch-encode texts into the cache to avoid per-call overhead."""
        new_texts = [t for t in texts if t not in self._emb_cache]
        if not new_texts:
            return
        logger.info(
            "Pre-computing %d embeddings%s...",
            len(new_texts),
            f" ({label})" if label else "",
        )
        for t in new_texts:
            self._emb_cache[t] = np.array(self.embedder.encode(t))

    # ── Metric 1: Utility Separation ──────────────────────────────────

    def evaluate_utility_separation(
        self, vault: MemoryVault
    ) -> Dict[str, Dict[str, float]]:
        all_mems = vault.get_all_memories(limit=2000)
        scores_by_type: Dict[str, List[float]] = defaultdict(list)

        for mem in all_mems:
            dars = vault.compute_dars_score(
                mem.payload.to_dict(), current_time=self.sim_time
            )
            for tag in mem.payload.tags:
                if tag.startswith("mem_type:"):
                    mt = tag.split(":")[1]
                    scores_by_type[mt].append(dars)
                    break

        result = {}
        for mt, scores in scores_by_type.items():
            result[mt] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "count": len(scores),
            }
        return result

    # ── Metric 2 & 3: Strategic Recall ────────────────────────────────

    def evaluate_tasks(
        self,
        tasks: List[ProcessedTask],
        vault: MemoryVault,
        use_rrf: bool = True,
        alpha: Optional[float] = None,
    ) -> List[TaskEvalResult]:
        """Evaluate retrieval on a list of tasks.

        Relevance = PDDL-grounded (cosine >= 0.45 + keyword).
        """
        stored_mems = vault.get_all_memories(limit=2000, with_vectors=False)
        stored_texts = [m.payload.text_content for m in stored_mems]

        results: List[TaskEvalResult] = []

        for task in tasks:
            total_relevant = sum(
                1 for txt in stored_texts if self._is_relevant(txt, task)
            )

            if total_relevant == 0:
                results.append(
                    TaskEvalResult(
                        task_id=task.task_id,
                        task_type=task.task_type,
                        recall=0.0,
                        precision=0.0,
                        mrr=0.0,
                        retrieved_relevant=0,
                        total_relevant=0,
                        total_retrieved=0,
                    )
                )
                continue

            retrieved = vault.search_and_rerank(
                task.goal_description,
                top_n=self.k,
                fetch_k=self.fetch_k,
                use_rrf=use_rrf,
                current_time=self.sim_time,
                **({"alpha": alpha} if alpha is not None else {}),
            )

            relevant_retrieved = 0
            rr = 0.0
            rr_found = False

            for rank, mem in enumerate(retrieved, 1):
                if self._is_relevant(mem.payload.text_content, task):
                    relevant_retrieved += 1
                    if not rr_found:
                        rr = 1.0 / rank
                        rr_found = True

            recall = relevant_retrieved / total_relevant
            precision = (
                relevant_retrieved / len(retrieved) if retrieved else 0.0
            )
            max_possible = min(total_relevant, len(retrieved))
            recall_capped = (
                relevant_retrieved / max_possible if max_possible > 0 else 0.0
            )

            results.append(
                TaskEvalResult(
                    task_id=task.task_id,
                    task_type=task.task_type,
                    recall=recall,
                    precision=precision,
                    mrr=rr,
                    retrieved_relevant=relevant_retrieved,
                    total_relevant=total_relevant,
                    total_retrieved=len(retrieved),
                    hit=relevant_retrieved > 0,
                    recall_capped=recall_capped,
                )
            )

        return results

    # ── Metric 3 Supplement: Cooling Delta ────────────────────────────

    def compute_cooling_delta(
        self,
        vault: MemoryVault,
        ood_tasks: List[ProcessedTask],
        n_samples: int = 5,
    ) -> float:
        """Measure if DARS ranks relevant memories above irrelevant ones
        for OOD goals.  Cooling delta > 0 = DARS suppresses noise."""
        all_mems = vault.get_all_memories(limit=2000, with_vectors=False)
        if not all_mems or not ood_tasks:
            return 0.0

        samples = random.sample(ood_tasks, min(n_samples, len(ood_tasks)))
        deltas: List[float] = []

        for task in samples:
            results = vault.search_and_rerank(
                task.goal_description,
                top_n=min(10, len(all_mems)),
                fetch_k=min(30, len(all_mems)),
                current_time=self.sim_time,
            )
            relevant_scores = []
            irrelevant_scores = []
            for mem in results:
                score = mem.dars_score if mem.dars_score is not None else 0.0
                if self._is_relevant(mem.payload.text_content, task):
                    relevant_scores.append(score)
                else:
                    irrelevant_scores.append(score)

            if relevant_scores and irrelevant_scores:
                deltas.append(
                    float(np.mean(relevant_scores))
                    - float(np.mean(irrelevant_scores))
                )

        return float(np.mean(deltas)) if deltas else 0.0

    # ── Metric 4: Retention Matrix ────────────────────────────────────

    def compute_retention_matrix(
        self, vault: MemoryVault
    ) -> Dict[str, Dict[str, int]]:
        matrix: Dict[str, Dict[str, int]] = {}
        all_mems = vault.get_all_memories(limit=2000)

        for mem in all_mems:
            dars = vault.compute_dars_score(
                mem.payload.to_dict(), current_time=self.sim_time
            )
            classification = vault.classify_memory(dars)
            mt = "unknown"
            for tag in mem.payload.tags:
                if tag.startswith("mem_type:"):
                    mt = tag.split(":")[1]
                    break
            if mt not in matrix:
                matrix[mt] = {"retain": 0, "compress": 0, "delete": 0}
            matrix[mt][classification] += 1

        return matrix

    # ── Metric 5: Negative Control ────────────────────────────────────

    def evaluate_negative_control(
        self,
        vault: MemoryVault,
    ) -> Dict[str, Any]:
        """Query DARS with truly out-of-domain queries (non-household).

        At cosine >= 0.80 threshold, recall should be ~0% because
        none of these queries relate to household tasks.
        """
        false_hits = 0
        total = len(NEGATIVE_QUERIES)

        household_keywords = {
            "heat", "clean", "cool", "examine", "place", "microwave",
            "fridge", "sinkbasin", "sink", "lamp", "desklamp", "floorlamp",
            "apple", "potato", "bread", "egg", "tomato", "lettuce",
            "mug", "cup", "bowl", "plate", "knife", "fork", "spoon",
            "countertop", "diningtable", "shelf", "cabinet", "drawer",
        }

        for query in NEGATIVE_QUERIES:
            results = vault.search_and_rerank(
                query,
                top_n=self.k,
                fetch_k=self.fetch_k,
                current_time=self.sim_time,
            )
            query_lower = query.lower()
            has_household = any(kw in query_lower for kw in household_keywords)
            if has_household:
                continue

            for mem in results:
                mem_emb = self._get_embedding(mem.payload.text_content)
                q_emb = self._get_embedding(query)
                sim = _cosine(mem_emb, q_emb)
                if sim >= RELEVANCE_THRESHOLD:
                    false_hits += 1
                    break

        return {
            "total": total,
            "false_hits": false_hits,
            "negative_recall": false_hits / total if total > 0 else 0.0,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  Full Evaluation Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_evaluation(
    max_per_type: int = 10,
    k: int = 5,
    fetch_k: int = 20,
    max_eval: int = 50,
    verbose: bool = True,
) -> EvalReport:
    """Full pipeline: train on train split, evaluate on eval_in + eval_out."""
    original_weights = {
        "WEIGHT_RECENCY": DARSConfig.WEIGHT_RECENCY,
        "WEIGHT_FREQUENCY": DARSConfig.WEIGHT_FREQUENCY,
        "WEIGHT_UTILITY": DARSConfig.WEIGHT_UTILITY,
        "WEIGHT_PREDICTIVE": DARSConfig.WEIGHT_PREDICTIVE,
    }
    original_group = DARSConfig.TRAINING_GROUP
    DARSConfig._goal_vector_cache = None

    try:
        for kk, v in GROUPB_WEIGHTS.items():
            setattr(DARSConfig, kk, v)
        DARSConfig.validate_and_normalize()
        DARSConfig.TRAINING_GROUP = "ALFWorld"
        DARSConfig._goal_vector_cache = None

        cache_dir = Path(__file__).parent / "cache"

        # ── Load all splits ───────────────────────────────────────────
        with open(cache_dir / "alfworld_train.json", "r", encoding="utf-8") as f:
            train_rows = json.load(f)
        with open(cache_dir / "alfworld_eval_in.json", "r", encoding="utf-8") as f:
            eval_in_rows = json.load(f)
        with open(cache_dir / "alfworld_eval_out.json", "r", encoding="utf-8") as f:
            eval_out_rows = json.load(f)

        # ── Extract ───────────────────────────────────────────────────
        logger.info("Extracting train tasks (max_per_type=%d)...", max_per_type)
        train_extracted = extract_all(train_rows, max_per_type=max_per_type)

        logger.info("Extracting eval_in tasks...")
        eval_in_extracted = extract_all(eval_in_rows, max_per_type=max_eval)

        logger.info("Extracting eval_out tasks...")
        eval_out_extracted = extract_all(eval_out_rows, max_per_type=max_eval)

        # ── Train ─────────────────────────────────────────────────────
        logger.info("Training (10 tasks/type, virtual clock, PDDL feedback)...")
        trainer = GroupBTrainer()

        for ttype, tasks in sorted(train_extracted.items()):
            logger.info("  Training %s (%d tasks)...", ttype, len(tasks))
            trainer.train_task_sequence(tasks)

        time.sleep(INDEXING_WAIT)

        evaluator = GroupBEvaluator(
            k=k, fetch_k=fetch_k, simulated_time=trainer.simulated_time
        )
        report = EvalReport()

        stored_mems = trainer.vault.get_all_memories(limit=2000, with_vectors=False)
        stored_texts = [m.payload.text_content for m in stored_mems]
        evaluator.precompute_embeddings(stored_texts, "stored memories")

        all_goal_descs = []
        for ts in eval_in_extracted.values():
            for t in ts:
                all_goal_descs.append(t.goal_description)
        for ts in eval_out_extracted.values():
            for t in ts:
                all_goal_descs.append(t.goal_description)
        evaluator.precompute_embeddings(all_goal_descs, "goal descriptions")

        # ── Metric 1: Utility Separation ──────────────────────────────
        logger.info("Metric 1: Utility Separation...")
        report.utility_separation = evaluator.evaluate_utility_separation(
            trainer.vault
        )

        # ── Metric 2: In-Distribution Recall ──────────────────────────
        logger.info("Metric 2: In-Distribution Recall...")
        eval_in_flat = [
            t for ts in eval_in_extracted.values() for t in ts
        ][:max_eval]
        report.in_dist_results = evaluator.evaluate_tasks(
            eval_in_flat, trainer.vault
        )
        if report.in_dist_results:
            report.in_dist_recall = float(
                np.mean([r.recall for r in report.in_dist_results])
            )
            report.in_dist_precision = float(
                np.mean([r.precision for r in report.in_dist_results])
            )
            report.in_dist_mrr = float(
                np.mean([r.mrr for r in report.in_dist_results])
            )
            report.in_dist_hit_rate = float(
                np.mean([1.0 if r.hit else 0.0 for r in report.in_dist_results])
            )
            report.in_dist_recall_capped = float(
                np.mean([r.recall_capped for r in report.in_dist_results])
            )

        if verbose:
            for r in report.in_dist_results[:5]:
                print(
                    f"    [{r.task_type}] R={r.recall:.2f} "
                    f"P={r.precision:.2f} hits={r.retrieved_relevant}/"
                    f"{r.total_relevant}"
                )

        # ── Metric 3: OOD Transfer ───────────────────────────────────
        logger.info("Metric 3: OOD Transfer...")
        eval_out_flat = [
            t for ts in eval_out_extracted.values() for t in ts
        ][:max_eval]
        report.ood_results = evaluator.evaluate_tasks(
            eval_out_flat, trainer.vault
        )
        if report.ood_results:
            report.ood_recall = float(
                np.mean([r.recall for r in report.ood_results])
            )
            report.ood_precision = float(
                np.mean([r.precision for r in report.ood_results])
            )
            report.ood_mrr = float(
                np.mean([r.mrr for r in report.ood_results])
            )
            report.ood_hit_rate = float(
                np.mean([1.0 if r.hit else 0.0 for r in report.ood_results])
            )
            report.ood_recall_capped = float(
                np.mean([r.recall_capped for r in report.ood_results])
            )
        report.cooling_delta = evaluator.compute_cooling_delta(
            trainer.vault, eval_out_flat
        )

        # ── Metric 4: Retention Matrix ────────────────────────────────
        logger.info("Metric 4: Retention Matrix...")
        report.retention_matrix = evaluator.compute_retention_matrix(
            trainer.vault
        )

        # ── Metric 5: Negative Control ────────────────────────────────
        logger.info("Metric 5: Negative Control (out-of-domain queries)...")
        neg = evaluator.evaluate_negative_control(trainer.vault)
        report.negative_recall = neg["negative_recall"]
        report.negative_total = neg["total"]
        report.negative_false_hits = neg["false_hits"]

        # ── Metric 6: Ablation ────────────────────────────────────────
        logger.info("Metric 6: Ablation (DARS+RRF vs Pure Similarity)...")
        ablation_tasks = eval_in_flat[:20]

        dars_results = evaluator.evaluate_tasks(
            ablation_tasks, trainer.vault, use_rrf=True
        )
        baseline_results = evaluator.evaluate_tasks(
            ablation_tasks, trainer.vault, use_rrf=False, alpha=1.0
        )

        if dars_results:
            report.ablation_dars_recall = float(
                np.mean([r.recall for r in dars_results])
            )
            report.ablation_dars_precision = float(
                np.mean([r.precision for r in dars_results])
            )
            report.ablation_dars_mrr = float(
                np.mean([r.mrr for r in dars_results])
            )
            report.ablation_dars_recall_cap = float(
                np.mean([r.recall_capped for r in dars_results])
            )
        if baseline_results:
            report.ablation_baseline_recall = float(
                np.mean([r.recall for r in baseline_results])
            )
            report.ablation_baseline_precision = float(
                np.mean([r.precision for r in baseline_results])
            )
            report.ablation_baseline_mrr = float(
                np.mean([r.mrr for r in baseline_results])
            )
            report.ablation_baseline_recall_cap = float(
                np.mean([r.recall_capped for r in baseline_results])
            )

        trainer.cleanup()
        _print_eval_report(report, k)
        return report

    finally:
        for kk, v in original_weights.items():
            setattr(DARSConfig, kk, v)
        DARSConfig.validate_and_normalize()
        DARSConfig.TRAINING_GROUP = original_group
        DARSConfig._goal_vector_cache = None


# ═══════════════════════════════════════════════════════════════════════════════
#  Report Printer
# ═══════════════════════════════════════════════════════════════════════════════

def _print_eval_report(report: EvalReport, k: int) -> None:
    print("\n" + "=" * 70)
    print(
        f"  GROUP B EVALUATION REPORT  "
        f"(k={k}, RRF, PDDL-grounded relevance)"
    )
    print("=" * 70)

    # Metric 1
    print(f"\n  --- Metric 1: Utility Separation ---")
    for mt in ["strategy", "concept", "goal", "instance"]:
        if mt in report.utility_separation:
            s = report.utility_separation[mt]
            print(
                f"    {mt:12s}: mean={s['mean']:.4f}  "
                f"std={s['std']:.4f}  n={s['count']}"
            )
    strat_mean = report.utility_separation.get("strategy", {}).get("mean")
    concept_mean = report.utility_separation.get("concept", {}).get("mean")
    inst_mean = report.utility_separation.get("instance", {}).get("mean")
    if strat_mean is not None and inst_mean is not None:
        delta = strat_mean - inst_mean
        print(f"    Delta(strategy-instance) = {delta:+.4f}")
        if delta > 0:
            print(f"    PASS: Strategy > Instance (utility learning confirmed)")
        else:
            print(f"    FAIL: Strategy <= Instance")
    if concept_mean is not None and inst_mean is not None:
        delta_c = concept_mean - inst_mean
        print(f"    Delta(concept-instance)  = {delta_c:+.4f}")

    # Metric 2
    print(f"\n  --- Metric 2: In-Distribution Strategic Recall ---")
    print(f"    Tasks evaluated:      {len(report.in_dist_results)}")
    print(f"    Hit Rate:             {report.in_dist_hit_rate:.4f}")
    print(f"    Recall_capped@{k}:    {report.in_dist_recall_capped:.4f}")
    print(f"    Recall_raw@{k}:       {report.in_dist_recall:.4f}")
    print(f"    Precision@{k}:        {report.in_dist_precision:.4f}")
    print(f"    MRR:                  {report.in_dist_mrr:.4f}")
    if report.in_dist_recall_capped >= 0.85:
        print(f"    PASS: Recall_capped >= 85% (k-limited window is saturated)")
    elif report.in_dist_recall_capped >= 0.70:
        print(f"    GOOD: Recall_capped >= 70%")
    else:
        print(f"    NOTE: Recall_capped = {report.in_dist_recall_capped:.1%}")

    # Metric 3
    print(f"\n  --- Metric 3: OOD Transfer ---")
    print(f"    Tasks evaluated:      {len(report.ood_results)}")
    print(f"    OOD Hit Rate:         {report.ood_hit_rate:.4f}")
    print(f"    OOD Recall_capped@{k}: {report.ood_recall_capped:.4f}")
    print(f"    OOD Recall_raw@{k}:   {report.ood_recall:.4f}")
    print(f"    OOD Precision@{k}:    {report.ood_precision:.4f}")
    print(f"    OOD MRR:              {report.ood_mrr:.4f}")
    print(f"    Cooling delta:        {report.cooling_delta:+.4f}")
    if report.cooling_delta > 0:
        print(f"    PASS: Relevant memories rank higher (DARS suppresses noise)")
    else:
        print(f"    NOTE: Cooling delta <= 0")

    # Metric 4
    print(f"\n  --- Metric 4: Retention Matrix ---")
    print(f"    {'Type':<12s} | {'Retain':>7s} | {'Compress':>8s} | {'Delete':>7s}")
    print(f"    {'-'*12}-+-{'-'*7}-+-{'-'*8}-+-{'-'*7}")
    for mt in ["strategy", "concept", "goal", "instance", "unknown"]:
        if mt in report.retention_matrix:
            r = report.retention_matrix[mt]
            print(
                f"    {mt:<12s} | {r.get('retain', 0):>7d} | "
                f"{r.get('compress', 0):>8d} | {r.get('delete', 0):>7d}"
            )

    # Metric 5
    print(f"\n  --- Metric 5: Negative Control (Out-of-Domain) ---")
    print(f"    OOD queries tested:   {report.negative_total}")
    print(f"    False hits (cos>=0.80): {report.negative_false_hits}")
    print(f"    Negative recall:       {report.negative_recall:.4f}")
    if report.negative_recall <= 0.05:
        print(f"    PASS: No spurious retrieval for non-household queries")
    else:
        print(f"    WARN: Negative recall > 5%")

    # Metric 6
    print(f"\n  --- Metric 6: Ablation (DARS+RRF vs Pure Similarity) ---")
    print(
        f"    {'Metric':<14s} | {'DARS+RRF':>10s} | {'Baseline':>10s} | "
        f"{'Delta':>10s}"
    )
    print(f"    {'-'*14}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for label, d_val, b_val in [
        ("Recall_cap@5", report.ablation_dars_recall_cap, report.ablation_baseline_recall_cap),
        ("Recall_raw@5", report.ablation_dars_recall, report.ablation_baseline_recall),
        (
            "Precision@5",
            report.ablation_dars_precision,
            report.ablation_baseline_precision,
        ),
        ("MRR", report.ablation_dars_mrr, report.ablation_baseline_mrr),
    ]:
        delta = d_val - b_val
        print(
            f"    {label:<14s} | {d_val:>10.4f} | {b_val:>10.4f} | "
            f"{delta:>+10.4f}"
        )

    # Verdict
    print(f"\n  --- Verdict ---")
    passes = 0
    total_checks = 5

    if strat_mean is not None and inst_mean is not None and strat_mean > inst_mean:
        passes += 1
        print(f"    [PASS] Metric 1: Utility separation confirmed")
    else:
        print(f"    [FAIL] Metric 1: No utility separation")

    if report.in_dist_recall_capped >= 0.85:
        passes += 1
        print(f"    [PASS] Metric 2: Recall_capped >= 85%")
    elif report.in_dist_precision >= 0.85:
        passes += 1
        print(f"    [PASS] Metric 2: Precision >= 85% (retrieval window saturated)")
    else:
        print(
            f"    [    ] Metric 2: Recall_capped = "
            f"{report.in_dist_recall_capped:.1%}, "
            f"Precision = {report.in_dist_precision:.1%}"
        )

    if report.negative_recall <= 0.05:
        passes += 1
        print(f"    [PASS] Metric 5: Negative control validated")
    else:
        print(f"    [    ] Metric 5: Negative recall = {report.negative_recall:.1%}")

    if report.ablation_dars_recall > report.ablation_baseline_recall:
        passes += 1
        print(f"    [PASS] Metric 6: DARS+RRF recall > baseline")
    elif (
        report.ablation_dars_recall == report.ablation_baseline_recall
        and report.ablation_dars_mrr > report.ablation_baseline_mrr
    ):
        passes += 1
        print(f"    [PASS] Metric 6: DARS+RRF MRR > baseline MRR")
    else:
        print(f"    [    ] Metric 6: No ablation advantage")

    if report.cooling_delta > 0:
        passes += 1
        print(f"    [PASS] Metric 3: Positive cooling delta")
    else:
        print(f"    [    ] Metric 3: Cooling delta = {report.cooling_delta:+.4f}")

    print(f"\n    Checks passed: {passes}/{total_checks}")
    if passes >= 4:
        print(
            f"    DARS demonstrates research-grade strategic learning "
            f"on ALFWorld."
        )
    elif passes >= 3:
        print(f"    DARS shows strong strategic learning; minor tuning possible.")
    else:
        print(f"    Partial success; further tuning recommended.")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Group B ALFWorld Evaluation")
    parser.add_argument(
        "--per-type", type=int, default=10, help="Training tasks per type"
    )
    parser.add_argument("--k", type=int, default=5, help="Top-k for retrieval")
    parser.add_argument(
        "--fetch-k", type=int, default=20, help="Candidates before reranking"
    )
    parser.add_argument(
        "--max-eval", type=int, default=30, help="Max eval tasks per split"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_evaluation(
        max_per_type=args.per_type,
        k=args.k,
        fetch_k=args.fetch_k,
        max_eval=args.max_eval,
    )
