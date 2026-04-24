"""
Group A – DARS Evaluation Pipeline (MSC Temporal Learning)
============================================================
Evaluates DARS retrieval quality using session 3 dialogue as queries
against a memory vault trained on sessions 0-2.

Metrics:
  - Recall@k:  fraction of ground-truth persona facts retrieved
  - Precision@k:  fraction of retrieved results that are relevant
  - MRR:  Mean Reciprocal Rank of first relevant hit
  - Frequency Bias Score:  Recall(high-freq) / Recall(low-freq)

Run standalone:  python -m data.groupA.evaluate [--dialogues N] [--k 3]
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
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
from data.groupA.loader import load_msc
from data.groupA.train import GroupATrainer, INDEXING_WAIT

logger = logging.getLogger(__name__)

RELEVANCE_THRESHOLD = 0.80


@dataclass
class QueryResult:
    query_text: str
    speaker: int
    retrieved_texts: List[str]
    retrieved_scores: List[float]
    relevant_facts: List[str]
    relevant_in_top_k: List[str]
    recall: float
    precision: float
    reciprocal_rank: float


@dataclass
class DialogueEvalResult:
    dialogue_id: int
    num_queries: int
    avg_recall: float
    avg_precision: float
    mrr: float
    recall_high_freq: Optional[float]
    recall_low_freq: Optional[float]
    frequency_bias: Optional[float]
    retention_stats: Dict[str, int]
    query_results: List[QueryResult] = field(default_factory=list)


def _cosine(a, b) -> float:
    a, b = np.array(a), np.array(b)
    d = np.dot(a, b)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(d / n) if n > 0 else 0.0


class GroupAEvaluator:
    """Evaluates DARS retrieval on session-3 queries after training on 0-2."""

    def __init__(self, k: int = 3):
        self.k = k
        self.embedder = EmbeddingEngine()

    def evaluate_dialogue(
        self,
        dialogue: ProcessedDialogue,
        vault: MemoryVault,
        pid_map: Dict[str, str],
    ) -> DialogueEvalResult:
        """Evaluate a single trained dialogue using session 3 queries."""
        fact_texts = [m["text"] for m in dialogue.memories]
        fact_embeddings = {
            t: self.embedder.encode(t) for t in fact_texts
        }

        high_freq_set = set(dialogue.high_freq_facts)
        low_freq_set = set(dialogue.low_freq_facts)

        query_results: List[QueryResult] = []
        high_freq_recalls: List[float] = []
        low_freq_recalls: List[float] = []

        for q in dialogue.eval_queries:
            speaker = q["speaker"]
            speaker_facts = [
                m["text"] for m in dialogue.memories
                if m["speaker"] == speaker
            ]
            if not speaker_facts:
                continue

            results = vault.search_and_rerank(q["text"], top_n=self.k)

            retrieved_texts = []
            retrieved_scores = []
            for mem in results:
                if f"speaker:{speaker}" not in mem.payload.tags:
                    continue
                retrieved_texts.append(mem.payload.text_content)
                retrieved_scores.append(mem.dars_score if mem.dars_score else 0.0)

            query_emb = self.embedder.encode(q["text"])
            relevant_facts = []
            for ft in speaker_facts:
                sim = _cosine(query_emb, fact_embeddings[ft])
                if sim >= RELEVANCE_THRESHOLD:
                    relevant_facts.append(ft)

            if not relevant_facts:
                continue

            relevant_in_top = [r for r in retrieved_texts if r in relevant_facts]
            recall = len(relevant_in_top) / len(relevant_facts) if relevant_facts else 0
            precision = len(relevant_in_top) / len(retrieved_texts) if retrieved_texts else 0

            rr = 0.0
            for rank, r_text in enumerate(retrieved_texts, 1):
                if r_text in relevant_facts:
                    rr = 1.0 / rank
                    break

            qr = QueryResult(
                query_text=q["text"][:80],
                speaker=speaker,
                retrieved_texts=retrieved_texts[:5],
                retrieved_scores=retrieved_scores[:5],
                relevant_facts=relevant_facts[:5],
                relevant_in_top_k=relevant_in_top,
                recall=recall,
                precision=precision,
                reciprocal_rank=rr,
            )
            query_results.append(qr)

            for fact in relevant_facts:
                if fact in high_freq_set:
                    is_retrieved = fact in relevant_in_top
                    high_freq_recalls.append(1.0 if is_retrieved else 0.0)
                elif fact in low_freq_set:
                    is_retrieved = fact in relevant_in_top
                    low_freq_recalls.append(1.0 if is_retrieved else 0.0)

        avg_recall = np.mean([qr.recall for qr in query_results]) if query_results else 0
        avg_precision = np.mean([qr.precision for qr in query_results]) if query_results else 0
        mrr = np.mean([qr.reciprocal_rank for qr in query_results]) if query_results else 0

        recall_high = np.mean(high_freq_recalls) if high_freq_recalls else None
        recall_low = np.mean(low_freq_recalls) if low_freq_recalls else None
        freq_bias = None
        if recall_high is not None and recall_low is not None and recall_low > 0:
            freq_bias = recall_high / recall_low

        all_mems = vault.get_all_memories(limit=500)
        retention_stats = {"retain": 0, "compress": 0, "delete": 0}
        for mem in all_mems:
            dars = vault.compute_dars_score(mem.payload.to_dict())
            classification = vault.classify_memory(dars)
            retention_stats[classification] += 1

        return DialogueEvalResult(
            dialogue_id=dialogue.dialogue_id,
            num_queries=len(query_results),
            avg_recall=float(avg_recall),
            avg_precision=float(avg_precision),
            mrr=float(mrr),
            recall_high_freq=float(recall_high) if recall_high is not None else None,
            recall_low_freq=float(recall_low) if recall_low is not None else None,
            frequency_bias=float(freq_bias) if freq_bias is not None else None,
            retention_stats=retention_stats,
            query_results=query_results,
        )


    def evaluate_persona_retention(
        self,
        dialogue: ProcessedDialogue,
        vault: MemoryVault,
        sessions_data: Dict[int, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluate whether session-3 persona facts are retrievable from DARS.

        This is the strongest test: session 3 persona facts that overlap with
        facts from sessions 0-2 should be retrieved by DARS.
        """
        eval_row = sessions_data.get(3, {})
        s3_personas = []
        for fact in eval_row.get("persona1", []):
            s3_personas.append((fact.strip(), 1))
        for fact in eval_row.get("persona2", []):
            s3_personas.append((fact.strip(), 2))

        trained_texts = set(m["text"] for m in dialogue.memories)
        hits = 0
        total = 0
        reciprocal_ranks = []

        for fact, speaker in s3_personas:
            results = vault.search_and_rerank(fact, top_n=self.k)
            filtered = [
                r for r in results
                if f"speaker:{speaker}" in r.payload.tags
            ]
            retrieved_texts = [r.payload.text_content for r in filtered]

            fact_emb = self.embedder.encode(fact)
            found = False
            for rank, r_text in enumerate(retrieved_texts, 1):
                if r_text not in trained_texts:
                    continue
                r_emb = self.embedder.encode(r_text)
                sim = _cosine(fact_emb, r_emb)
                if sim >= RELEVANCE_THRESHOLD:
                    if not found:
                        reciprocal_ranks.append(1.0 / rank)
                        found = True
                    hits += 1
                    break

            if not found:
                reciprocal_ranks.append(0.0)
            total += 1

        recall = hits / total if total > 0 else 0
        mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0

        return {
            "total_s3_facts": total,
            "hits": hits,
            "recall": float(recall),
            "mrr": float(mrr),
        }


def run_evaluation(
    max_dialogues: int = 50,
    k: int = 3,
    verbose: bool = True,
) -> List[DialogueEvalResult]:
    """Full pipeline: extract -> train -> evaluate for each dialogue."""
    original_group = DARSConfig.TRAINING_GROUP
    DARSConfig.TRAINING_GROUP = "MSC"
    DARSConfig._goal_vector_cache = None

    try:
        logger.info("Extracting dialogues (max=%d)...", max_dialogues)
        dialogues = extract_all(max_dialogues=max_dialogues)
        if not dialogues:
            logger.error("No complete dialogues found.")
            return []

        raw_dialogues = load_msc()
        evaluator = GroupAEvaluator(k=k)
        trainer = GroupATrainer()
        results: List[DialogueEvalResult] = []
        persona_results: List[Dict[str, Any]] = []

        for i, d in enumerate(dialogues):
            logger.info(
                "Training + Evaluating dialogue %d/%d (id=%d)...",
                i + 1, len(dialogues), d.dialogue_id,
            )
            train_result = trainer.train_dialogue(d)
            time.sleep(INDEXING_WAIT)

            eval_result = evaluator.evaluate_dialogue(d, trainer.vault, trainer._pid_map)
            results.append(eval_result)

            sessions_data = raw_dialogues.get(d.dialogue_id, {})
            pr = evaluator.evaluate_persona_retention(d, trainer.vault, sessions_data)
            persona_results.append(pr)

            if verbose:
                fb = eval_result.frequency_bias
                fb_str = f"{fb:.2f}" if fb is not None else "N/A"
                print(
                    f"  [{i+1}/{len(dialogues)}] id={d.dialogue_id} "
                    f"R@{k}={eval_result.avg_recall:.3f} "
                    f"P@{k}={eval_result.avg_precision:.3f} "
                    f"MRR={eval_result.mrr:.3f} "
                    f"FreqBias={fb_str} "
                    f"persona_ret={pr['hits']}/{pr['total_s3_facts']} "
                    f"ret={eval_result.retention_stats}"
                )

        trainer.cleanup()
        _print_eval_report(results, k, persona_results)
        return results

    finally:
        DARSConfig.TRAINING_GROUP = original_group
        DARSConfig._goal_vector_cache = None


def _print_eval_report(
    results: List[DialogueEvalResult],
    k: int,
    persona_results: Optional[List[Dict[str, Any]]] = None,
) -> None:
    recalls = [r.avg_recall for r in results]
    precisions = [r.avg_precision for r in results]
    mrrs = [r.mrr for r in results]
    freq_biases = [r.frequency_bias for r in results if r.frequency_bias is not None]
    high_recalls = [r.recall_high_freq for r in results if r.recall_high_freq is not None]
    low_recalls = [r.recall_low_freq for r in results if r.recall_low_freq is not None]

    total_retain = sum(r.retention_stats.get("retain", 0) for r in results)
    total_compress = sum(r.retention_stats.get("compress", 0) for r in results)
    total_delete = sum(r.retention_stats.get("delete", 0) for r in results)
    total_mem = total_retain + total_compress + total_delete

    print("\n" + "=" * 70)
    print(f"  GROUP A EVALUATION REPORT  (k={k})")
    print("=" * 70)
    print(f"  Dialogues evaluated:         {len(results)}")
    print(f"  Total queries:               {sum(r.num_queries for r in results)}")

    print(f"\n  --- Core Retrieval Metrics ---")
    print(f"  Recall@{k}:      {np.mean(recalls):.4f}  (std={np.std(recalls):.4f})")
    print(f"  Precision@{k}:   {np.mean(precisions):.4f}  (std={np.std(precisions):.4f})")
    print(f"  MRR:             {np.mean(mrrs):.4f}  (std={np.std(mrrs):.4f})")

    if freq_biases:
        print(f"\n  --- Frequency Bias Score ---")
        print(f"  Mean FreqBias (R_high / R_low): {np.mean(freq_biases):.4f}")
        print(f"  Dialogues with FreqBias data:   {len(freq_biases)}")
        if np.mean(freq_biases) > 1.0:
            print(f"  RESULT: DARS preferentially retains high-frequency facts (PASS)")
        else:
            print(f"  RESULT: Frequency bias <= 1.0 (investigate weight tuning)")

    if high_recalls:
        print(f"\n  --- High-Freq Recall (>=3 sessions) ---")
        print(f"  Mean: {np.mean(high_recalls):.4f}  (n={len(high_recalls)})")

    if low_recalls:
        print(f"\n  --- Low-Freq Recall (1 session) ---")
        print(f"  Mean: {np.mean(low_recalls):.4f}  (n={len(low_recalls)})")

    if total_mem > 0:
        print(f"\n  --- Retention Classification ---")
        print(f"  Retain:   {total_retain:,}  ({total_retain/total_mem*100:.1f}%)")
        print(f"  Compress: {total_compress:,}  ({total_compress/total_mem*100:.1f}%)")
        print(f"  Delete:   {total_delete:,}  ({total_delete/total_mem*100:.1f}%)")

    if persona_results:
        pr_recalls = [pr["recall"] for pr in persona_results]
        pr_mrrs = [pr["mrr"] for pr in persona_results]
        pr_total = sum(pr["total_s3_facts"] for pr in persona_results)
        pr_hits = sum(pr["hits"] for pr in persona_results)
        print(f"\n  --- Persona Retention (Session-3 Facts as Queries) ---")
        print(f"  Total session-3 persona facts: {pr_total}")
        print(f"  Facts retrieved from DARS:     {pr_hits}")
        print(f"  Persona Recall:  {np.mean(pr_recalls):.4f}  (std={np.std(pr_recalls):.4f})")
        print(f"  Persona MRR:     {np.mean(pr_mrrs):.4f}  (std={np.std(pr_mrrs):.4f})")

    print("\n  --- Verdict ---")
    mean_recall = np.mean(recalls) if recalls else 0
    mean_mrr = np.mean(mrrs) if mrrs else 0
    persona_mean = np.mean([pr["recall"] for pr in persona_results]) if persona_results else 0

    if persona_mean >= 0.4:
        print(f"  DARS demonstrates strong persona memory retention.")
    elif persona_mean >= 0.2:
        print(f"  DARS shows moderate persona retention; further tuning may improve.")
    elif mean_recall >= 0.15 or persona_mean > 0:
        print(f"  DARS shows basic retrieval; weight tuning recommended.")
    else:
        print(f"  DARS retrieval is weak; investigate scoring components.")

    if freq_biases and np.mean(freq_biases) > 1.0:
        print(f"  Temporal learning (Recency + Frequency) validated.")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Group A MSC Evaluation")
    parser.add_argument("--dialogues", type=int, default=5, help="Number of dialogues to evaluate")
    parser.add_argument("--k", type=int, default=3, help="Top-k for retrieval metrics")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_evaluation(max_dialogues=args.dialogues, k=args.k)
