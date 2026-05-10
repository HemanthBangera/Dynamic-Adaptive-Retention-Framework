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
  - Recall_retrievable:  recall excluding novel S3 facts (primary metric)
  - Novel_rate:  fraction of S3 facts absent from sessions 0-2
  - Negative_recall:  false-retrieval rate from foreign dialogues (control)

Run standalone:  python -m data.groupA.evaluate [--dialogues N] [--k 5]
"""

from __future__ import annotations

import logging
import random
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

    def __init__(self, k: int = 5, fetch_k: int = 20):
        self.k = k
        self.fetch_k = fetch_k
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

            results = vault.search_and_rerank(
                q["text"], top_n=self.k, fetch_k=self.fetch_k,
            )

            retrieved_texts = []
            retrieved_scores = []
            for mem in results:
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

        Separates novel facts (no match in sessions 0-2 at threshold 0.80)
        from retrievable facts for honest recall computation.
        """
        eval_row = sessions_data.get(3, {})
        s3_personas = []
        for fact in eval_row.get("persona1", []):
            s3_personas.append((fact.strip(), 1))
        for fact in eval_row.get("persona2", []):
            s3_personas.append((fact.strip(), 2))

        trained_texts = list(set(m["text"] for m in dialogue.memories))
        trained_embeddings = {t: self.embedder.encode(t) for t in trained_texts}
        trained_set = set(trained_texts)

        hits = 0
        total = 0
        novel_count = 0
        retrievable_hits = 0
        retrievable_total = 0
        reciprocal_ranks = []
        retrievable_rrs = []

        for fact, speaker in s3_personas:
            total += 1
            fact_emb = self.embedder.encode(fact)

            max_sim_to_trained = 0.0
            for t_text, t_emb in trained_embeddings.items():
                sim = _cosine(fact_emb, t_emb)
                if sim > max_sim_to_trained:
                    max_sim_to_trained = sim

            is_novel = max_sim_to_trained < RELEVANCE_THRESHOLD
            if is_novel:
                novel_count += 1
                reciprocal_ranks.append(0.0)
                continue

            retrievable_total += 1

            results = vault.search_and_rerank(
                fact, top_n=self.k, fetch_k=self.fetch_k,
            )
            retrieved_texts = [r.payload.text_content for r in results]

            found = False
            for rank, r_text in enumerate(retrieved_texts, 1):
                if r_text not in trained_set:
                    continue
                r_emb = trained_embeddings.get(r_text)
                if r_emb is None:
                    r_emb = self.embedder.encode(r_text)
                sim = _cosine(fact_emb, r_emb)
                if sim >= RELEVANCE_THRESHOLD:
                    if not found:
                        reciprocal_ranks.append(1.0 / rank)
                        retrievable_rrs.append(1.0 / rank)
                        found = True
                    hits += 1
                    retrievable_hits += 1
                    break

            if not found:
                reciprocal_ranks.append(0.0)
                retrievable_rrs.append(0.0)

        recall_total = hits / total if total > 0 else 0
        recall_retrievable = (
            retrievable_hits / retrievable_total if retrievable_total > 0 else 0
        )
        novel_rate = novel_count / total if total > 0 else 0
        mrr_total = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0
        mrr_retrievable = float(np.mean(retrievable_rrs)) if retrievable_rrs else 0

        return {
            "total_s3_facts": total,
            "novel_count": novel_count,
            "novel_rate": float(novel_rate),
            "retrievable_total": retrievable_total,
            "hits": hits,
            "retrievable_hits": retrievable_hits,
            "recall_total": float(recall_total),
            "recall_retrievable": float(recall_retrievable),
            "mrr_total": float(mrr_total),
            "mrr_retrievable": float(mrr_retrievable),
        }

    def evaluate_negative_control(
        self,
        vault: MemoryVault,
        foreign_facts: List[str],
    ) -> Dict[str, Any]:
        """Control test: query DARS with persona facts from a different dialogue.

        At RELEVANCE_THRESHOLD=0.80, recall should be ~0%.
        """
        trained_mems = vault.get_all_memories(limit=500)
        trained_texts = set(m.payload.text_content for m in trained_mems)
        trained_embeddings = {
            t: self.embedder.encode(t) for t in trained_texts
        }

        false_hits = 0
        total = len(foreign_facts)

        for fact in foreign_facts:
            results = vault.search_and_rerank(
                fact, top_n=self.k, fetch_k=self.fetch_k,
            )
            fact_emb = self.embedder.encode(fact)

            for r in results:
                r_text = r.payload.text_content
                if r_text not in trained_texts:
                    continue
                r_emb = trained_embeddings.get(r_text)
                if r_emb is None:
                    r_emb = self.embedder.encode(r_text)
                sim = _cosine(fact_emb, r_emb)
                if sim >= RELEVANCE_THRESHOLD:
                    false_hits += 1
                    break

        negative_recall = false_hits / total if total > 0 else 0
        return {
            "total_foreign_facts": total,
            "false_hits": false_hits,
            "negative_recall": float(negative_recall),
        }


def run_evaluation(
    max_dialogues: int = 50,
    k: int = 5,
    fetch_k: int = 20,
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
        evaluator = GroupAEvaluator(k=k, fetch_k=fetch_k)
        trainer = GroupATrainer()
        results: List[DialogueEvalResult] = []
        persona_results: List[Dict[str, Any]] = []
        negative_results: List[Dict[str, Any]] = []

        all_dialogue_ids = sorted(raw_dialogues.keys())

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

            foreign_id = _pick_foreign_dialogue(d.dialogue_id, all_dialogue_ids)
            if foreign_id is not None:
                foreign_sessions = raw_dialogues.get(foreign_id, {})
                foreign_s0 = foreign_sessions.get(0, {})
                foreign_facts = (
                    [f.strip() for f in foreign_s0.get("persona1", [])]
                    + [f.strip() for f in foreign_s0.get("persona2", [])]
                )
                if foreign_facts:
                    nr = evaluator.evaluate_negative_control(trainer.vault, foreign_facts)
                    negative_results.append(nr)

            if verbose:
                fb = eval_result.frequency_bias
                fb_str = f"{fb:.2f}" if fb is not None else "N/A"
                print(
                    f"  [{i+1}/{len(dialogues)}] id={d.dialogue_id} "
                    f"R_ret={pr['recall_retrievable']:.3f} "
                    f"R_tot={pr['recall_total']:.3f} "
                    f"novel={pr['novel_count']}/{pr['total_s3_facts']} "
                    f"hits={pr['retrievable_hits']}/{pr['retrievable_total']} "
                    f"ret={eval_result.retention_stats}"
                )

        trainer.cleanup()
        _print_eval_report(results, k, persona_results, negative_results)
        return results

    finally:
        DARSConfig.TRAINING_GROUP = original_group
        DARSConfig._goal_vector_cache = None


def _pick_foreign_dialogue(current_id: int, all_ids: List[int]) -> Optional[int]:
    """Pick a dialogue_id that is far from the current one."""
    candidates = [did for did in all_ids if abs(did - current_id) > 100]
    if not candidates:
        candidates = [did for did in all_ids if did != current_id]
    return random.choice(candidates) if candidates else None


def _print_eval_report(
    results: List[DialogueEvalResult],
    k: int,
    persona_results: Optional[List[Dict[str, Any]]] = None,
    negative_results: Optional[List[Dict[str, Any]]] = None,
) -> None:
    recalls = [r.avg_recall for r in results]
    precisions = [r.avg_precision for r in results]
    mrrs = [r.mrr for r in results]
    freq_biases = [r.frequency_bias for r in results if r.frequency_bias is not None]

    total_retain = sum(r.retention_stats.get("retain", 0) for r in results)
    total_compress = sum(r.retention_stats.get("compress", 0) for r in results)
    total_delete = sum(r.retention_stats.get("delete", 0) for r in results)
    total_mem = total_retain + total_compress + total_delete

    print("\n" + "=" * 70)
    print(f"  GROUP A EVALUATION REPORT  (k={k}, RRF, threshold={RELEVANCE_THRESHOLD})")
    print("=" * 70)
    print(f"  Dialogues evaluated:         {len(results)}")
    print(f"  Total queries:               {sum(r.num_queries for r in results)}")

    if freq_biases:
        print(f"\n  --- Frequency Bias Score ---")
        print(f"  Mean FreqBias (R_high / R_low): {np.mean(freq_biases):.4f}")
        if np.mean(freq_biases) > 1.0:
            print(f"  RESULT: DARS preferentially retains high-frequency facts")
        else:
            print(f"  RESULT: Frequency bias <= 1.0 (investigate weight tuning)")

    if total_mem > 0:
        print(f"\n  --- Retention Classification ---")
        print(f"  Retain:   {total_retain:,}  ({total_retain/total_mem*100:.1f}%)")
        print(f"  Compress: {total_compress:,}  ({total_compress/total_mem*100:.1f}%)")
        print(f"  Delete:   {total_delete:,}  ({total_delete/total_mem*100:.1f}%)")

    if persona_results:
        pr_ret_recalls = [pr["recall_retrievable"] for pr in persona_results]
        pr_tot_recalls = [pr["recall_total"] for pr in persona_results]
        pr_ret_mrrs = [pr["mrr_retrievable"] for pr in persona_results]
        pr_total = sum(pr["total_s3_facts"] for pr in persona_results)
        pr_novel = sum(pr["novel_count"] for pr in persona_results)
        pr_retrievable = sum(pr["retrievable_total"] for pr in persona_results)
        pr_hits = sum(pr["retrievable_hits"] for pr in persona_results)

        print(f"\n  --- Persona Retention (Primary Metric) ---")
        print(f"  Total S3 persona facts:      {pr_total}")
        print(f"  Novel (no match in S0-S2):   {pr_novel}  ({pr_novel/pr_total*100:.1f}%)")
        print(f"  Retrievable:                 {pr_retrievable}")
        print(f"  Retrieved (hits):            {pr_hits}")
        print(f"")
        print(f"  Recall_retrievable:  {np.mean(pr_ret_recalls):.4f}  (std={np.std(pr_ret_recalls):.4f})")
        print(f"  Recall_total:        {np.mean(pr_tot_recalls):.4f}  (std={np.std(pr_tot_recalls):.4f})")
        print(f"  MRR_retrievable:     {np.mean(pr_ret_mrrs):.4f}  (std={np.std(pr_ret_mrrs):.4f})")
        print(f"  Novel_rate:          {pr_novel/pr_total*100:.1f}%")

    if negative_results:
        neg_recalls = [nr["negative_recall"] for nr in negative_results]
        neg_total = sum(nr["total_foreign_facts"] for nr in negative_results)
        neg_false = sum(nr["false_hits"] for nr in negative_results)
        print(f"\n  --- Negative Control (Foreign Dialogue Facts) ---")
        print(f"  Foreign facts tested:        {neg_total}")
        print(f"  False retrievals:            {neg_false}")
        print(f"  Negative recall:             {np.mean(neg_recalls):.4f}")
        if np.mean(neg_recalls) <= 0.05:
            print(f"  RESULT: Threshold {RELEVANCE_THRESHOLD} is statistically meaningful (PASS)")
        else:
            print(f"  RESULT: Threshold may be too loose; consider raising it")

    print("\n  --- Verdict ---")
    if persona_results:
        ret_mean = np.mean(pr_ret_recalls)
        if ret_mean >= 0.80:
            print(f"  Recall_retrievable >= 80%: DARS achieves research-grade retention.")
        elif ret_mean >= 0.60:
            print(f"  Recall_retrievable >= 60%: DARS shows strong retention; further tuning possible.")
        elif ret_mean >= 0.40:
            print(f"  Recall_retrievable >= 40%: Moderate retention; weight tuning recommended.")
        else:
            print(f"  Recall_retrievable < 40%: Weak retention; investigate scoring pipeline.")

    if negative_results and np.mean(neg_recalls) <= 0.05:
        print(f"  Negative control passed: threshold validated.")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Group A MSC Evaluation")
    parser.add_argument("--dialogues", type=int, default=5, help="Number of dialogues")
    parser.add_argument("--k", type=int, default=5, help="Top-k for retrieval")
    parser.add_argument("--fetch-k", type=int, default=20, help="Candidates to fetch before reranking")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_evaluation(max_dialogues=args.dialogues, k=args.k, fetch_k=args.fetch_k)
