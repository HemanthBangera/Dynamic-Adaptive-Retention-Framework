"""
Group A – MSC Memory Extractor
================================
Converts raw MSC rows into DARS-compatible memories with:
- Semantic deduplication (cosine > 0.75 clusters)
- Fact frequency tracking across sessions
- High-freq / low-freq classification for evaluation ground truth
- Speaker-tagged dialogue interactions for feedback simulation

Run standalone:  python -m data.groupA.extractor
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.layer_d.embedding import EmbeddingEngine
from data.groupA.loader import load_msc

logger = logging.getLogger(__name__)

DEDUP_THRESHOLD = 0.75
FEEDBACK_MATCH_THRESHOLD = 0.45


@dataclass
class FactCluster:
    """A group of semantically equivalent persona facts across sessions."""
    canonical: str
    canonical_embedding: List[float]
    variants: List[str] = field(default_factory=list)
    sessions: set = field(default_factory=set)
    speaker: int = 1

    @property
    def frequency(self) -> int:
        return len(self.sessions)


@dataclass
class ProcessedDialogue:
    """Fully extracted dialogue ready for DARS training and evaluation."""
    dialogue_id: int
    memories: List[Dict[str, Any]]
    interactions: List[Dict[str, Any]]
    eval_queries: List[Dict[str, Any]]
    fact_clusters: List[FactCluster]
    high_freq_facts: List[str]
    low_freq_facts: List[str]


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def _cluster_facts(
    facts_by_session: Dict[int, List[Tuple[str, int]]],
    embedder: EmbeddingEngine,
) -> List[FactCluster]:
    """Cluster persona facts by semantic similarity across sessions.

    Args:
        facts_by_session: {session_id: [(fact_text, speaker_num), ...]}
        embedder: embedding engine for cosine similarity
    Returns:
        List of FactCluster with deduplicated canonical forms
    """
    all_facts: List[Tuple[str, int, int]] = []
    for sid, facts in facts_by_session.items():
        for text, speaker in facts:
            all_facts.append((text, speaker, sid))

    if not all_facts:
        return []

    texts = [f[0] for f in all_facts]
    embeddings = embedder.encode_batch(texts)
    emb_array = np.array(embeddings)

    clusters: List[FactCluster] = []
    assigned = [False] * len(all_facts)

    for i, (text_i, speaker_i, sid_i) in enumerate(all_facts):
        if assigned[i]:
            continue

        variant_indices = [i]
        cluster = FactCluster(
            canonical=text_i,
            canonical_embedding=embeddings[i],
            variants=[text_i],
            sessions={sid_i},
            speaker=speaker_i,
        )
        assigned[i] = True

        for j in range(i + 1, len(all_facts)):
            if assigned[j]:
                continue
            text_j, speaker_j, sid_j = all_facts[j]
            if speaker_j != speaker_i:
                continue
            sim = _cosine_sim(emb_array[i], emb_array[j])
            if sim >= DEDUP_THRESHOLD:
                assigned[j] = True
                variant_indices.append(j)
                cluster.variants.append(text_j)
                cluster.sessions.add(sid_j)
                if len(text_j) > len(cluster.canonical):
                    cluster.canonical = text_j

        centroid = np.mean(emb_array[variant_indices], axis=0).tolist()
        cluster.canonical_embedding = centroid
        clusters.append(cluster)

    return clusters


def extract_dialogue(
    dialogue_id: int,
    sessions: Dict[int, Dict[str, Any]],
    embedder: EmbeddingEngine,
    train_sessions: Tuple[int, ...] = (0, 1, 2),
    eval_session: int = 3,
) -> Optional[ProcessedDialogue]:
    """Extract a single dialogue into DARS-ready structures.

    Returns None if the dialogue doesn't have all required sessions.
    """
    required = set(train_sessions) | {eval_session}
    if not required.issubset(sessions.keys()):
        return None

    facts_by_session: Dict[int, List[Tuple[str, int]]] = defaultdict(list)
    for sid in train_sessions:
        row = sessions[sid]
        for fact in row.get("persona1", []):
            facts_by_session[sid].append((fact.strip(), 1))
        for fact in row.get("persona2", []):
            facts_by_session[sid].append((fact.strip(), 2))

    clusters = _cluster_facts(facts_by_session, embedder)

    memories: List[Dict[str, Any]] = []
    for cluster in clusters:
        first_session = min(cluster.sessions)
        memories.append({
            "text": cluster.canonical,
            "source": "persona",
            "tags": [
                f"speaker:{cluster.speaker}",
                f"session:{first_session}",
                f"dialogue:{dialogue_id}",
            ],
            "speaker": cluster.speaker,
            "first_session": first_session,
            "frequency_ground_truth": cluster.frequency,
            "centroid_embedding": cluster.canonical_embedding,
        })

    interactions: List[Dict[str, Any]] = []
    for sid in train_sessions:
        if sid == 0:
            continue
        row = sessions[sid]
        dialogue_turns = row.get("dialogue", [])
        speakers = row.get("speaker", [])
        for idx, (turn, spk) in enumerate(zip(dialogue_turns, speakers)):
            speaker_num = 1 if "1" in spk else 2
            interactions.append({
                "text": turn.strip(),
                "speaker": speaker_num,
                "session_id": sid,
                "turn_index": idx,
            })

    eval_queries: List[Dict[str, Any]] = []
    eval_row = sessions[eval_session]
    for idx, (turn, spk) in enumerate(
        zip(eval_row.get("dialogue", []), eval_row.get("speaker", []))
    ):
        speaker_num = 1 if "1" in spk else 2
        eval_queries.append({
            "text": turn.strip(),
            "speaker": speaker_num,
            "turn_index": idx,
        })

    high_freq = [c.canonical for c in clusters if c.frequency >= 3]
    low_freq = [c.canonical for c in clusters if c.frequency == 1]

    return ProcessedDialogue(
        dialogue_id=dialogue_id,
        memories=memories,
        interactions=interactions,
        eval_queries=eval_queries,
        fact_clusters=clusters,
        high_freq_facts=high_freq,
        low_freq_facts=low_freq,
    )


def extract_all(
    max_dialogues: Optional[int] = None,
) -> List[ProcessedDialogue]:
    """Extract all complete dialogues (those with sessions 0-3).

    Args:
        max_dialogues: cap for development/testing (None = all)
    Returns:
        List of ProcessedDialogue objects
    """
    dialogues = load_msc()
    embedder = EmbeddingEngine()

    results: List[ProcessedDialogue] = []
    count = 0
    for did, sessions in sorted(dialogues.items()):
        if max_dialogues is not None and count >= max_dialogues:
            break
        pd = extract_dialogue(did, sessions, embedder)
        if pd is not None:
            results.append(pd)
            count += 1
            if count % 50 == 0:
                logger.info("Extracted %d dialogues...", count)

    logger.info("Extraction complete: %d dialogues processed.", len(results))
    return results


def print_extraction_report(processed: List[ProcessedDialogue]) -> None:
    """Print summary statistics of the extraction."""
    total_memories = sum(len(p.memories) for p in processed)
    total_interactions = sum(len(p.interactions) for p in processed)
    total_eval = sum(len(p.eval_queries) for p in processed)
    total_high = sum(len(p.high_freq_facts) for p in processed)
    total_low = sum(len(p.low_freq_facts) for p in processed)
    total_clusters = sum(len(p.fact_clusters) for p in processed)

    freq_counts = defaultdict(int)
    for p in processed:
        for c in p.fact_clusters:
            freq_counts[c.frequency] += 1

    print("\n" + "=" * 70)
    print("  MSC EXTRACTION REPORT")
    print("=" * 70)
    print(f"  Dialogues processed:         {len(processed):,}")
    print(f"  Total fact clusters (dedup):  {total_clusters:,}")
    print(f"  Total memories to ingest:    {total_memories:,}")
    print(f"  Total interactions (s1+s2):  {total_interactions:,}")
    print(f"  Total eval queries (s3):     {total_eval:,}")
    print(f"  High-freq facts (>=3 sess):  {total_high:,}")
    print(f"  Low-freq facts (1 sess):     {total_low:,}")
    print(f"\n  --- Frequency Distribution ---")
    for freq in sorted(freq_counts.keys()):
        print(f"  Appeared in {freq} session(s): {freq_counts[freq]:,} clusters")

    if processed:
        sample = processed[0]
        print(f"\n  --- Sample Dialogue #{sample.dialogue_id} ---")
        print(f"  Memories: {len(sample.memories)}")
        for m in sample.memories[:5]:
            print(f"    [{m['tags'][0]}] freq={m['frequency_ground_truth']} | {m['text'][:80]}")
        print(f"  High-freq: {sample.high_freq_facts[:3]}")
        print(f"  Low-freq:  {sample.low_freq_facts[:3]}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    processed = extract_all(max_dialogues=10)
    print_extraction_report(processed)
