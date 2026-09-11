"""
Retrieval metrics against evidence groups (no LLM calls).

A question's evidence is a list of groups of memory-unit indices; a group is
covered when any of its units appears in the retrieved set.  Retrieved sets are
cut by *token budget* so every method is compared at the same context size.
"""

from __future__ import annotations

from math import log2
from typing import Dict, List, Optional, Sequence

MEMORY_HEADER_TOKENS = 5  # "Memory {i}:\n" plus the joining newline, in gpt-4o-mini tokens


def cut_to_budget(ranked: Sequence[int], unit_tokens: Sequence[int], budget: int,
                  overhead: int = MEMORY_HEADER_TOKENS) -> List[int]:
    """Longest prefix of ``ranked`` whose formatted size fits ``budget`` tokens."""
    out: List[int] = []
    used = 0
    for u in ranked:
        cost = unit_tokens[u] + overhead
        if used + cost > budget:
            break
        out.append(u)
        used += cost
    return out


def evidence_metrics(retrieved: Sequence[int], groups: Sequence[Sequence[int]]) -> Optional[Dict[str, float]]:
    """
    Coverage metrics for one question, or None if it has no evidence label.

    group_recall   fraction of evidence groups covered by ``retrieved``
    all_covered    1.0 if every group is covered
    mrr            1 / rank of the first retrieved unit that covers any group
    ndcg           binary nDCG where each group earns gain once (at its first hit)
    """
    if not groups or not all(groups):
        return None
    pos = {u: r for r, u in enumerate(retrieved, 1)}
    first_hits = []
    for g in groups:
        ranks = [pos[u] for u in g if u in pos]
        first_hits.append(min(ranks) if ranks else None)
    covered = [r for r in first_hits if r is not None]
    dcg = sum(1.0 / log2(r + 1) for r in covered)
    ideal = sum(1.0 / log2(i + 1) for i in range(1, min(len(groups), len(retrieved)) + 1))
    return {
        "group_recall": len(covered) / len(groups),
        "all_covered": float(len(covered) == len(groups)),
        "mrr": 1.0 / min(covered) if covered else 0.0,
        "ndcg": dcg / ideal if ideal > 0 else 0.0,
    }


def precedence(retrieved: Sequence[int], gold: int, older_versions: Sequence[int]) -> Optional[float]:
    """
    FactConsolidation: 1.0 if the gold (newest) fact is retrieved and ranked above
    every other version of the same fact that was retrieved; 0.0 if it is not
    retrieved or is outranked; None if the fact has no other versions.
    """
    if not older_versions:
        return None
    pos = {u: r for r, u in enumerate(retrieved, 1)}
    if gold not in pos:
        return 0.0
    rivals = [pos[v] for v in older_versions if v in pos]
    return float(all(pos[gold] < r for r in rivals))
