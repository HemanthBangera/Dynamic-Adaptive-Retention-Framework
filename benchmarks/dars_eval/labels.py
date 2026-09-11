"""
Deterministic evidence labels.

Evidence for a question is a list of *groups*; a group is a list of memory-unit
indices, and it is covered when any of its units is retrieved.  Single-hop
questions have one group; multi-hop questions have one group per required
document / turn.

RULER QA1 (SQuAD) and QA2 (HotpotQA)
    The gold paragraph(s) are recovered by matching each question to its source
    dataset (SQuAD v2 dev; HotpotQA distractor dev supporting paragraphs) and
    locating them among the ``Document N`` blocks.  The mapping is cached as JSON.
LongMemEval
    Turn-level ``has_answer`` flags (built in ``memory_units.lme_units``).
FactConsolidation
    The fact that states the answer and, for single-hop questions, shares the
    question's subject; the newest such fact (highest serial number) is gold.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from benchmarks.dars_eval.memory_units import MemoryUnit, parse_facts, parse_ruler_documents

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABEL_CACHE = PROJECT_ROOT / "benchmark_runs" / "_labels"

_STOP = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "by", "with", "from", "is", "are",
    "was", "were", "be", "been", "which", "what", "who", "whom", "whose", "where", "when",
    "how", "did", "does", "do", "that", "this", "as", "and", "or", "it", "its", "their", "his",
    "her", "country", "city",
}


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def content_tokens(text: str) -> set:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP}


def normalize_answer(text: str) -> str:
    """SQuAD-style normalisation (lowercase, strip punctuation and articles)."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return _norm_ws(text)


# ═══════════════════════════════════════════════════════════════════════════════
#  RULER
# ═══════════════════════════════════════════════════════════════════════════════


def ruler_gold_documents(source: str, questions: Sequence[str], context: str,
                         cache_dir: Path = LABEL_CACHE) -> Dict[int, List[int]]:
    """Map question index → gold ``Document N`` ids (cached)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"ruler_gold_{source}.json"
    if cache_file.is_file():
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        return {int(k): v for k, v in data["gold"].items()}

    from datasets import load_dataset

    docs = {k: _norm_ws(v) for k, v in parse_ruler_documents(context).items()}
    gold: Dict[int, List[int]] = {}

    def locate(snippet: str) -> List[int]:
        s = _norm_ws(snippet)[:200]
        return sorted(k for k, v in docs.items() if s and s in v)

    if "qa1" in source:
        squad = load_dataset("rajpurkar/squad_v2", split="validation")
        by_q: Dict[str, List[str]] = {}
        for ex in squad:
            by_q.setdefault(ex["question"].strip(), []).append(ex["context"])
        for qi, q in enumerate(questions):
            hits = sorted({d for c in by_q.get(q.strip(), []) for d in locate(c)})
            if hits:
                gold[qi] = hits
        origin = "rajpurkar/squad_v2:validation"
    elif "qa2" in source:
        hotpot = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
        by_q = {ex["question"].strip(): ex for ex in hotpot}
        for qi, q in enumerate(questions):
            ex = by_q.get(q.strip())
            if ex is None:
                continue
            titles = set(ex["supporting_facts"]["title"])
            paras = ["".join(s) for t, s in zip(ex["context"]["title"], ex["context"]["sentences"]) if t in titles]
            hits = [locate(p) for p in paras]
            if all(hits):
                gold[qi] = sorted({d for h in hits for d in h})
        origin = "hotpotqa/hotpot_qa:distractor:validation"
    else:
        raise ValueError(f"Not a RULER QA source: {source}")

    cache_file.write_text(
        json.dumps({"source": source, "origin": origin, "n_questions": len(questions), "gold": gold}, indent=1),
        encoding="utf-8",
    )
    return gold


def ruler_evidence(units: Sequence[MemoryUnit], gold_docs: Dict[int, List[int]],
                   answers: Sequence[Any], n_questions: int) -> List[List[List[int]]]:
    """One group per gold document: its units that contain a gold answer, else all its units."""
    by_doc: Dict[int, List[int]] = {}
    for i, u in enumerate(units):
        by_doc.setdefault(u.meta["doc_id"], []).append(i)
    evidence: List[List[List[int]]] = []
    for qi in range(n_questions):
        ans = answers[qi] if isinstance(answers[qi], list) else [answers[qi]]
        norm_ans = [normalize_answer(a) for a in ans if a and normalize_answer(a)]
        groups: List[List[int]] = []
        for doc in gold_docs.get(qi, []):
            members = by_doc.get(doc, [])
            strict = [i for i in members if any(a in normalize_answer(units[i].text) for a in norm_ans)]
            groups.append(strict or members)
        evidence.append(groups)
    return evidence


# ═══════════════════════════════════════════════════════════════════════════════
#  FactConsolidation
# ═══════════════════════════════════════════════════════════════════════════════


def _strip_one_period(text: str) -> str:
    text = text.strip()
    return text[:-1] if text.endswith(".") else text


def fc_gold_fact(facts: Dict[int, str], question: str, answers: Iterable[str],
                 multi_hop: bool) -> Optional[Dict[str, Any]]:
    """
    Gold fact for one FactConsolidation question.

    Candidates are facts whose text ends with a gold answer.  For single-hop
    questions only candidates whose subject/relation prefix shares the most
    content words with the question are kept (multi-hop questions do not name
    the final-hop subject).  The newest candidate is gold; facts with the same
    prefix are its (older or newer) conflicting versions.
    """
    ans = [a.strip() for a in answers if a and a.strip()]
    cands = []
    for serial, fact in facts.items():
        body = _strip_one_period(fact)
        for a in ans:
            if body.lower().endswith(a.lower()):
                cands.append((serial, body[: len(body) - len(a)].strip()))
                break
    if not cands:
        return None
    if not multi_hop:
        q_tokens = content_tokens(question)
        overlaps = [len(q_tokens & content_tokens(prefix)) for _, prefix in cands]
        best = max(overlaps)
        if best > 0:
            cands = [c for c, o in zip(cands, overlaps) if o == best]
    serial, prefix = max(cands)
    same_prefix = sorted(
        s for s, f in facts.items()
        if _strip_one_period(f).startswith(prefix + " ")
    )
    return {
        "gold_serial": serial,
        "prefix": prefix,
        "versions": same_prefix,
        "conflict_serials": [s for s in same_prefix if s != serial],
        "gold_is_newest": bool(same_prefix) and serial == max(same_prefix),
    }


def fc_labels(context: str, questions: Sequence[str], answers: Sequence[Any],
              multi_hop: bool) -> List[Optional[Dict[str, Any]]]:
    """Per-question gold facts; ``None`` when no fact states the answer, and
    labels whose gold fact is not the newest version are kept but flagged."""
    facts = parse_facts(context)
    out = []
    for q, a in zip(questions, answers):
        out.append(fc_gold_fact(facts, q, a if isinstance(a, list) else [a], multi_hop))
    return out


def fc_evidence(units: Sequence[MemoryUnit], labels: Sequence[Optional[Dict[str, Any]]]) -> List[List[List[int]]]:
    """One group holding the gold fact's unit (empty evidence when unlabelled)."""
    by_serial = {u.meta["serial"]: i for i, u in enumerate(units)}
    return [
        [[by_serial[lab["gold_serial"]]]] if lab and lab["gold_serial"] in by_serial else []
        for lab in labels
    ]


def label_coverage(evidence: Sequence[List[List[int]]]) -> Dict[str, Any]:
    n = len(evidence)
    labelled = sum(1 for e in evidence if e and all(e))
    return {"questions": n, "labelled": labelled, "coverage": labelled / n if n else 0.0}
