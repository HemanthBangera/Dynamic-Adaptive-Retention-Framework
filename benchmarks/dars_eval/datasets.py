"""
Load MemoryAgentBench sources and turn each context into memory units and
per-question evidence groups (see ``memory_units`` and ``labels``).

Supported families:
- **Accurate Retrieval:** EventQA, RULER QA, LongMemEval.
- **Conflict Resolution:** FactConsolidation.
- **E11:**
  - Test-Time Learning — in-context-learning (ICL) sources, one labelled example per unit;
  - Long-Range Understanding — DetectiveQA, book chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from benchmarks.dars_eval.labels import fc_evidence, fc_labels, ruler_evidence, ruler_gold_documents
from benchmarks.dars_eval.memory_units import (
    MemoryUnit,
    WindowGuard,
    book_units,
    fact_units,
    lme_question_times,
    lme_units,
    ruler_units,
)
from benchmarks.memory_agent_bench.loader import load_mab_filtered
from benchmarks.memory_agent_bench.qa_builder import MAB_RAG_TEMPLATE_KEY, build_qa_pairs

# Domain-matched goal descriptions for the predictive component P (E4 variant "task").
TASK_GOALS = {
    "eventqa": "Events, character actions and plot developments in the storyline of a novel",
    "ruler_qa": "Encyclopedic passages that state facts answering factual questions",
    "longmemeval": "Personal information, preferences, plans and events a user shared in past conversations with an assistant",
    "factconsolidation": "Current factual attributes of entities, where newer facts supersede older ones",
    "icl": "Labelled example texts that map short user utterances to category labels",
    "detective_qa": "Clues, suspects, motives and events in the plot of a detective novel",
}


_EVENTQA_BOILERPLATE = (
    re.compile(r"\s*Your task is to choose from the above events.*$", re.DOTALL),
    re.compile(r"These are the events that have already occurred:\s*"),
    re.compile(r"Below is a list of possible subsequent events:\s*"),
)
# DetectiveQA questions repeat a fixed worked example; the real question and its options
# follow "Now Answer the Question:" and precede the final "Output:".
_DETECTIVE_QUESTION = re.compile(r"Now Answer the Question:\s*(.*?)\s*Output:\s*$", re.DOTALL)


def retrieval_query(source: str, question: str, guard: WindowGuard) -> str:
    """
    The text embedded / BM25-scored as the retrieval query (identical for every method).

    The raw question field is used, with two exceptions:
    - EventQA: the fixed instruction sentences are removed.
    - DetectiveQA: only the question and its options are kept (the worked example is dropped).

    Queries longer than the embedder window keep their *tail* (for EventQA: the latest
    events and the candidate options) instead of being silently truncated at the head
    by the embedder.
    """
    text = question
    fam = family_of(source)
    if fam == "eventqa":
        for pattern in _EVENTQA_BOILERPLATE:
            text = pattern.sub(" ", text)
        text = re.sub(r"\s+\n", "\n", text).strip()
    elif fam == "detective_qa":
        m = _DETECTIVE_QUESTION.search(text)
        if m:
            text = m.group(1).strip()
    if guard.fits(text):
        return text
    # Keep the longest word-aligned tail that fits (original casing and punctuation preserved).
    words = text.split(" ")
    lo, hi = 1, len(words) - 1          # search the smallest start index whose tail fits
    while lo < hi:
        mid = (lo + hi) // 2
        if guard.fits(" ".join(words[mid:])):
            hi = mid
        else:
            lo = mid + 1
    return " ".join(words[lo:]).strip()


def split_name_for(source: str) -> str:
    if source.startswith("factconsolidation"):
        return "Conflict_Resolution"
    if source.startswith("icl_"):
        return "Test_Time_Learning"
    if source.startswith("detective_qa"):
        return "Long_Range_Understanding"
    return "Accurate_Retrieval"


def family_of(source: str) -> str:
    for fam in ("eventqa", "longmemeval", "factconsolidation", "detective_qa"):
        if source.startswith(fam):
            return fam
    if source.startswith("ruler_qa"):
        return "ruler_qa"
    if source.startswith("icl_"):
        return "icl"
    raise ValueError(f"Unsupported source {source!r}")


def icl_units(context: str, guard: WindowGuard) -> List[MemoryUnit]:
    """In-context-learning sources: one labelled example ("<text>\\nlabel: <id>") per memory unit."""
    blocks = [b.strip() for b in context.split("\n\n") if b.strip()]
    units = []
    for i, block in enumerate(blocks):
        if not guard.fits(block):
            raise ValueError(f"ICL example {i} does not fit the embedding window")
        units.append(MemoryUnit(block, None, {"example": i}))
    return units


@dataclass
class Context:
    """One benchmark context with its memory units and questions."""

    source: str
    index: int
    units: List[MemoryUnit]
    questions: List[str]                   # raw question text
    queries: List[str]                     # retrieval query (see retrieval_query)
    formatted_queries: List[str]           # MemoryAgentBench rag_agent query (given to the reader)
    answers: List[Any]
    evidence: List[List[List[int]]]        # per question: groups of unit indices ([] = unlabelled)
    question_times: Optional[List[float]] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def load_contexts(source: str, guard: WindowGuard, *, revision: str = "main",
                  fc_seconds_per_serial: float = 3600.0, fc_serial_prefix: bool = True) -> List[Context]:
    rows, _stats = load_mab_filtered(split_name_for(source), source, revision=revision)
    fam = family_of(source)
    out: List[Context] = []
    for ci, row in enumerate(rows):
        questions = list(row["questions"])
        answers = list(row["answers"])
        formatted = [fq for fq, _a, _id in build_qa_pairs(row, source, agent_key=MAB_RAG_TEMPLATE_KEY)]
        q_times = None
        extra: Dict[str, Any] = {}
        if fam in ("eventqa", "detective_qa"):
            units = book_units(row["context"], guard)
            evidence = [[] for _ in questions]
        elif fam == "icl":
            units = icl_units(row["context"], guard)
            evidence = [[] for _ in questions]
        elif fam == "ruler_qa":
            units = ruler_units(row["context"], guard)
            gold = ruler_gold_documents(source, questions, row["context"])
            evidence = ruler_evidence(units, gold, answers, len(questions))
            extra["gold_documents"] = gold
        elif fam == "longmemeval":
            units, evidence = lme_units(row, guard)
            q_times = lme_question_times(row)
            extra["question_types"] = list(row["metadata"]["question_types"])
        else:
            units = fact_units(row["context"], seconds_per_serial=fc_seconds_per_serial,
                               serial_prefix=fc_serial_prefix)
            labels = fc_labels(row["context"], questions, answers, multi_hop="_mh_" in source)
            evidence = fc_evidence(units, labels)
            extra["fc_labels"] = labels
        queries = [retrieval_query(source, q, guard) for q in questions]
        out.append(Context(source, ci, units, questions, queries, formatted, answers, evidence, q_times, extra))
    return out
