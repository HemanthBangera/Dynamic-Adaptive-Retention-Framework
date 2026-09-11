"""
Memory units: the granularity at which each dataset is stored as memories.

Every unit must fit the embedder's input window (all-MiniLM-L6-v2 reads 256
wordpieces including [CLS] and [SEP]); otherwise the tail of the memory is
silently truncated and invisible to retrieval.  ``WindowGuard`` enforces this.

Units per dataset
-----------------
EventQA          sentence-bounded chunks of <= 200 tokens
RULER QA         the same, within ``Document N:`` boundaries (header kept)
LongMemEval      one unit per turn part, prefixed with the session time and role
FactConsolidation one unit per numbered fact (serial number kept in the text)
"""

from __future__ import annotations

import ast
import datetime as dt
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from benchmarks.memory_agent_bench.chunking import ensure_nltk_punkt
from third_party.memoryagentbench_eval import chunk_text_into_sentences

DEFAULT_CHUNK_TOKENS = 200
TIKTOKEN_MODEL = "gpt-4o-mini"
DOC_RE = re.compile(r"(?:^|\n)Document (\d+):\n")
FACT_RE = re.compile(r"^\s*(\d+)\.\s+(.*?)\s*$")
CHAT_TIME_RE = re.compile(r"(\d{4})/(\d{2})/(\d{2})\s*\(\w+\)\s*(\d{2}):(\d{2})")


@dataclass
class MemoryUnit:
    """One memory to be stored: text, optional timestamp (unix seconds) and metadata."""

    text: str
    timestamp: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
#  Embedder window guard
# ═══════════════════════════════════════════════════════════════════════════════


class WindowGuard:
    """Counts embedder wordpieces and splits any text that would be truncated."""

    def __init__(self, embedder=None):
        from core.layer_d.embedding import EmbeddingEngine

        self.embedder = embedder or EmbeddingEngine()
        self.embedder._load_model()
        model = self.embedder._model
        self.tokenizer = model.tokenizer
        self.max_wordpieces = int(model.max_seq_length) - 2  # [CLS] and [SEP]

    def count(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def fits(self, text: str, limit: Optional[int] = None) -> bool:
        return self.count(text) <= (self.max_wordpieces if limit is None else limit)

    def split(self, text: str, limit: Optional[int] = None) -> List[str]:
        """Split ``text`` into pieces that each fit ``limit`` wordpieces (default: the window)."""
        limit = self.max_wordpieces if limit is None else limit
        text = text.strip()
        if not text:
            return []
        if self.fits(text, limit):
            return [text]
        from nltk import sent_tokenize

        ensure_nltk_punkt()
        pieces: List[str] = []
        for sent in sent_tokenize(text):
            pieces.extend([sent] if self.fits(sent, limit) else self._split_words(sent, limit))
        return self._pack(pieces, limit)

    def split_with_prefix(self, prefix: str, body: str) -> List[str]:
        """Split ``body`` so that ``prefix + " " + piece`` fits the window; returns full texts."""
        budget = self.max_wordpieces - self.count(prefix + " ")
        if budget < 16:
            raise ValueError(f"Prefix too long for the embedder window: {prefix[:60]!r}")
        return [f"{prefix} {piece}" for piece in self.split(body, budget)]

    def _split_words(self, text: str, limit: int) -> List[str]:
        out: List[str] = []
        cur: List[str] = []
        for word in text.split():
            if cur and not self.fits(" ".join(cur + [word]), limit):
                out.append(" ".join(cur))
                cur = [word]
            else:
                cur.append(word)
        if cur:
            out.append(" ".join(cur))
        final: List[str] = []
        for piece in out:  # a single token longer than the window (e.g. a long URL)
            if self.fits(piece, limit):
                final.append(piece)
            else:
                ids = self.tokenizer(piece, add_special_tokens=False)["input_ids"]
                for i in range(0, len(ids), limit):
                    final.append(self.tokenizer.decode(ids[i:i + limit]))
        return final

    def _pack(self, pieces: Sequence[str], limit: int) -> List[str]:
        packed: List[str] = []
        cur = ""
        for piece in pieces:
            cand = f"{cur} {piece}".strip()
            if cur and not self.fits(cand, limit):
                packed.append(cur)
                cur = piece
            else:
                cur = cand
        if cur:
            packed.append(cur)
        return packed


# ═══════════════════════════════════════════════════════════════════════════════
#  EventQA / generic long documents
# ═══════════════════════════════════════════════════════════════════════════════


def chunk_text(text: str, guard: WindowGuard, max_tokens: int = DEFAULT_CHUNK_TOKENS) -> List[str]:
    """Sentence-bounded chunks of <= ``max_tokens`` tiktoken tokens that fit the window."""
    ensure_nltk_punkt()
    out: List[str] = []
    for chunk in chunk_text_into_sentences(text, model_name=TIKTOKEN_MODEL, chunk_size=max_tokens):
        out.extend(guard.split(chunk))
    return out


def book_units(context: str, guard: WindowGuard, max_tokens: int = DEFAULT_CHUNK_TOKENS) -> List[MemoryUnit]:
    return [MemoryUnit(text, meta={"chunk": i}) for i, text in enumerate(chunk_text(context, guard, max_tokens))]


# ═══════════════════════════════════════════════════════════════════════════════
#  RULER QA
# ═══════════════════════════════════════════════════════════════════════════════


def parse_ruler_documents(context: str) -> Dict[int, str]:
    parts = DOC_RE.split(context)
    return {int(parts[i]): parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def ruler_units(context: str, guard: WindowGuard, max_tokens: int = DEFAULT_CHUNK_TOKENS) -> List[MemoryUnit]:
    """Chunks within document boundaries; each keeps its ``Document N:`` header."""
    ensure_nltk_punkt()
    units: List[MemoryUnit] = []
    for doc_id, text in parse_ruler_documents(context).items():
        prefix = f"Document {doc_id}:"
        for ci, chunk in enumerate(chunk_text_into_sentences(text, model_name=TIKTOKEN_MODEL, chunk_size=max_tokens)):
            for pi, piece in enumerate(guard.split_with_prefix(prefix, chunk)):
                units.append(MemoryUnit(piece, meta={"doc_id": doc_id, "chunk": ci, "part": pi}))
    return units


# ═══════════════════════════════════════════════════════════════════════════════
#  LongMemEval
# ═══════════════════════════════════════════════════════════════════════════════


def parse_chat_time(text: str) -> dt.datetime:
    """``'2023/05/21 (Sun) 16:10'`` (optionally prefixed ``Chat Time:``) → aware UTC datetime."""
    m = CHAT_TIME_RE.search(text)
    if not m:
        raise ValueError(f"Unrecognised LongMemEval date: {text!r}")
    y, mo, d, h, mi = (int(g) for g in m.groups())
    return dt.datetime(y, mo, d, h, mi, tzinfo=dt.timezone.utc)


def parse_lme_sessions(context: str) -> List[Tuple[dt.datetime, List[Dict[str, str]]]]:
    """The context is a Python-literal list alternating ``'Chat Time: …'`` and session turn lists."""
    items = ast.literal_eval(context)
    dates, sessions = items[0::2], items[1::2]
    if len(dates) != len(sessions):
        raise ValueError("LongMemEval context does not alternate dates and sessions")
    return [(parse_chat_time(d), s) for d, s in zip(dates, sessions)]


def _turn_key(turn: Dict[str, Any]) -> Tuple[str, str]:
    return (turn.get("role", ""), turn.get("content", ""))


def lme_units(row: Dict[str, Any], guard: WindowGuard) -> Tuple[List[MemoryUnit], List[List[List[int]]]]:
    """
    Units for one LongMemEval row, plus evidence groups per question.

    Each turn becomes one or more units ``"[YYYY/MM/DD HH:MM] Role: text"``
    timestamped with its session time.  Evidence for question q is one group per
    turn flagged ``has_answer`` in q's own haystack sessions; a group is covered
    when any unit (part) of that turn is retrieved.
    """
    sessions = parse_lme_sessions(row["context"])
    haystacks = row["metadata"]["haystack_sessions"]

    units: List[MemoryUnit] = []
    units_by_turn: Dict[Tuple[str, str], List[int]] = {}
    for si, (when, turns) in enumerate(sessions):
        stamp = f"[{when:%Y/%m/%d %H:%M}]"
        for ti, turn in enumerate(turns):
            content = (turn.get("content") or "").strip()
            if not content:
                continue
            prefix = f"{stamp} {turn.get('role', '').capitalize()}:"
            for pi, text in enumerate(guard.split_with_prefix(prefix, content)):
                units_by_turn.setdefault(_turn_key(turn), []).append(len(units))
                units.append(MemoryUnit(
                    text,
                    timestamp=when.timestamp(),
                    meta={"session": si, "turn": ti, "part": pi, "role": turn.get("role", "")},
                ))

    evidence: List[List[List[int]]] = []
    for q_sessions in haystacks:
        groups: List[List[int]] = []
        for session in q_sessions:
            for turn in session:
                if turn.get("has_answer") and _turn_key(turn) in units_by_turn:
                    groups.append(list(units_by_turn[_turn_key(turn)]))
        evidence.append(groups)
    flagged: Set[int] = {u for q in evidence for g in q for u in g}
    for i in flagged:
        units[i].meta["has_answer"] = True
    return units, evidence


def lme_question_times(row: Dict[str, Any]) -> List[float]:
    return [parse_chat_time(d).timestamp() for d in row["metadata"]["question_dates"]]


# ═══════════════════════════════════════════════════════════════════════════════
#  FactConsolidation
# ═══════════════════════════════════════════════════════════════════════════════


def parse_facts(context: str) -> Dict[int, str]:
    facts: Dict[int, str] = {}
    for line in context.split("\n"):
        m = FACT_RE.match(line)
        if m:
            facts[int(m.group(1))] = m.group(2)
    return facts


def fact_units(context: str, t0: float = 0.0, seconds_per_serial: float = 3600.0) -> List[MemoryUnit]:
    """One unit per fact, ``"N. fact"``; a higher serial number is a newer fact (later timestamp)."""
    return [
        MemoryUnit(f"{serial}. {fact}", timestamp=t0 + serial * seconds_per_serial, meta={"serial": serial})
        for serial, fact in sorted(parse_facts(context).items())
    ]
