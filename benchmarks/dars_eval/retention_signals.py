"""
Retention signals for the confirmatory MSC addendum: write-side components and prior-art baselines.

The DARS vault updates recency R, frequency F and utility U when a memory is *retrieved*, so
these read-side signals partly record what the retriever happened to return. The same kinds of
signal can be computed from what was *written* instead, independently of retrieval:

    R_w = exp(-lambda * hours since the fact was last stated in a persona summary)
    F_w = log(1 + times stated) / log(1 + 50)                 (the vault's F transform, cap 50)
    U_w = (later sessions that restated it + 1) / (later sessions it could appear in + 2)

P is unchanged. ``dars_write_score`` combines them with the same weighted sum and clipping as
the vault. Rows must come from ``run_msc run --record-writes``.

Prior-art retention scores, computed from the same rows:

* Generative Agents (Park et al., 2023): recency 0.995 ** (hours since last retrieval) plus an
  LLM importance rating (1-10), each min-max normalised within the dialogue, weights 1. There is
  no query when a store decides what to keep, so the relevance term is omitted.
* MemoryBank (Zhong et al., 2024): retention exp(-t / S), where t is days since the last recall
  and the strength S starts at 1 and grows by 1 with every recall.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

FREQUENCY_CAP = 50
GA_DECAY_PER_HOUR = 0.995


def restated_sessions(row: Mapping[str, Any]) -> int:
    """Distinct sessions after creation whose summary restated the fact."""
    created = row["created_session"]
    return len({s for s in row["mention_sessions"] if s > created})


def write_components(row: Mapping[str, Any], lam: float) -> Dict[str, float]:
    if "mention_sessions" not in row:
        raise KeyError("row has no write history; re-run run_msc with --record-writes")
    hours = max(row["t_end"] - row["last_write_time"], 0.0) / 3600.0
    return {
        "R": math.exp(-lam * hours),
        "F": min(math.log(1 + row["mentions"]) / math.log(1 + FREQUENCY_CAP), 1.0),
        "U": (restated_sessions(row) + 1) / (row["opportunities"] + 2),
        "P": float(row["components"]["P"]),
    }


def weighted(components: Mapping[str, float], weights: Sequence[float]) -> float:
    w_r, w_f, w_u, w_p = weights
    s = w_r * components["R"] + w_f * components["F"] + w_u * components["U"] + w_p * components["P"]
    return round(min(max(s, 0.0), 1.0), 6)


def write_component_matrix(rows: Sequence[Mapping[str, Any]], lam: float) -> np.ndarray:
    """4 x N matrix of (R_w, F_w, U_w, P), for vectorised weight grids."""
    comps = [write_components(r, lam) for r in rows]
    return np.array([[c[k] for c in comps] for k in ("R", "F", "U", "P")])


def dars_write_score(rows: Sequence[Mapping[str, Any]], lam: float, weights: Sequence[float]) -> np.ndarray:
    return np.array([weighted(write_components(r, lam), weights) for r in rows])


def dars_read_score(rows: Sequence[Mapping[str, Any]], lam: float, weights: Sequence[float]) -> np.ndarray:
    """The pre-registered E9 score: vault components, R recomputed from last access at ``lam``."""
    out = []
    for r in rows:
        comps = dict(r["components"])
        comps["R"] = math.exp(-lam * max(r["t_end"] - r["recency"], 0.0) / 3600.0)
        out.append(weighted(comps, weights))
    return np.array(out)


def _minmax_within(values: np.ndarray, groups: Sequence[Any]) -> np.ndarray:
    out = np.zeros(len(values), dtype=float)
    idx: Dict[Any, list] = defaultdict(list)
    for i, g in enumerate(groups):
        idx[g].append(i)
    for ix in idx.values():
        v = values[ix]
        lo, hi = float(v.min()), float(v.max())
        out[ix] = 0.0 if hi - lo <= 0 else (v - lo) / (hi - lo)
    return out


def generative_agents_score(rows: Sequence[Mapping[str, Any]], importance: Mapping[str, float]) -> np.ndarray:
    """Recency plus importance, each min-max normalised within the dialogue (equal weights)."""
    hours = np.array([max(r["t_end"] - r["recency"], 0.0) / 3600.0 for r in rows])
    recency = GA_DECAY_PER_HOUR ** hours
    imp = np.array([float(importance[r["text"]]) for r in rows])
    groups = [r["dialogue"] for r in rows]
    return _minmax_within(recency, groups) + _minmax_within(imp, groups)


def memorybank_score(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Ebbinghaus retention exp(-t/S): t in days since last recall, S = 1 + recalls."""
    days = np.array([max(r["t_end"] - r["recency"], 0.0) / 86400.0 for r in rows])
    strength = 1.0 + np.array([float(r["frequency"]) for r in rows])
    return np.exp(-days / strength)


def metadata_features(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Sessions since creation, and how often the fact was stated: the metadata model's inputs."""
    return np.array([[r["opportunities"], r["mentions"]] for r in rows], dtype=float)


def metadata_score(rows: Sequence[Mapping[str, Any]], intercept: float, coef: Sequence[float]) -> np.ndarray:
    z = intercept + metadata_features(rows) @ np.asarray(coef, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))


def ga_importance_prompt(memory: str) -> str:
    """The importance prompt of Park et al. (2023), verbatim apart from the memory text."""
    return ("On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) and 10 is "
            "extremely poignant (e.g., a break up, college acceptance), rate the likely poignancy of the "
            f"following piece of memory.\nMemory: {memory}\nRating: <fill in>")


def parse_rating(text: Optional[str]) -> Optional[int]:
    """First integer 1-10 in the reply, or None."""
    if not text:
        return None
    m = re.search(r"\b(10|[1-9])\b", text)
    return int(m.group(1)) if m else None


_SCALE_PHRASES = re.compile(r"\b1\s*(?:-|–|to)\s*10\b|\bout of\s*10\b|/\s*10\b|\bscale of\b", re.IGNORECASE)


def parse_rating_verbose(text: Optional[str]) -> Optional[int]:
    """Rating from a sentence-length reply: the scale itself ("1 to 10", "out of 10") is removed first,
    so "On a scale of 1 to 10, I would rate this a 4" gives 4, not 1."""
    if not text:
        return None
    return parse_rating(_SCALE_PHRASES.sub(" ", text))
