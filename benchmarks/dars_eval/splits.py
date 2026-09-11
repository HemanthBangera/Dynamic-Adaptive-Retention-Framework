"""
Pre-registered development / test split.

Within every context (row), a seeded permutation assigns ``round(dev_frac * n)``
questions to *dev* and the rest to *test*.  Splitting at question level keeps
all contexts as bootstrap clusters in both splits.  The assignment depends only
on (seed, source, context index, question index) — never on results.
"""

from __future__ import annotations

import zlib
from typing import List

import numpy as np

SPLIT_SEED = 20260911
DEV_FRACTION = 0.30


def question_splits(source: str, context_index: int, n_questions: int,
                    seed: int = SPLIT_SEED, dev_fraction: float = DEV_FRACTION) -> List[str]:
    """``["dev" | "test"]`` for each question of one context."""
    rng = np.random.default_rng([seed, zlib.crc32(f"{source}:{context_index}".encode("utf-8"))])
    order = rng.permutation(n_questions)
    n_dev = int(round(dev_fraction * n_questions))
    labels = ["test"] * n_questions
    for q in order[:n_dev]:
        labels[int(q)] = "dev"
    return labels
