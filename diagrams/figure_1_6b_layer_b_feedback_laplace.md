---
figure_id: "1.6b"
title: "Layer B feedback and Laplacian utility update"
deliverable_type: "sequence_or_flowchart"
---

# Figure 1.6b: Layer B feedback (judge → Laplace → vault)

## Purpose

Show **post-hoc** learning from **LLM-as-judge** verdicts to **utility / frequency / recency** updates, including **NEUTRAL** short-circuit and **Laplacian** utility formula.

## Audience

Methods: feedback loop; discussion of stability vs sparse rewards.

## Components

| Block | Code |
|-------|------|
| **Judge** | `SuccessEvaluator.evaluate_success` → returns **`YES`**, **`NO`**, or **`NEUTRAL`** (`core/layer_b/evaluator.py`). |
| **Learning orchestration** | `LearningEngine.process_feedback_loop` (`core/layer_b/engine.py`). |
| **Utility math** | `ScoreCalculator.calculate_updates` — Laplace-style **`utility = (new_success + 1) / (attempts + 2)`** when `attempts > 0`, else `0.5` (`core/layer_b/calculator.py`). |
| **Vault writes** | `MemoryVault.update_utility`, `increment_frequency`, `update_recency` (optimistic locking in storage implementation). |

## Control flow

1. Build concatenated **memory_texts** from retrieved payloads.
2. `judgment = await evaluator.evaluate_success(query, response, memory_texts)`.
3. If **`judgment == "NEUTRAL"`** → **exit** (no metadata patching).
4. Else `success = (judgment == "YES")` → for each retrieved memory id: **`update_utility(pid, success)`**, **`increment_frequency(pid)`**, **`update_recency(pid)`** (errors logged; loop continues in current implementation).

**Separate path (not the judge loop):** **`ingest_new_facts`** → `store_memory` for new text with predictive **P** from **GOAL_VECTOR** cosine inside storage (see `engine.py` docstring). Optional inset arrow from Layer A “new facts” list.

## Visual specification

- Small sequence: **Runner/User** → **LearningEngine** → **SuccessEvaluator** → **Gemini** (dashed) → verdict back.
- Second swimlane: **parallel fan-out** to vault ops per memory id.
- Callout box with the **utility formula** copied verbatim from `calculator.py`.

## Do / Don't

- **Do** show **NEUTRAL** as a hard stop before any vault mutation.
- **Don't** label this as “full RL”; it is **supervised-style binary feedback** on retrieved set.

## Source files

- `core/layer_b/engine.py` — `process_feedback_loop`, `ingest_new_facts`
- `core/layer_b/evaluator.py` — `SuccessEvaluator`
- `core/layer_b/calculator.py` — `ScoreCalculator.calculate_updates`
- `core/layer_d/storage.py` — `update_utility`, `increment_frequency`, `update_recency`

## Caption draft

*Figure 1.6b — Layer B asynchronous feedback: Gemini judge verdicts map to Laplacian-smoothed utility and vault metadata updates, with NEUTRAL as a no-update path.*
