---
figure_id: "2.1"
title: "Discussion: limitations and hyperparameter surface"
deliverable_type: "structured_bullet_figure_or_table"
---

# Figure 2.1: Discussion — limitations and “hyperoptimization” risks

## Purpose

Give the diagram agent material for a **limitations / threats to validity** figure (mind-map, table, or risk matrix) grounded in **repository-honest** gaps, not marketing copy.

## Audience

Discussion section; reviewer-facing.

## Primary references

- `claude_review.md` §8 — **Missing / incomplete implementations** table (predictive dynamics, semantic drift forgetting, systematic ablations, baselines, auto memory extraction, etc.).
- `docs/implementation_guide.md` — **Honest implementation notes** (reformulator constraints, gateway concerns).
- `docs/test_suite_details.md` — adversarial / loophole test philosophy (what green tests do **not** guarantee).

## Suggested figure layout

- **Columns:** *Limitation* | *Evidence / doc anchor* | *Mitigation or scope statement*.
- **Rows (examples — verify wording against live docs):**
  - **Path A vs B asymmetry** — different reader contracts may change EM vs ROUGE tradeoffs; always cite `path_mode`.
  - **Prompt truncation** — `MAX_PROMPT_CHARS` can drop memories silently; affects long-context splits.
  - **Judge / NEUTRAL** — feedback loop skips updates on `NEUTRAL`; sparse learning signal.
  - **GOAL_VECTOR / predictive cold-start** — documented risks in reviews if goal embedding misconfigured.
  - **No built-in survival telemetry** — figure 1.10 caveat.
  - **External rate limits** — `gemini_transport` pacing changes duration, not necessarily accuracy.

## Do / Don't

- **Do** separate **“not implemented”** from **“implemented but fragile”** (tests exist vs missing).
- **Don't** present roadmap items as completed features.

## Source files

- `claude_review.md`
- `docs/implementation_guide.md`
- `docs/test_suite_details.md`
- `Architecture.md` — spec vs implementation deltas (if called out)
- `diagrams/figure_1_6_retrieval_path_decisions.md`, `diagrams/figure_1_10_memory_survival_concept.md`

## Caption draft

*Figure 2.1 — Limitations and evaluation threats: mapping documented gaps and operational constraints to their impact on measured benchmark outcomes.*
