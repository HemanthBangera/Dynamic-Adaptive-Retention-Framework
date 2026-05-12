---
figure_id: "1.1"
title: "Project implementation roadmap (phased)"
deliverable_type: "gantt_or_roadmap"
---

# Figure 1.1: Project implementation roadmap

## Purpose

Give a **timeline-style** figure (Gantt or phased roadmap) for thesis planning. The repository does **not** contain an authoritative project calendar; dates are **placeholders** until filled from commit history or your thesis schedule.

## Audience

Introduction or project management appendix.

## Phases (dependency order)

Use this **logical order** (align arrows left-to-right or top-to-bottom):

1. **Layer D — Vault / Qdrant** — persistence, `MemoryPayload`, embeddings, `search_and_rerank`.
2. **Layer A — Gateway / prompt** — `PromptConstructor`, Path A XML contract.
3. **Layer B — Scoring & feedback** — DARS components, Laplace updates, `ingest_new_facts` where applicable.
4. **Layer C — Janitor** — triage, grace period, priority-tag branch, compress/delete.
5. **MemoryAgentBench** — `loader.py` → `qa_builder.py` → `chunking.py` → `runner.py` → metrics.

Optional footnote: **API reliability** — `core/gemini_transport.py` (multi-key rotation, minimum interval on 429) affects wall-clock duration of runs, not mathematical correctness.

## Placeholder timeline table (fill by author)

| Phase | Start (TBD) | End (TBD) | Milestone |
|-------|-------------|-----------|-----------|
| D — Vault | | | First `store_memory` + Qdrant collection |
| A — Gateway | | | Path A answer XML parse stable |
| B — Feedback | | | Judge + Laplace path exercised |
| C — Janitor | | | Scheduled triage on collection |
| MAB — Benchmark | | | `paper_final` or equivalent run archived |

## Visual specification

- Horizontal bars per phase **or** a milestone diamond chart.
- Dashed vertical “freeze” line for thesis submission if desired.
- Footnote: “Dates from repository commit history / author records.”

## Do / Don't

- **Do** label placeholder dates explicitly as **TBD**.
- **Don't** invent specific calendar dates from this brief alone.

## Source files

- `Architecture.md` §1–3 (layer overview)
- `core/gemini_transport.py` (ops / rate limiting footnote)

## Caption draft

*Figure 1.1 — Phased implementation roadmap (D → A → B → C → MemoryAgentBench). Dates are placeholders pending author verification.*
