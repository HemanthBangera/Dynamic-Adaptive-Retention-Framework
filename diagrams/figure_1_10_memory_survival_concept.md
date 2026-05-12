---
figure_id: "1.10"
title: "Memory survival (conceptual + instrumentation)"
deliverable_type: "conceptual_plot_or_recipe"
---

# Figure 1.10: Memory survival by policy zone

## Purpose

Provide an **honest** visualization path for “how long memories survive” when the repo **does not ship a telemetry dataset** of per-memory lifetimes over calendar time.

## Audience

Discussion / qualitative systems behavior.

## Part A — Conceptual diagram (allowed without real curves)

- X-axis: **synthetic time** or **triage cycles** (clearly labeled “schematic”).
- Y-axis: **fraction of cohort remaining** in vault (not from logs).
- **Zones** aligned with janitor policy:
  - **Grace** (first 24h: no triage for normal memories).
  - **Retain band** (S > 0.7).
  - **Compress band** (0.3 < S ≤ 0.7).
  - **Delete** (S ≤ 0.3).
- Optional stylized curves **differing by initial S** — mark as **hypothetical**.

## Part B — Instrumentation recipe (if the thesis needs empirical survival)

To produce real curves later:

1. Log **`last_triage_timestamp`**, **`created_at`**, **DARS S at triage**, and **action** (`retain` / `compress` / `delete`) per `point_id` when `DecisionEngine.triage_memory` runs (`core/layer_c/janitor.py` already patches `last_triage_timestamp` on retain).
2. Reconstruct Kaplan–Meier-style survival from **first ingest** to **delete** (censor on compress if you treat compression as “alive but transformed”).
3. Join with **tag** presence (`system:high_priority_distillation`) to show priority-path behavior.

## Do / Don't

- **Do** label conceptual panels **“schematic — not measured in archived runs.”**
- **Don't** fabricate numeric half-lives in the figure without a cited log pipeline.

## Source files

- `core/layer_c/janitor.py`
- `core/layer_d/storage.py` — `classify_memory`, payload timestamps
- `config/settings.py` — thresholds

## Caption draft

*Figure 1.10 — Conceptual survival of memories under grace and DARS score bands, with optional empirical extension via triage logging (not included in baseline benchmark artifacts).*

### LaTeX build specification

- **TeX:** `reports_latex/diagrams/tex/fig_1_10.tex` — `pgfplots` schematic decay curves + shaded policy bands; **explicitly not** empirical telemetry.
- **PDF:** `reports_latex/diagrams/pdf/fig_1_10.pdf`; build via `reports_latex/diagrams/build.ps1`.
