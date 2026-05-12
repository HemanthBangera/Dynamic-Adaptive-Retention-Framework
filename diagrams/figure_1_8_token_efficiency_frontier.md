---
figure_id: "1.8"
title: "Token efficiency frontier"
deliverable_type: "scatter_or_line_plot_spec"
---

# Figure 1.8: Token efficiency frontier

## Purpose

Relate **memory token savings** to **task accuracy** using fields already written into **`results.json`** rows by the runner + `metrics_summarization`.

## Audience

Results: cost / efficiency narrative; tie-in to **`core/gemini_transport.py`** pacing in prose (not necessarily on plot).

## Per-row fields (from `results.json`)

| Field | Meaning |
|-------|---------|
| `token_savings_ratio` | `1 - (retrieved_memory_tokens / context_tokens)` when `context_tokens > 0`, else `0` (`runner.py`). |
| `context_tokens` | Target context window size for the episode. |
| `retrieved_memory_tokens` | Tokens counted for retrieved memory block. |
| `exact_match` | Boolean accuracy for that QA. |
| `f1` | Token F1 vs gold. |
| `path_mode` | `a` or `b` — stratify markers or facets. |
| `gemini_key_index` | Which API key slot served the call (useful for ops/debug plots, not core science). |

## Suggested plot types

- **Scatter:** x = `token_savings_ratio`, y = `f1` or `exact_match` (jitter or alpha for overplotting).
- **Frontier:** optional Pareto hull per **run directory** (do not merge incompatible manifests without faceting).

## Visual specification

- Facet panels by **`path_mode`** or by **`run_manifest.json` `split`**.
- Footnote **`gemini_min_interval_s`** and multi-key rotation if discussing wall-clock (from manifest + `gemini_transport.py`).

## Do / Don't

- **Do** keep each point tied to a **`qa_pair_id`** / `query_id` for traceability.
- **Don't** extrapolate token savings beyond the **recorded** `context_tokens` regime.

## Source files

- `benchmark_runs/**/results.json`
- `benchmark_runs/**/run_manifest.json`
- `benchmarks/memory_agent_bench/runner.py` — token fields
- `core/gemini_transport.py` — rate limiting / key rotation (caption footnote)

## Caption draft

*Figure 1.8 — Per-question token savings versus accuracy; points aggregated only within a single manifest-controlled run.*

### LaTeX build specification

- **TeX:** `reports_latex/diagrams/tex/fig_1_8.tex` — scatter of `token_savings_ratio` vs `f1`; **data:** `reports_latex/diagrams/data/fig_1_8_scatter.csv` (same export script; capped row count for readability).
- **PDF:** `reports_latex/diagrams/pdf/fig_1_8.pdf`; build via `reports_latex/diagrams/build.ps1`.
