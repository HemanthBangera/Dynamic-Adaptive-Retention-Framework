---
figure_id: "1.9"
title: "Ablation: Path A versus Path B"
deliverable_type: "table_plus_optional_bars"
---

# Figure 1.9: Ablation Path A vs Path B

## Purpose

Document a **controlled comparison** between **`--path a`** (gateway XML reader) and **`--path b`** (default bullet reader), listing **manifest-held constants** so the diagram agent does not compare incomparable runs.

## Audience

Ablation / sensitivity section.

## CLI control

`python -m benchmarks.memory_agent_bench ... --path a|b` (`benchmarks/memory_agent_bench/__main__.py`).

## Manifest fields to hold constant across A/B (when claiming ablation)

From `build_manifest` / `run_manifest.json` (`benchmarks/memory_agent_bench/manifest.py`):

- `hf_revision`
- `split`, `metadata_source`
- `chunk_size_tokens`, `seed`, `max_test_samples`, `max_qa_per_context`
- `path_mode` (**only deliberate change** between the two runs)
- `dars_fetch_k`, `dars_top_n`, `dars_rerank_alpha`
- `baseline_mode`
- `gemini_inter_qa_sleep_s`, `gemini_max_retries`, `gemini_min_interval_s`, `gemini_keys_file`
- `min_context_tokens`, `max_context_tokens`, `load_stats`
- `vault_recreate`, `keep_collection`
- `mab_use_virtual_time`, `mab_virtual_time_step_s`, `mab_injection_initial_success`
- `embedding_model`, `gemini_model`

## Table template (fill from disk)

| Field | Run Path A | Run Path B |
|-------|------------|------------|
| Output directory | | |
| `path_mode` | a | b |
| `hf_revision` | | |
| `dars_fetch_k` / `dars_top_n` | | |
| `exact_match.mean` | | |
| `f1.mean` | | |
| `token_savings_ratio` (mean) | | |

## Automation note

There is **no** `grid_search.py` in this repository snapshot; ablations are typically **repeated CLI invocations** or an external driver. Say so in the caption if asked how sweeps were automated.

## Do / Don't

- **Do** store both manifests next to results when archiving (`paper_final/...`).
- **Don't** cite `paper_replication_*` paths unless they exist in your checkout.

## Source files

- `benchmarks/memory_agent_bench/__main__.py`
- `benchmarks/memory_agent_bench/manifest.py`
- `benchmark_runs/**/run_manifest.json`
- `benchmark_runs/**/metrics_summary.json`

## Caption draft

*Figure 1.9 — Path A vs Path B under matched manifest parameters (only `path_mode` differs).*

### LaTeX build specification

- **TeX:** `reports_latex/diagrams/tex/fig_1_9.tex` — `pgfplotstable` table over `data/fig_1_7_metrics.csv` (run metadata + EM/F1); same CSV as figure 1.7 by design.
- **PDF:** `reports_latex/diagrams/pdf/fig_1_9.pdf`; build via `reports_latex/diagrams/build.ps1` (requires `pgfplotstable`, bundled with most TeX installs).
