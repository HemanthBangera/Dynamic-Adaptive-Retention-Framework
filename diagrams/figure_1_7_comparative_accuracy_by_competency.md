---
figure_id: "1.7"
title: "Comparative accuracy by competency split"
deliverable_type: "bar_chart_spec"
---

# Figure 1.7: Comparative accuracy by competency

## Purpose

Specify a **bar chart** (or grouped bars) comparing runs across **MemoryAgentBench splits** and **metadata sources**, using only **real** `metrics_summary.json` files.

## Audience

Results chapter.

## Split vocabulary

`SUPPORTED_SPLITS` in `benchmarks/memory_agent_bench/loader.py`:

- `Accurate_Retrieval`
- `Test_Time_Learning`
- `Long_Range_Understanding`
- `Conflict_Resolution`

**Not every archived run covers every split or every `metadata.source`.** Before plotting, **list directories** under `benchmark_runs/` and only include bars for runs that exist. To discover valid **`metadata.source`** values for a split, run:

`python -m benchmarks.memory_agent_bench list-sources --split <SplitName>` (see `benchmarks/memory_agent_bench/README.md`). Example paths present in this repo snapshot:

- `benchmark_runs/paper_final/accurate_retrieval_eventqa_65536_path_b/metrics_summary.json`
- `benchmark_runs/paper_final/accurate_retrieval_ruler_qa1_197k_path_b/metrics_summary.json`

Both are under split naming convention in folder names; always open **`run_manifest.json`** beside each summary for authoritative `split` + `metadata_source`.

## JSON keys (primary y-axis candidates)

From `metrics_summary.json` structure:

- **`exact_match.mean`** (± `std`, `n`)
- **`f1.mean`**
- **`eventqa_recall.mean`** (when applicable to dataset)
- **`rougeL_f1.mean`** (for generative QA runs)

Pick **one primary metric** per figure panel and move others to appendix tables.

## Visual specification

- X-axis: **run labels** derived from directory names **or** manifest `run_label` + `path_mode`.
- Y-axis: chosen metric mean in **[0, 1]**.
- Error bars: use `std` / sqrt(n) per your thesis stats policy; `metrics_summary.json` stores `std` and `n` per metric.
- Legend: **`path_mode`**, **`dars_top_n` / `fetch_k`** from manifest if comparing retrieval width.

## Do / Don't

- **Do** print **n** from the summary on each bar or in caption.
- **Don't** imply a full **4×sources** matrix unless those JSON files exist.

## Source files

- `benchmark_runs/**/metrics_summary.json`
- `benchmark_runs/**/run_manifest.json`
- `benchmarks/memory_agent_bench/loader.py` — `SUPPORTED_SPLITS`
- `benchmarks/memory_agent_bench/runner.py` — metric accumulation

## Caption draft

*Figure 1.7 — Comparative accuracy (mean ± dispersion) across available MemoryAgentBench runs; bars omitted where no `metrics_summary.json` exists.*

### LaTeX build specification

- **TeX:** `reports_latex/diagrams/tex/fig_1_7.tex` — `pgfplots` bar chart; **data:** `reports_latex/diagrams/data/fig_1_7_metrics.csv` (regenerate with `python reports_latex/diagrams/scripts/export_plot_data.py` from repo root).
- **PDF:** `reports_latex/diagrams/pdf/fig_1_7.pdf`; build via `reports_latex/diagrams/build.ps1`.
