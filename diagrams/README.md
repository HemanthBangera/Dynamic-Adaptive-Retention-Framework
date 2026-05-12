# Diagram briefs (for a diagramming agent)

This folder holds **self-contained Markdown briefs**: one file per figure. Each brief is written to be handed to an illustrator or another agent that produces PNG/SVG/Mermaid without inventing system behavior.

**Compiled LaTeX (figures 1.3–1.10):** see [`../reports_latex/diagrams/README.md`](../reports_latex/diagrams/README.md) — TikZ/pgfplots sources, CSV data export, and `build.ps1` for PDF output.

## Index

| Figure | Brief file | Typical report use |
|--------|------------|---------------------|
| 0.1 | [figure_0_1_system_scope_and_codepaths.md](figure_0_1_system_scope_and_codepaths.md) | Scope: DARS + MemoryAgentBench vs parallel `data/groupA` / `data/groupB` pipelines |
| 1.1 | [figure_1_1_project_implementation_roadmap.md](figure_1_1_project_implementation_roadmap.md) | Phased roadmap / Gantt-style timeline (placeholder dates) |
| 1.2 | [figure_1_2_dars_multilayer_architecture.md](figure_1_2_dars_multilayer_architecture.md) | Four-layer architecture, Qdrant, Gemini, HF embedder |
| 1.3 | [figure_1_3_multiturn_interaction_sequence.md](figure_1_3_multiturn_interaction_sequence.md) | Path A vs Path B sequences (MAB runner + gateway) |
| 1.4 | [figure_1_4_dars_scoring_variable_dataflow.md](figure_1_4_dars_scoring_variable_dataflow.md) | DFD: raw text → R,F,U,P → S → hybrid / RRF retrieval |
| 1.4b | [figure_1_4b_benchmark_scoring_contract.md](figure_1_4b_benchmark_scoring_contract.md) | MAB: model output → parsed metrics (EM, EventQA recall, etc.) |
| 1.5 | [figure_1_5_janitor_selective_retention.md](figure_1_5_janitor_selective_retention.md) | Janitor: grace, priority tag, triage → retain / compress / delete |
| 1.6 | [figure_1_6_retrieval_path_decisions.md](figure_1_6_retrieval_path_decisions.md) | Path A/B, dual-query, truncation, reader contracts |
| 1.6b | [figure_1_6b_layer_b_feedback_laplace.md](figure_1_6b_layer_b_feedback_laplace.md) | Layer B: LLM judge → Laplace utility → payload patch |
| 1.7 | [figure_1_7_comparative_accuracy_by_competency.md](figure_1_7_comparative_accuracy_by_competency.md) | Bar chart from `metrics_summary.json` / splits |
| 1.8 | [figure_1_8_token_efficiency_frontier.md](figure_1_8_token_efficiency_frontier.md) | Token savings vs accuracy from `results.json` rows |
| 1.9 | [figure_1_9_ablation_path_a_vs_path_b.md](figure_1_9_ablation_path_a_vs_path_b.md) | Controlled comparison; manifest fields to hold constant |
| 1.10 | [figure_1_10_memory_survival_concept.md](figure_1_10_memory_survival_concept.md) | Conceptual survival / instrumentation note (no baked telemetry) |
| 2.1 | [figure_2_1_discussion_limitations_hyperoptimization.md](figure_2_1_discussion_limitations_hyperoptimization.md) | Discussion / limitations (aligned with `further_improvements.md`) |

## Canonical narrative (read first)

- [Architecture.md](../Architecture.md) — equations, layer roles, thresholds.
- [ARCHITECTURE_DIAGRAMS.md](../ARCHITECTURE_DIAGRAMS.md) — Mermaid seeds F1–F2 and glossary.
- [docs/implementation_guide.md](../docs/implementation_guide.md) — implementation tradeoffs and honest notes (use for limitations if no separate doc exists).
- [claude_review.md](../claude_review.md) §8 — “Missing / incomplete implementations” inventory (critical for discussion figures).

## Reproducibility (MAB)

Benchmark runs should ship a **`run_manifest.json`** in each run directory (see `benchmarks/memory_agent_bench/manifest.py` for the field list: HF revision, `path_mode`, `fetch_k`, `top_n`, models, virtual-time flags, etc.). Any experimental-setup figure or table should cite manifest paths, not only `metrics_summary.json`.

## Data sourcing rule

**Only plot runs that exist** under `benchmark_runs/` (or paths the author supplies). As of this repo snapshot, archived `metrics_summary.json` files may cover **only some** split/source combinations; do not imply a full four-split matrix unless those files are present.

## Visual conventions (for the illustrator)

- **Solid arrows**: synchronous data or control flow in the main path.
- **Dashed arrows**: optional branches, async/background (e.g. janitor cycle), or external API calls subject to rate limits.
- **Layer coloring**: align with `ARCHITECTURE_DIAGRAMS.md` if present; otherwise use distinct hues per layer (D, A, B, C) and neutral for Qdrant/Gemini/HF.
- **No invented metrics**: every numeric series must map to a JSON key or an explicitly labeled hypothetical curve.
- **Captions**: cite the brief path (e.g. `diagrams/figure_1_4_dars_scoring_variable_dataflow.md`) in internal notes or appendix.

## RRF note (Layer D)

Hybrid ranking in `QdrantVault.search_and_rerank` defaults to **RRF** fusion of similarity-ranked and DARS-ranked lists (`use_rrf=True`, `rrf_k=60`). When `use_rrf=False`, the code uses a weighted sum of normalized similarity and DARS score (`alpha`). Brief 1.4 documents both.
