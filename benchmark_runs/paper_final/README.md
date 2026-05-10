# Paper-ready MemoryAgentBench runs

These are the **latest full-stack Path B** evaluations (normal baseline, DARS retrieval + reader), using the governed Gemini transport and `keys.txt` key pool where noted in each `run_manifest.json`.

## Runs in this folder

| Folder | Split / source | What to cite |
|--------|----------------|--------------|
| `accurate_retrieval_ruler_qa1_197k_path_b` | Accurate_Retrieval / `ruler_qa1_197K` | Long-context **needle-style** ruler QA; dataset exposes **1** row for this source — metrics aggregate **5 QA pairs** on that single context (`max_qa_per_context: 5`). |
| `accurate_retrieval_eventqa_65536_path_b` | Accurate_Retrieval / `eventqa_65536` | **5** contexts × up to **5** QA → **25** scored rows; stronger sample for mean/SD in tables. |

Primary artifacts in each folder: `summary.md`, `results.json`, `metrics_summary.json`, `run_manifest.json`, `audit.jsonl`, `per_sample.jsonl`.

**Removed from repo root (obsolete smoke / verify):** `mab_verify_*`, `mab_final_report_envkey` (empty baseline, 1 sample), `mab_final_report_run`.
