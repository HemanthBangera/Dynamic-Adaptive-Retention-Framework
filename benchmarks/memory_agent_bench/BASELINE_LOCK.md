# Baseline lock (64% EventQA reference)

Use this manifest to reproduce the pre-stack baseline and compare against the narrative / Path A stack.

## Canonical EventQA pilot (Path B historical ~64% EM)

- Split: `Accurate_Retrieval`
- Source: `eventqa_65536` (or the row source used in your paper run)
- HF revision: pin the same `MAB_HF_REVISION` as `benchmark_runs/paper_final/accurate_retrieval_eventqa_65536_path_b/run_manifest.json`
- Path: `b` (legacy), `fetch_k=10`, `top_n=3`, narrative profile **off**

Example:

```bash
python -m benchmarks.memory_agent_bench run ^
  --split Accurate_Retrieval --source eventqa_65536 ^
  --max-samples 25 --chunk-size 4096 --path b --fetch-k 10 --top-n 3 ^
  --no-narrative --chunk-overlap-tokens 0 --no-failure-detail ^
  --output-dir ./benchmark_runs/baseline_lock/eventqa_path_b
```

## Target stack (≥90% EM goal)

Default CLI today: Path **a**, narrative DARS profile, overlap, tombstones, failure_detail.jsonl, virtual time on (via `apply_mab_narrative_profile`).

Use **`--max-samples 0`** to run every row that survives token filters (no subsampling). Pair with **`--max-qa 0`** for all QA pairs per context.

Archive `run_manifest.json`, `results.json`, `metrics_summary.json`, `failure_detail.jsonl`, and `audit_jsonl` (if any) for every regression run.
