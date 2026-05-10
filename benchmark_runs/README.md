# Benchmark run outputs

All MemoryAgentBench × DARS outputs should live **under this directory** so the repo root stays clean.

| Path | Purpose |
|------|--------|
| `paper_final/` | Curated runs intended for paper tables and claims (Path B, normal baseline, governed Gemini, multi-key file when used). |
| `pilot/` | Default drop zone for ad-hoc runs (`--output-dir` default; safe to delete or archive). |

**Run a new pilot** (from project root):

```bash
python -m benchmarks.memory_agent_bench run --split Accurate_Retrieval --source ruler_qa1_197K \
  --output-dir ./benchmark_runs/pilot/my_run_name ...
```

For publication-quality bundles, copy or rename the run folder into `paper_final/` with a descriptive slug, or pass `--output-dir ./benchmark_runs/paper_final/<slug>` directly.
