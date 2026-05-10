# MemoryAgentBench × DARS

Journal-oriented pilot driver using the **same HF filtering, token chunking, query templates, and string metrics** as [MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench) (`third_party/memoryagentbench_eval/`, see `VENDOR.md`).

## Prereqs

- `.env` with `QDRANT_URL`, `QDRANT_API_KEY`, `GEMINI_API_KEY` (reader + Path A reformulator).
- For neutral predictive scoring on generic benchmark text, set **`GOAL_DESCRIPTION`** in `.env` to a short generic mission string (see project plan §7).
- NLTK: first chunking run may download `punkt` / `punkt_tab`.

## Discover valid `metadata.source` values

```bash
cd DARS-Mini-Project
python -m benchmarks.memory_agent_bench list-sources --split Accurate_Retrieval
python -m benchmarks.memory_agent_bench list-sources --split Test_Time_Learning
python -m benchmarks.memory_agent_bench list-sources --split Long_Range_Understanding
python -m benchmarks.memory_agent_bench list-sources --split Conflict_Resolution
```

Pick **1–2** sources per split that match a template family in `templates.py` (e.g. names containing `ruler_` + `qa`, or `longmemeval_`, etc.). Wrong strings raise `NotImplementedError` from `normalize_dataset_name`.

## Pilot run (Path B default)

```bash
python -m benchmarks.memory_agent_bench run ^
  --split Accurate_Retrieval ^
  --source YOUR_SOURCE_FROM_LIST ^
  --max-samples 1 ^
  --max-qa 5 ^
  --gemini-sleep 4 ^
  --chunk-size 4096 ^
  --path b ^
  --fetch-k 15 ^
  --top-n 3 ^
  --output-dir ./mab_pilot_results/ar_src1
```

`--max-qa` caps questions per long context (many RULER rows have large QA lists); use `--max-qa 0` for no cap. `--gemini-sleep` spaces reader calls to reduce **429** rate limits on free/low quotas.

Outputs: `run_manifest.json`, `results.json`, `metrics_summary.json`, `summary.md`, `per_sample.jsonl`.

## Baselines

- **Empty memory**: `--baseline empty` (retrieval returns nothing; sanity for AR-style splits).

## Path modes

- **`--path b`**: `reranker.rerank` → bullet memories → Gemini reader (closest to Mem0 I/O).
- **`--path a`**: full `CognitiveGateway.process_query` XML → Gemini reader (full Layer A).

## Batch pilot (`run-presets`)

1. Copy `pilot_presets.template.json` to e.g. `pilot_presets.json` and replace remaining `REPLACE_*` sources using `list-sources`.
2. Run:

```bash
python -m benchmarks.memory_agent_bench run-presets --preset-file pilot_presets.json
```

Each job runs in a **subprocess** with `cwd` at the project root.

## Full paper scale

Increase `--max-samples`, sweep more `--source` values, pin `HF_DATASET_REVISION` / `--hf-revision` to a dataset commit SHA in the manifest, and archive `output-dir` with the paper.
