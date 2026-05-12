# MemoryAgentBench × DARS

Journal-oriented pilot driver using the **same HF filtering, token chunking, query templates, and string metrics** as [MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench) (`third_party/memoryagentbench_eval/`, see `VENDOR.md`).

## Prereqs

- `.env` with `QDRANT_URL`, `QDRANT_API_KEY`, `GEMINI_API_KEY` (reader + Path A reformulator).
- Optional **`GEMINI_API_KEYS`** (comma-separated) and/or **`--gemini-keys-file`**. The parser scans the whole file for **`AIzaSy…`** tokens (same as curl `X-goog-api-key`), including **embedded** lines such as `Api key : AIzaSy…` or GCP export blocks; labels and project IDs are ignored. Add `keys.txt` to `.gitignore`—never commit keys. Logs record **key index** and **suffix only**, never full keys.
- For neutral predictive scoring on generic benchmark text, set **`GOAL_DESCRIPTION`** in `.env` to a short generic mission string.
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

**PDDL extractors:** The MAB driver does **not** use `data/groupA/extractor.py` or `data/groupB/extractor.py` (PDDL-tuned). The path is HF → `loader` → `chunking` → `MemoryVault.store_memory`.

## Stratified / free-tier runs (Rule of ~30)

- **`--min-context-tokens` / `--max-context-tokens`:** Filter rows by tiktoken length of `context` (model = `--tiktoken-model`) *before* subsampling. Manifest records `load_stats` (`rows_after_source`, `rows_after_token_filter`, `rows_returned`).
- **`--max-samples`:** After filtering, use **30–50** for stable mean/SD in tables without exhausting free-tier in one day. Use **`0`** for **all** rows that pass filters (no subsampling). Pair **`--max-qa 0`** with **`0`** for every QA pair per context (full exhaust).
- **`--gemini-min-interval`:** Default **4.0** seconds between **all** `generateContent` calls in the governed transport (RPM safety). Use **`0`** for fast local/CI only.
- **`--gemini-keys-file`:** Multi-key rotation on **429/403** with capped backoff (`core/gemini_transport.py`). Same transport is wired into **`QueryReformulator`**, **`GeminiBenchmarkReader`**, **`SuccessEvaluator`**, and **`SemanticCompressor`** when keys resolve so Layer B/C cannot bypass pacing during full-stack runs.

## Virtual time and acquisition (research knobs)

- **`MAB_USE_VIRTUAL_TIME`**, **`MAB_VIRTUAL_TIME_STEP_S`:** When enabled, each ingested chunk gets `recency` / `created_at` advanced by `step` seconds so hour-scale λ decay is meaningful during fast benchmark replays (Conflict / selective forgetting).
- **`MAB_INJECTION_INITIAL_SUCCESS`:** If set to `1`, benchmark ingestion seeds a mild Laplace-friendly success count on new points (reduces cold-start retrieval bias vs static RAG during acquisition-only phases).

## LRU / long-context narrative

Top-\(N\) retrieval may miss **global** narrative questions. The runner logs **token savings** and **DARS mass ratio** in `results.json` rows and optional **`--audit-jsonl`** so fragmentation is measurable; vault-level abstractive summaries remain a separate research item.

## Pilot run (narrative stack defaults)

By default the driver applies a **narrative DARS profile** (recency-heavy weights, λ=0.01, Narrative goal, virtual time on), **Path A**, `fetch_k=25`, `top_n=5`, **chunk overlap** (`--chunk-overlap-tokens` default 128), **neighbor chunk expansion** after rerank, **tombstone supersession** on ingest, and **`failure_detail.jsonl`** for every wrong `exact_match`. Use **`--no-narrative`** to revert weights/goal/virtual-time defaults; **`--path b`** for Mem0-style ablation; **`--no-failure-detail`** to skip failure JSONL.

See [`BASELINE_LOCK.md`](BASELINE_LOCK.md) for a frozen **Path B / pre-stack** command line and [`grid_search.py`](grid_search.py) for fetch_k × top_n × overlap sweeps.

```bash
python -m benchmarks.memory_agent_bench run ^
  --split Accurate_Retrieval ^
  --source YOUR_SOURCE_FROM_LIST ^
  --max-samples 40 ^
  --min-context-tokens 10000 ^
  --max-context-tokens 100000 ^
  --max-qa 5 ^
  --gemini-sleep 4 ^
  --gemini-min-interval 4 ^
  --chunk-size 4096 ^
  --audit-jsonl audit.jsonl ^
  --output-dir ./benchmark_runs/pilot/ar_stratified
```

`--max-qa` caps questions per long context; `--max-qa 0` = no cap.

**Outputs:** `run_manifest.json`, `results.json`, `metrics_summary.json`, `summary.md`, `per_sample.jsonl`, optional `failure_detail.jsonl`. Per-QA rows include **money metrics**: `context_tokens`, `retrieved_memory_tokens`, `token_savings_ratio`, `dars_mean_topk`, `dars_mean_vault`, `dars_mass_ratio`, Path A timing fields, `gemini_key_index`. With **`--audit-jsonl`**, rows also include `exact_match`, `expanded_query`, `retrieved_point_ids`, and `retrieved_chunk_indices`.

**Qdrant:** `--no-vault-recreate` skips delete/recreate at episode start; **`--keep-collection`** skips delete after each context (debug / inspection; uses more storage).

**Shim:** `python -m benchmarks.runner` delegates to this module.

## Baselines

- **Empty memory**: `--baseline empty` (retrieval returns nothing; sanity for AR-style splits).

## Path modes (ablation)

- **`--path a` (default)**: `CognitiveGateway.process_query_timed` → XML → Gemini reader (reformulator + rerank + ordered memory stream).
- **`--path b`**: `reranker.rerank` → bullet memories → Gemini reader (closest to Mem0 I/O). Compare **a** vs **b** on the same split/source for ablation tables.

## Batch pilot (`run-presets`)

1. Copy `pilot_presets.template.json` or `research_stages.template.json` and fill sources via `list-sources`.
2. Run:

```bash
python -m benchmarks.memory_agent_bench run-presets --preset-file pilot_presets.json
```

Each job runs in a **subprocess** with `cwd` at the project root. Optional preset keys: `min_context_tokens`, `max_context_tokens`, `gemini_keys_file`, `gemini_min_interval`, `run_label`, `keep_collection`, `no_vault_recreate`, `audit_jsonl`.

## Full paper scale

Increase `--max-samples`, sweep more `--source` values, pin `--hf-revision` to a dataset commit SHA in the manifest, and archive outputs under `benchmark_runs/` (see `benchmark_runs/README.md`) with the paper.
