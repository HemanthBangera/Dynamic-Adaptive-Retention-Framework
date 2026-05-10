# MemoryAgentBench pilot Accurate_Retrieval / eventqa_65536

## Manifest

```json
{
  "benchmark": "MemoryAgentBench",
  "hf_dataset": "ai-hyz/MemoryAgentBench",
  "hf_revision": "main",
  "upstream_mab_eval_utils_pin": "main",
  "split": "Accurate_Retrieval",
  "metadata_source": "eventqa_65536",
  "chunk_size_tokens": 4096,
  "tiktoken_model": "gpt-4o-mini",
  "max_test_samples": 5,
  "seed": 42,
  "path_mode": "b",
  "dars_fetch_k": 15,
  "dars_top_n": 3,
  "dars_rerank_alpha": 0.5,
  "baseline_mode": "normal",
  "max_qa_per_context": 5,
  "gemini_inter_qa_sleep_s": 4.0,
  "gemini_max_retries": 4,
  "gemini_min_interval_s": 4.0,
  "gemini_keys_file": "c:\\Users\\Harsh\\Downloads\\DARS-Mini-Project\\keys.txt",
  "min_context_tokens": null,
  "max_context_tokens": null,
  "load_stats": {
    "rows_after_source": 5,
    "rows_after_token_filter": 5,
    "rows_returned": 5
  },
  "vault_recreate": true,
  "keep_collection": false,
  "run_label": "framework_final",
  "audit_jsonl": "audit.jsonl",
  "mab_use_virtual_time": false,
  "mab_virtual_time_step_s": 3600.0,
  "mab_injection_initial_success": 0,
  "gemini_model": "gemini-2.5-flash",
  "embedding_model": "all-MiniLM-L6-v2",
  "goal_description_set": false,
  "training_group": "ALFWorld",
  "python": "3.14.3 (tags/v3.14.3:323c59a, Feb  3 2026, 16:04:56) [MSC v.1944 64 bit (AMD64)]",
  "platform": "Windows-11-10.0.26200-SP0",
  "created_unix": 1778419010.3820806
}
```

## Metrics

- **eventqa_recall**: mean=0.6400 std=0.4800 (n=25)
- **exact_match**: mean=0.6400 std=0.4800 (n=25)
- **f1**: mean=0.7635 std=0.3243 (n=25)
- **input_len**: mean=12483.5200 std=54.3545 (n=25)
- **memory_construction_time**: mean=2.2485 std=5.2326 (n=25)
- **output_len**: mean=16.6800 std=3.3433 (n=25)
- **query_time_len**: mean=28.3894 std=43.3627 (n=25)
- **rougeL_f1**: mean=0.7880 std=0.2864 (n=25)
- **rougeL_recall**: mean=0.7977 std=0.2749 (n=25)
- **rougeLsum_f1**: mean=0.7880 std=0.2864 (n=25)
- **rougeLsum_recall**: mean=0.7977 std=0.2749 (n=25)
- **substring_exact_match**: mean=0.6400 std=0.4800 (n=25)
