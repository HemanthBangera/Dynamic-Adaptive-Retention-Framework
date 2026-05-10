# MemoryAgentBench pilot Accurate_Retrieval / ruler_qa1_197K

## Manifest

```json
{
  "benchmark": "MemoryAgentBench",
  "hf_dataset": "ai-hyz/MemoryAgentBench",
  "hf_revision": "main",
  "upstream_mab_eval_utils_pin": "main",
  "split": "Accurate_Retrieval",
  "metadata_source": "ruler_qa1_197K",
  "chunk_size_tokens": 4096,
  "tiktoken_model": "gpt-4o-mini",
  "max_test_samples": 1,
  "seed": 42,
  "path_mode": "b",
  "dars_fetch_k": 10,
  "dars_top_n": 3,
  "dars_rerank_alpha": 0.5,
  "baseline_mode": "normal",
  "gemini_model": "gemini-2.5-flash",
  "embedding_model": "all-MiniLM-L6-v2",
  "goal_description_set": false,
  "training_group": "ALFWorld",
  "python": "3.14.3 (tags/v3.14.3:323c59a, Feb  3 2026, 16:04:56) [MSC v.1944 64 bit (AMD64)]",
  "platform": "Windows-11-10.0.26200-SP0",
  "created_unix": 1778404410.0960023
}
```

## Metrics

- **exact_match**: mean=0.0100 std=0.0995 (n=100)
- **f1**: mean=0.0159 std=0.1023 (n=100)
- **input_len**: mean=12245.5000 std=21.8694 (n=100)
- **memory_construction_time**: mean=1.0830 std=10.7754 (n=100)
- **output_len**: mean=6.8700 std=9.9616 (n=100)
- **query_time_len**: mean=11.7748 std=3.8556 (n=100)
- **rougeL_f1**: mean=0.0190 std=0.1063 (n=100)
- **rougeL_recall**: mean=0.0452 std=0.1860 (n=100)
- **rougeLsum_f1**: mean=0.0190 std=0.1063 (n=100)
- **rougeLsum_recall**: mean=0.0452 std=0.1860 (n=100)
- **substring_exact_match**: mean=0.0400 std=0.1960 (n=100)
