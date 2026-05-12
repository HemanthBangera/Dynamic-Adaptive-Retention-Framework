# Changelog

All notable MemoryAgentBench × DARS narrative-stack changes are summarized here. Older repo history may predate this file.

## [Unreleased]

### MemoryAgentBench narrative / EventQA stack

- **Reformulator:** Narrative-temporal Gemini prompt (anchors, exclusions, anti–synonym-only); whitespace-only input skips LLM and returns raw query; `_is_degenerate_expansion` rejects weak expansions.
- **Clock:** `narrative_query_clock` at end of injected narrative (virtual time per chunk when `MAB_USE_VIRTUAL_TIME`); passed through `CognitiveGateway` → `DARSReranker` → `MemoryVault.search_and_rerank` / DARS scoring.
- **Coverage:** Default `fetch_k=25`, `top_n=5`; chunk **N±1** expansion after rerank when `MAB_EXPAND_NEIGHBOR_CHUNKS`; optional **dual-query** merge (reformulated + raw) when `MAB_DUAL_QUERY_RETRIEVAL` (narrative profile enables both).
- **Tombstones:** `MemoryPayload.superseded`; Qdrant filter in `semantic_search`; ingest calls `supersede_similar_lower_chunks` (high similarity, lower chunk index, not `chunk:i-1`).
- **Driver:** Path **A** default; `--max-samples 0` uses all filtered contexts; `failure_detail.jsonl` and audit fields for wrong EM. See `Architecture.md` and `benchmarks/memory_agent_bench/BASELINE_LOCK.md`.
