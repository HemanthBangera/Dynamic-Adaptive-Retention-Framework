# DARS Research Architecture — Diagrams and Reference

This document provides **publication-style** architecture views for the **Dynamic Adaptive Retention Scoring (DARS)** codebase: ecosystem context, four-layer runtime, Layer D retrieval internals, **Path A vs Path B** query pipelines (including MemoryAgentBench), learning and maintenance loops, and a **glossary** mapping every major diagram element to source files.

**Companion:** Full mathematical and contract specification remains in [Architecture.md](Architecture.md).

**Rendering:** Mermaid diagrams render on GitHub and in many Markdown previews (VS Code “Open Preview”). If a diagram fails to render, check for corporate firewall stripping of diagram blocks.

---

## Figure index

| ID | Title | Section |
|----|--------|---------|
| F1 | Ecosystem and external dependencies | §1 |
| F2 | Four-layer logical architecture and policy boundaries | §2 |
| F3 | Layer D — vault, embedding, search, hybrid rerank | §3 |
| F4 | Path A — CognitiveGateway query pipeline (sequence) | §4 |
| F5 | Path B — Rerank-only + bullet reader (sequence, MAB) | §5 |
| F6 | MemoryAgentBench driver — end-to-end evaluation pipeline | §6 |
| F7 | Layer B — post-response learning loop | §7 |
| F8 | Layer C — maintenance triage lifecycle | §8 |
| F9 | Auxiliary research data paths (Group A / Group B) | §9 |

**Legend (edges):** Solid arrows indicate primary **data** or **control** flow. Dashed lines in F2 indicate **configuration** injection from `DARSConfig`. Layer A does **not** write vectors; Layer B/C mutate **payload metadata** or lifecycle state **only** through Layer D APIs.

---

## 1. Ecosystem and external dependencies (F1)

**Narrative:** The Python runtime orchestrates **DARS** against **Qdrant** (vector persistence) and **Google Gemini** (optional LLM calls for reformulation, benchmark reading, evaluation, compression). MemoryAgentBench evaluation additionally pulls **Hugging Face** tabular data.

```mermaid
flowchart TB
  subgraph dev [Operator_and_codebase]
    Dev[Researcher_or_CI]
    Repo[DARS_Mini_Project_repo]
  end

  subgraph runtime [Python_runtime]
    Driver[MAB_CLI_or_run_dars]
    Core[DARS_core_layers]
  end

  subgraph cloud [External_services]
    HF[(HuggingFace_MemoryAgentBench)]
    GEM[(Google_Gemini_API)]
    QD[(Qdrant_vector_DB)]
  end

  Dev --> Repo
  Repo --> Driver
  Driver --> Core
  Core --> GEM
  Core --> QD
  Driver --> HF
```

### F1 element reference

| Element | Role |
|---------|------|
| `Researcher_or_CI` | Runs `python -m benchmarks.memory_agent_bench`, tests, or `run_dars.py`. |
| `DARS_Mini_Project_repo` | Source under `config/`, `core/`, `benchmarks/`, `tests/`, `third_party/`. |
| `MAB_CLI_or_run_dars` | [`benchmarks/memory_agent_bench/__main__.py`](benchmarks/memory_agent_bench/__main__.py) or [`run_dars.py`](run_dars.py). |
| `DARS_core_layers` | Layers A–D under [`core/`](core/). |
| `HuggingFace_MemoryAgentBench` | Dataset `ai-hyz/MemoryAgentBench` via [`benchmarks/memory_agent_bench/loader.py`](benchmarks/memory_agent_bench/loader.py). |
| `Google_Gemini_API` | HTTP `generateContent`; governed by [`core/gemini_transport.py`](core/gemini_transport.py). |
| `Qdrant_vector_DB` | Collections per MAB episode; [`core/layer_d/storage.py`](core/layer_d/storage.py). |

---

## 2. Four-layer logical architecture (F2)

**Narrative:** **Layer A** prepares queries and reads memory **read-only**. **Layer B** updates **metadata** after judged outcomes. **Layer C** runs **retain / compress / delete** policies. **Layer D** owns **vectors + payloads + scoring + retrieval**.

```mermaid
flowchart TB
  subgraph cfg [Configuration]
    DC[DARSConfig_settings_py]
  end

  subgraph layerA [Layer_A_Cognitive_Gateway]
    GW[CognitiveGateway]
    RF[QueryReformulator]
    RR[DARSReranker]
    PC[PromptConstructor]
  end

  subgraph layerB [Layer_B_Learning]
    LE[LearningEngine]
    EV[SuccessEvaluator]
    SC[ScoreCalculator]
  end

  subgraph layerC [Layer_C_Maintenance]
    TO[TriageOrchestrator]
    DE[DecisionEngine]
    CM[SemanticCompressor]
  end

  subgraph layerD [Layer_D_Memory_Vault]
    EM[EmbeddingEngine]
    MV[MemoryVault]
    SCH[Schema_MemoryPayload_MemoryPoint]
    QD[(Qdrant)]
  end

  DC -.->|read| layerA
  DC -.->|read| layerB
  DC -.->|read| layerC
  DC -.->|read| layerD

  GW --> RF
  GW --> RR
  GW --> PC
  RR --> MV

  LE --> EV
  LE --> SC
  LE --> MV

  TO --> DE
  DE --> CM
  DE --> MV
  TO --> MV

  MV --> EM
  MV --> SCH
  MV --> QD
```

### F2 policy summary

| Layer | May read vault | May mutate vectors | May patch payload fields |
|-------|----------------|----------------------|---------------------------|
| A | Yes | No | No |
| B | Yes | No | Yes (via `patch_payload` / update APIs) |
| C | Yes | Delete / compress text | Yes |
| D | Yes | Yes (upsert / delete) | Yes |

---

## 3. Layer D — retrieval and scoring internals (F3)

**Narrative:** Text is **embedded**, stored as a **point** (vector + `MemoryPayload`). Queries use **semantic search** then **DARS-weighted** hybrid score (`alpha` blends similarity and DARS). Results are **`MemoryPoint`** lists consumed by Layer A.

```mermaid
flowchart LR
  subgraph ingest [Ingestion]
    Txt[text_content]
    Enc[EmbeddingEngine_encode]
    Ups[MemoryVault_store_memory]
  end

  subgraph query [Query_path]
    Qtxt[query_text]
    EmbQ[encode_query]
    Sem[semantic_search_top_k]
    Rnk[search_and_rerank_fetch_topn]
    Out[MemoryPoint_list]
  end

  Txt --> Enc --> Ups
  Qtxt --> EmbQ --> Sem --> Rnk --> Out
  Ups --> QD[(Qdrant)]
  Sem --> QD
```

### F3 notes

- **Hybrid score:** `alpha * normalized_similarity + (1 - alpha) * dars_score` (see [Architecture.md](Architecture.md) §5–6).
- **`MemoryPayload`:** Fields such as `utility`, `frequency`, `recency`, `predictive`, `tags`, `source` — see [`core/layer_d/schema.py`](core/layer_d/schema.py).
- **MAB ingest:** Benchmark runner tags chunks with `chunk:{i}` and optional virtual timestamps per [`DARSConfig.MAB_USE_VIRTUAL_TIME`](config/settings.py) in [`benchmarks/memory_agent_bench/runner.py`](benchmarks/memory_agent_bench/runner.py).

---

## 4. Path A — CognitiveGateway pipeline (F4)

**Narrative:** **Reformulate** (async LLM, fail-open to raw query) → **rerank** in a thread pool (sync Qdrant) → **XML prompt** with raw user query + ranked memories → downstream LLM (e.g. MAB **reader**).

```mermaid
sequenceDiagram
  participant Caller as Benchmark_or_UI
  participant GW as CognitiveGateway
  participant RF as QueryReformulator
  participant RR as DARSReranker
  participant V as MemoryVault
  participant PC as PromptConstructor

  Caller->>GW: process_query_timed(raw_query)
  GW->>RF: reformulate_query(raw)
  RF-->>GW: expanded_or_raw
  GW->>RR: rerank(query=expanded, fetch_k, top_n, alpha)
  RR->>V: search_and_rerank(...)
  V-->>RR: ranked_MemoryPoints
  RR-->>GW: memories
  GW->>PC: build(raw_query, memories)
  PC-->>GW: xml_prompt
  GW-->>Caller: xml_prompt, timings, memories
```

### F4 timing fields

Returned `timings` (MAB path) include `reformulate_s`, `retrieve_s`, `xml_build_s`, `gateway_total_s` — see [`core/layer_a/gateway.py`](core/layer_a/gateway.py) and row payloads in [`benchmarks/memory_agent_bench/runner.py`](benchmarks/memory_agent_bench/runner.py).

---

## 5. Path B — Rerank-only + bullet reader (F5)

**Narrative:** **No** reformulation and **no** XML gateway assembly for the reader. Raw **formatted_query** goes to **`DARSReranker.rerank`**, memories become **markdown bullets**, then **`GeminiBenchmarkReader.answer_with_memories_bullets`**.

```mermaid
sequenceDiagram
  participant R as MAB_runner
  participant RR as DARSReranker
  participant V as MemoryVault
  participant RD as GeminiBenchmarkReader

  R->>RR: rerank(formatted_query, fetch_k, top_n, alpha)
  RR->>V: search_and_rerank(...)
  V-->>RR: ranked_MemoryPoints
  RR-->>R: memories
  R->>RD: answer_with_memories_bullets(query, bullets)
  RD-->>R: raw_answer_text
```

---

## 6. MemoryAgentBench evaluation driver (F6)

**Narrative:** Load filtered HF rows → per **context** create **ephemeral Qdrant collection** → **chunk** long context → **store** each chunk → for each QA pair run **Path A or B** → accumulate **vendored metrics** (`metrics_summarization`) → write **`run_manifest.json`**, **`results.json`**, **`metrics_summary.json`**, **`summary.md`**.

```mermaid
flowchart TB
  subgraph load [Data_load]
    HF[HF_MemoryAgentBench]
    LD[loader_load_mab_filtered]
    QB[qa_builder_build_qa_pairs]
  end

  subgraph episode [Per_context_episode]
    CN[collection_name_unique]
    CH[chunk_context_tiktoken]
    ST[MemoryVault_store_memory_per_chunk]
    LP[QA_loop]
  end

  subgraph answer [Per_QA]
    PA[Path_A_gateway_plus_XML_reader]
    PB[Path_B_rerank_bullet_reader]
    MS[metrics_summarization]
  end

  subgraph out [Artifacts]
    MF[run_manifest_json]
    RS[results_json]
    MM[metrics_summary_json]
    SM[summary_md]
  end

  HF --> LD --> QB
  LD --> CN
  CN --> CH --> ST --> LP
  LP --> PA
  LP --> PB
  PA --> MS
  PB --> MS
  MS --> MF
  MS --> RS
  MS --> MM
  MS --> SM
```

### F6 branching

- **Path selection:** `path_mode` in [`runner.py`](benchmarks/memory_agent_bench/runner.py) (`a` vs `b`).
- **CLI:** [`benchmarks/memory_agent_bench/__main__.py`](benchmarks/memory_agent_bench/__main__.py) wires `GovernedGeminiTransport`, `QueryReformulator`, `CognitiveGateway`, `GeminiBenchmarkReader`.

---

## 7. Layer B — learning loop (F7)

**Narrative:** After an answer, the **SuccessEvaluator** (Gemini) returns YES/NO/NEUTRAL. **ScoreCalculator** updates counts and utility; **LearningEngine** applies patches **through Layer D** only.

```mermaid
flowchart TB
  subgraph feedback [Feedback_path]
    Ctx[retrieved_memories_as_text]
    EV[SuccessEvaluator]
    Verdict{YES_NO_NEUTRAL}
    SC[ScoreCalculator]
    LE[LearningEngine]
    MV[MemoryVault_patch]
  end

  Ctx --> EV --> Verdict
  Verdict -->|YES_or_NO| SC --> LE --> MV
  Verdict -->|NEUTRAL| Z[no_metadata_write]
```

---

## 8. Layer C — maintenance triage (F8)

**Narrative:** **TriageOrchestrator** scans when vault size exceeds threshold; **DecisionEngine** classifies each point (retain / compress / delete); **SemanticCompressor** may rewrite `text_content` while keeping vectors unchanged per [Architecture.md](Architecture.md) §8.

```mermaid
flowchart TB
  TO[TriageOrchestrator_trigger]
  Cnt{count_above_threshold}
  Scan[scroll_chunked_points]
  DE[DecisionEngine_triage_memory]
  Pol{score_zones}
  Ret[patch_last_triage]
  Cmp[SemanticCompressor]
  Del[delete_memory]

  TO --> Cnt
  Cnt -->|yes| Scan --> DE --> Pol
  Pol -->|retain| Ret
  Pol -->|compress| Cmp
  Pol -->|delete| Del
  Cnt -->|no| Idle[skip]
```

---

## 9. Auxiliary research data paths (F9)

**Narrative:** **`data/groupA`** and **`data/groupB`** hold **separate** PDDL-oriented training and evaluation pipelines (not used by the HF MemoryAgentBench driver). They illustrate historical / parallel research tracks.

```mermaid
flowchart LR
  subgraph mab [Primary_MAB_driver]
    MAB[benchmarks_memory_agent_bench]
  end

  subgraph legacy [PDDL_group_pipelines]
    GA[data_groupA_train_evaluate]
    GB[data_groupB_train_evaluate]
  end

  MAB -.->|not_same_codepath| GA
  MAB -.->|not_same_codepath| GB
```

---

## 10. Master glossary — diagram elements to source files

| Diagram ID | Symbol / name | Primary module(s) | Responsibility |
|------------|---------------|-------------------|----------------|
| F1 | `DARS_Mini_Project_repo` | repo root | Versioned implementation + tests. |
| F1 | `MAB_CLI` | [`benchmarks/memory_agent_bench/__main__.py`](benchmarks/memory_agent_bench/__main__.py) | Argparse, `run` / `list-sources`, manifest write. |
| F1 | `run_dars` | [`run_dars.py`](run_dars.py) | Long-running process; starts Layer C orchestrator. |
| F2 | `DARSConfig` | [`config/settings.py`](config/settings.py) | Global constants, DARS weights, MAB toggles, API env. |
| F2 | `CognitiveGateway` | [`core/layer_a/gateway.py`](core/layer_a/gateway.py) | Async orchestration Path A. |
| F2 | `QueryReformulator` | [`core/layer_a/reformulator.py`](core/layer_a/reformulator.py) | LLM query expansion, fail-open. |
| F2 | `DARSReranker` | [`core/layer_a/reranker.py`](core/layer_a/reranker.py) | Thin adapter: calls `MemoryVault.search_and_rerank` with `fetch_k`, `top_n`, `alpha`. |
| F2 | `PromptConstructor` | [`core/layer_a/prompt_constructor.py`](core/layer_a/prompt_constructor.py) | XML-safe `<memory_stream>` assembly; char budget. |
| F2 | `LearningEngine` | [`core/layer_b/engine.py`](core/layer_b/engine.py) | Feedback + ingest orchestration. |
| F2 | `SuccessEvaluator` | [`core/layer_b/evaluator.py`](core/layer_b/evaluator.py) | LLM YES/NO judgment. |
| F2 | `ScoreCalculator` | [`core/layer_b/calculator.py`](core/layer_b/calculator.py) | Deterministic metadata deltas. |
| F2 | `TriageOrchestrator` | [`core/layer_c/triage.py`](core/layer_c/triage.py) | Maintenance scheduler. |
| F2 | `DecisionEngine` | [`core/layer_c/janitor.py`](core/layer_c/janitor.py) | Per-point retain/compress/delete. |
| F2 | `SemanticCompressor` | [`core/layer_c/compressor.py`](core/layer_c/compressor.py) | Summarization patches. |
| F2 | `EmbeddingEngine` | [`core/layer_d/embedding.py`](core/layer_d/embedding.py) | Sentence-transformers encode. |
| F2 | `MemoryVault` | [`core/layer_d/storage.py`](core/layer_d/storage.py) | Qdrant CRUD, search, rerank, scoring. |
| F2 | `Schema` | [`core/layer_d/schema.py`](core/layer_d/schema.py) | `MemoryPayload`, `MemoryPoint`, weights, decisions. |
| F3 | `semantic_search` | [`storage.py`](core/layer_d/storage.py) | Vector nearest neighbors + optional filters. |
| F3 | `search_and_rerank` | [`storage.py`](core/layer_d/storage.py) | Hybrid ranking pipeline. |
| F4–F5 | `GeminiBenchmarkReader` | [`benchmarks/memory_agent_bench/reader.py`](benchmarks/memory_agent_bench/reader.py) | Gemini calls; `Answer:` prefix for `parse_output`. |
| F6 | `load_mab_filtered` | [`benchmarks/memory_agent_bench/loader.py`](benchmarks/memory_agent_bench/loader.py) | HF load + source filter + optional subsample. |
| F6 | `chunk_context` | [`benchmarks/memory_agent_bench/chunking.py`](benchmarks/memory_agent_bench/chunking.py) | Token-bounded text chunks. |
| F6 | `build_qa_pairs` | [`benchmarks/memory_agent_bench/qa_builder.py`](benchmarks/memory_agent_bench/qa_builder.py) | Dataset-specific QA extraction. |
| F6 | `metrics_summarization` | [`third_party/memoryagentbench_eval/`](third_party/memoryagentbench_eval/) | EM / ROUGE / token stats. |
| F6 | `manifest` | [`benchmarks/memory_agent_bench/manifest.py`](benchmarks/memory_agent_bench/manifest.py) | Reproducibility JSON. |
| F1 | `GovernedGeminiTransport` | [`core/gemini_transport.py`](core/gemini_transport.py) | Rate limits, key rotation, retries. |

---

## 11. How to cite this document in a paper

Suggested wording:

> Figure X presents the four-layer DARS architecture and the MemoryAgentBench evaluation pipeline. Layer A performs query reformulation and hybrid retrieval over Qdrant; Layer D owns embeddings and DARS-weighted scoring; Layers B and C implement post-hoc learning and retention maintenance respectively. The benchmark driver ingests long-context episodes from MemoryAgentBench, stores chunked memories, and scores answers with the benchmark’s official metrics.

Appendix: point readers to this file’s **Figure index** and **§10 glossary** for traceability to open-source paths.

---

## Revision note

Diagrams reflect the **current** tree under `DARS-Mini-Project/`. Research forks may add Layer A features (dual-query retrieval, neighbor chunk expansion, narrative `current_time` in rerank, tombstone filters); if your branch diverges, update **F2–F5** and extend **§10** with any new modules and env flags.
