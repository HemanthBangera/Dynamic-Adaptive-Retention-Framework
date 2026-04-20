# DARS Implementation Guide

## Overview
This document serves as a comprehensive overview of the Dynamic Adaptive Retention Scoring (DARS) system implemented in Layer D.

### Core Mechanisms
1. **Utility Formula & Laplacian Smoothing:**
   - Modified from simple average to Laplacian: `U = (S + 1) / (S + F + 2)`. This ensures single early failures don't drop utility to 0 outright.
2. **Predictive Cold-Start (Goal Vector):**
   - Initialization leverages a static `GOAL_VECTOR` to assess initial predictive scoring `P` via cosine similarity, mitigating neutral cold-start biases.
3. **Score Normalization:**
   - Vector similarity scores are min-max scaled prior to the alpha-blend with DARS scores in hybrid search. This normalizes the weighting and balances the DARS impact.
4. **Compression Blindness Mitigation:**
   - Added an `original_vector` to payloads when memories are compressed, ensuring nuanced embeddings are maintained instead of overwriting raw semantic context.
5. **Batched Triage & Scroll API:**
   - Replaced a full memory load during triage with Qdrant's batched `scroll` approach preventing Out-Of-Memory exceptions and hang timers on massive datasets.
6. **Optimistic Locking:**
   - Modified `increase_frequency` and `update_utility` to use Qdrant filtering as optimistic locks to avoid concurrency overrides on metadata increments.

*(Note: These structural improvements arose from the code audit and scale review logs.)*

## Layer A: The Interaction Layer (Cognitive Gateway)
The `core/layer_a` module operates as an asynchronous orchestrator mediating raw user input and the Layer D Vault. It guarantees non-blocking processing and strict adherence to the DARS scoring parameters to enrich inputs for the primary "Brain" LLM.

### 1. Abstracting Intent Normalization (`QueryReformulator`)
**Mechanism:** Uses the Gemini REST API via `aiohttp` to pass underspecified user queries to a lightweight completion model (`gemini-2.5-flash`), deploying **Constraint-Based Expansion**. It commands the model to extract declarative facts alongside 3-5 technical synonyms related to the inquiry while strictly prohibiting "summarization" to prevent semantic loss.
**Honest Implementation Notes & Critical Concerns:**
- **Latency vs. Accuracy:** The asynchronous REST call limits block overhead, yet introducing any network call here creates an irrefutable bottleneck ranging anywhere from 0.4s to 2.5s. As a safeguard, an explicit 3.0s `aiohttp.ClientTimeout` encapsulates the POST requests.
- **Fail-Open Design & Length Validation:** Under connection failure, rate-limits, or timeout conditions, the code catches the exceptions, logs the error, and falls back to passing the raw user query completely untouched. Similarly, if the LLM output deviates in length by $>50\%$, it aggressively triggers the fallback. While this guarantees runtime stability and prevents runaway hallucinations, it degrades vector retrieval accuracy during those failed intervals, removing the "Cognitive" step for that specific inference.

### 2. DARS Scoring & Reranking Engine (`DARSReranker`)
**Mechanism:** Ties the expanded user intent query to `MemoryVault.search_and_rerank()` from Layer D. It queries 15 candidates (`DEFAULT_FETCH_K`) via semantic search, applies **Variance-Aware Min-Max normalization** scaling `[0, 1]`, and returns the best 3 targets (`DEFAULT_TOP_N`) via hybrid DARS alpha-blending ($\alpha = 0.5$).
**Honest Implementation Notes & Critical Concerns:**
- **Variance-Aware Tiebreaking:** If the 15 gathered semantic candidates return very similar closeness scores (e.g. `max_sim - min_sim < 0.05`), Min-Max scaling is artificially set to `1.0`. This prevents microscopic float fluctuations from crashing similarities entirely to 0.0, allowing DARS metrics to natively decide ties objectively.
- **Mathematical Constriction:** DARS relies precisely on proper distribution metrics mathematically ($S = \omega_r R + \omega_f F + \omega_u U + \omega_p P$). `DARSConfig.validate_and_normalize()` strictly bounds configuration inputs via a floating-point Epsilon (`1e-7`) to force the sum to `1.0`. Without it, scores could arbitrarily bloat rendering alpha blending useless.
- **Thread Execution Sync:** Qdrant Client libraries executing standard `search()` are fundamentally synchronous blocking calls. To avoid disrupting the asyncio event loop during high-throughput batches, `search_and_rerank` must explicitly be pushed off onto `asyncio.get_running_loop().run_in_executor()`.

### 3. Emancipating Content Through XML (`PromptConstructor`)
**Mechanism:** Employs standard Python `f` strings dynamically rendering `<system_context>`, injecting each target into distinct `<memory>` tags containing core metadata `system_weight` and `last_accessed` payload attributes, and enveloping the raw user target in `<current_user_query>`.
**Honest Implementation Notes & Critical Concerns:**
- **Prompt Confusion Vulnerability (Metadata Misinterpretation):** Separating system directives from augmented contexts drastically narrows hallucination probability in Brain LLMs. However, LLMs often interpret numerical metadata (like `utility="0.12"`) as operational commands, frequently responding by "apologizing" for retrieving low-utility data. To mitigate this, `<system_context>` strictly instructs the Brain LLM to treat these uniquely as "historical reliability records" to bypass hallucinated apologies, and we completely scrub the word `utility` from the prompt, replacing it with `system_weight` to prevent pre-trained RLHF logic from anchoring negatively.
- **Schema Strictness:** Layer D's memory payload relies distinctly on parameter names (`text_content`, `vector`). The Prompt Constructor inherently couples directly with `MemoryPayload` and fails instantly upon schema drifts or dynamic typing violations.

## Layer B: The Learning Engine (Experience Feedback Loop)
**Mechanism:** Operates entirely asynchronously behind the scenes using `asyncio.create_task()` to intercept conversational output without pausing user generation logic. 
- **The Judge (`SuccessEvaluator`):** Acts as an "LLM-as-a-Judge" pointing `gemini-2.5-flash` at the Query, Output, and Retrieved Memories.
  - *Judge Prompt:* 
    ```
    You are the DARS Success Evaluator.
    USER QUERY: {query}
    AGENT RESPONSE: {response}
    RETRIEVED MEMORIES: {memories}
    EVALUATION TASK: Did the provided memories actually help the agent answer the user query accurately?
    Respond ONLY with 'YES' or 'NO'. No explanation.
    ```
  - *Fail-Open Logic:* Any non-binary hallucination or timeout results in a "NEUTRAL" signal, bypassing patches to protect DB math logic.
- **The Calculator (`ScoreCalculator`):** Transforms the Judge's binary signal into pure numbers. 
  - To prevent early zero-drops (like penalizing 0 Successes and 1 Failure as a 0% utility), DARS shifts to **Laplacian Smoothing**: $U = (S + 1) / (S + F + 2)$.
- **The Orchestrator (`LearningEngine`):** Receives the calculated dictionary and performs zero-latency `patch_payload` commands offloaded onto the main event loop's ThreadPoolExecutor (`run_in_executor`) to prevent blocking asynchronous I/O calls to Qdrant REST endpoints.

**Honest Implementation Notes & Critical Concerns:**
- **Metadata Framing Bias:** We explicitly eliminated the term **"Utility"** from any prompts to the Judge, instead leaning entirely on "Historical Relevance" or simply binary logic to judge correctness. RLHF models natively struggle avoiding implicit judgments when words like "utility" are present, actively sabotaging data-driven systems.

## Layer C: The Maintenance Engine & Shadow Indexing
**Mechanism:** Layer C acts as the background orchestration layer for memory health (`core/layer_c`). It controls long-term data efficiency and removes "Retrieval Noise" by executing batched pruning and compressions evaluated against the current DARS equation. 
- **The Scheduler (`TriageOrchestrator`):** Implements a dual-trigger architecture. Volume-Based maintenance begins exclusively when the DB point count exceeds `MAX_MEMORY_THRESHOLD` (e.g., 1000). It pulls data via `MemoryVault.get_all_memories(scroll_yield=True)`, preventing memory spikes by streaming 100-point chunks incrementally onto the ThreadPoolExecutor.
- **The Janitor (`DecisionEngine`):** Employs strict mathematically-bounded decision logic mapped securely with `1e-7` epsilon bounds. It guarantees memories remain completely untouched (`RETAIN`) if $S > 0.7$, summarizes text logic (`COMPRESS`) when $0.3 \le S \le 0.7$, and purges permanently (`DELETE`) if $S < 0.3$. To prevent deleting new mappings pre-maturely, an active 24-hour "Grace Period" protects all files created or modified recently. 
- **The Distiller (`SemanticCompressor`):** Gemini 2.5 Flash compresses data guided explicitly by the prompt: `Summarize... Preserve all proper nouns, technical terms... Strip conversational filler... Output ONLY the summarized bullet point.`

**Honest Implementation Notes & Critical Concerns:**
- **Storage Bloat Prevention via Shadow Indexing:** Adding an `original_vector` explicitly to the JSON payload effectively doubles the storage cost for those points (leading to bloated RAM and OOM crashes at scale). For this project, memory compression via shadowing is executed *without* appending the vector into the JSON payload natively; Qdrant's payload set operations strictly leave the existing high-res vector completely intact while allowing the new compressed `text_content` payload and the `original_text_backup` log to swap underneath it seamlessly. This absolutely eliminates "Retrieval Blindness" without causing memory blowouts.
- **Logging Audits:** Due to the explicit danger of permanent deletions occurring automatically upon threshold drops, any $S < 0.3$ trigger is logged structurally with `logger.critical()` noting the ID and DARS score to safeguard data audits natively.

## Verification Interpretation (How to Read Test Results)
- **Baseline Functional Tests:** Confirm that implemented workflows currently execute as intended under standard and mocked conditions.
- **Loophole Audit Tests:** Intentionally stress contract boundaries between layers (A/B/C/D), fail-open behavior, optimistic-lock conflict handling, prompt safety, and docs/runtime consistency.
- A **PASS** in baseline tests means the path is operational.
- A **FAIL** in loophole audit tests means there is a real pre-production risk (schema drift, silent conflict, injection surface, or policy inconsistency) that should be fixed before trusting benchmark metrics.
## Adversarial Input Handling
Clamping predictive values and null-byte sanitation added.

## Context Window Management
Enforces 20,000 chars limit. Outliers tagged with system:high_priority_distillation.

## Termination-Aware background tasks
Graceful shutdown logic.

## High-Priority Repair Loop
With Semaphore-based rate limiting.
