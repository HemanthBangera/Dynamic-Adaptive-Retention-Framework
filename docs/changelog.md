# Changelog

## [Unreleased]
### Added
- Added `tests/test_loophole_audit.py` as a criticism-first verification gate spanning Layer A/B/C/D contracts and docs/runtime consistency.
- Added deterministic fail-open safety checks (timeout/non-binary response handling) and prompt XML safety checks to expose pre-hardening vulnerabilities before production benchmarking.
- Implemented **Layer C: Maintenance Manager** (`core/layer_c/`) separating orchestrator triggers (`TriageOrchestrator`), score-based decision policies (`DecisionEngine`), and distillation (`SemanticCompressor`).
- Enabled **Volume-Based Batch Processing** inside Layer C, executing the DARS pruning cycles seamlessly via limits (100-chunk limits) running concurrently across Pythons Native ThreadPoolExecutor without hanging `asyncio` loops.
- Set a **24-hour Grace Period** in Layer C `DecisionEngine` preventing low-frequency "Fresh" memories from being improperly penalized and deleted pre-maturely ensuring the $r$ (Recency) variable naturally stabilizes natively.
- Embedded strict `logger.critical` audit logging upon automated deletions in Layer C.
- Implemented **Layer B: Learning Engine** (`core/layer_b/engine.py`) orchestrating asynchronous metadata patching securely via optimistic locking and executor offloading.
- Implemented **Success Evaluator** (`core/layer_b/evaluator.py`) providing an LLM-as-a-Judge API intercept to determine utility through strict binary string signals ("YES" or "NO").
- Implemented **Score Calculator** (`core/layer_b/calculator.py`) calculating DARS metric modifications, utilizing Laplacian Smoothing for Utility ($U$) logic natively in pure Python.
- Enabled **Memory Creation Pipeline** inside Layer B handling newly discovered facts by parsing static embeddings and mathematically rendering them valid targets mapped securely to `GOAL_VECTOR`.
- Added test coverage in `test_accuracy.py` for preventing metadata reasoning leakage, preventing false positives on short-query expansions, shadow vector overhead checks, and tighter epsilon precision limits.
- Implemented **Constraint-Based Expansion** in `QueryReformulator` instructing the LLM to separate facts and strictly generate technical synonyms, halting creative inference/summarization.
- Implemented **Layer C: Maintenance Manager** (`core/layer_c/maintenance.py`) to execute memory compression logic relying on *Shadow Indexing*.
- Created `tests/test_accuracy.py` including strict hallucination testing, low-variance tracking, and shadow search compression audits.
- Implemented **Layer A: Cognitive Gateway** (`core/layer_a/`) to serve as the non-blocking asynchronous pipeline bridging raw user input and Layer D Vaults.
- Created `QueryReformulator` (`reformulator.py`) integrating Gemini API (`gemini-2.5-flash`) via `aiohttp` for async inference to expand underserved queries with strict timeouts and raw-query fallbacks.
- Created `DARSReranker` (`reranker.py`) to connect the reformulated query with `MemoryVault.search_and_rerank()` to natively apply DARS math and alpha-blending logic.
- Created `PromptConstructor` (`prompt_constructor.py`) with zero-latency f-strings to format the prompt wrapping elements in `<system_context>`, `<memory_stream>`, and `<current_user_query>` XML tags.
- Created `CognitiveGateway` (`gateway.py`) orchestration class to run the pipeline seamlessly, pushing synchronous Layer D vector logic into `asyncio.get_running_loop().run_in_executor()` to prevent event loop blocking.
- Added comprehensive unit, integration, and performance tests for Layer A (`tests/test_layer_a.py`).
- Added Gemini API integrations (`GEMINI_API_KEY`, `GEMINI_PROJECT_NUMBER`, `GEMINI_MODEL`) into `DARSConfig` and `.env.example`.
- Added a `validate_and_normalize()` class method to `DARSConfig` to mathematically enforce the unit sum of DARS weights ($w_r, w_f, w_u, w_p$) to 1.0.
- Added `aiohttp` and `pytest-asyncio` dependencies to `requirements.txt`.
- Created `docs/implementation_guide.md` and this `changelog.md` to establish DARS documentation.
- Integrated `GOAL_VECTOR` to `DARSConfig` and logic.
- Included `original_vector` inside `MemoryPayload` model to fix "Compression Blindness".

### Changed
- Changed the Epsilon precision threshold in `DARSConfig` to `1e-7` ensuring `validate_and_normalize()` mathematically stabilizes even during extreme ablation studies using tiny weights.
- Prevented **Metadata Reasoning Leakage** in `PromptConstructor` by replacing the `utility=` metadata attribute inside the generated XML with `system_weight=` to prevent LLMs from anchoring on RLHF definitions of "utility".
- Guarded against **Short-Query False Positives** in `QueryReformulator` by adding a character threshold (`len(raw_query) > 20`) before enforcing the >50% length-variance fail-open check.
- Prevented **Shadow Vector Storage Overhead** in `MaintenanceManager` mapping. `compress_memory` no longer appends `original_vector` directly into the JSON payload text to prevent Memory/RAM bloat, utilizing native Qdrant patching instead.
- Refactored `search_and_rerank` into a **Variance-Aware Scaling** system—low variance clusters (`range < 0.05`) bypass min-max scaling to allow DARS Utility (`U`) & Predictive (`P`) metrics to act as objective tie-breakers.
- Redesigned `PromptConstructor` XML metadata instructions strictly commanding the Brain LLM *not* to apologize for or reference `utility` / `last_accessed` tags to block "Metadata Misinterpretation" hallucination.
- Updated `DARSConfig.validate_and_normalize()` math validation to enforce floating-point epsilon (`1e-7` threshold) avoiding rigid `.00000000001` tolerance lockouts.
- Shifted predictive value scoring (`P`) to use cosine similarity against static `GOAL_VECTOR` on initialization.
- Replaced direct `triage_all_memories` full scan with Qdrant scroll API for batched triage processing.
- Altered `compute_utility` formula array from `S / (S + F + 1)` to `(S + 1) / (S + F + 2)` (Laplacian Smoothing) alleviating early failure drops.
- Modified `hybrid_search` array to apply min-max scaling to initial similarity search outputs before alpha-blending with DARS scores.
- Added optimistic locks to `update_utility` and `increment_frequency` using conditional payload updates.

### Fixed
- Fixed `MemoryPoint` instantiation bugs in `test_layer_a.py` by adhering to Layer D's strict formal schema rules requiring explicitly named `vector=[]` and `text_content=` arguments.
- Fixed concurrency race conditions mathematically possible during rapid DB updates.
- Fixed: Resolved authentication issues for gated models by allowing the `SentenceTransformer` within `EmbeddingEngine` to load tokens implicitly using `os.getenv("HF_TOKEN")`.