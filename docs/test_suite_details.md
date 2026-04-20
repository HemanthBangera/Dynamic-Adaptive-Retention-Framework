# DARS Framework - Complete Test Suite Documentation

This document explains every major test group in the project and, critically:
1. **Why the test exists**
2. **What behavior it validates**
3. **What it means when that test passes**

Current verification split:
- **Baseline functional suite:** 102 tests (latest baseline run: passing)
- **Loophole audit suite:** 10 criticism-first contract tests (latest run: passing)
- **Verifier research audit suite (Phase-1):** 6 research-grade tests (latest run: passing)
- **Verifier lifecycle audit suite (Phase-2):** 7 adversarial production-risk tests (latest run: expected mixed/failing until hardened)

Total tests present in repository: **125**

These suites are complementary:
- Baseline tests verify implemented behavior works.
- Loophole tests verify architecture contracts are not drifting.

---

## 1. Accuracy & Fallback Heuristics (`tests/test_accuracy.py`)
Responsible for ensuring system prompts and inference safety guardrails remain deterministic, preventing RLHF models from poisoning system metrics.

| Test Case | Why it exists | What it validates | What pass means |
| :--- | :--- | :--- |
| `test_hallucination_proper_noun_retention` | Proper nouns are high-risk for LLM distortion. | Reformulator keeps critical entities intact. | Retrieval remains anchored to exact domain terms. |
| `test_low_variance_scaling` | Similarity ranges can collapse ranking signal. | Low-variance bypass protects DARS tie-breaking. | Reranking stays stable when vectors are nearly identical. |
| `test_shadow_search_compression` | Compression can cause retrieval blindness. | Shadow indexing preserves searchability after summarization. | Compression does not destroy discoverability. |
| `test_system_weight_prompt_metadata` | Metadata framing can bias model behavior. | Prompt uses `system_weight` contract, not utility leakage text. | Prompt structure follows safety policy for model steering. |
| `test_short_query_false_positive_fix` | Short queries can falsely trip length guards. | Threshold logic avoids invalid fallback on short inputs. | Valid short expansions are accepted instead of discarded. |
| `test_epsilon_precision` | Tiny float drift can break config checks. | `1e-7` tolerance handles precision safely. | Weight normalization remains mathematically safe. |

---

## 2. The Cognitive Gateway - Layer A (`tests/test_layer_a.py`)
Responsible for measuring the asynchronous inference mapping spanning the prompt orchestration and query expansion logic.

| Test Case | Why it exists | What it validates | What pass means |
| :--- | :--- | :--- |
| `test_reformulate_query` | Reformulation is the first retrieval gate. | Gemini response handling and transformed query output. | Layer A can enrich underspecified user intent. |
| `test_dars_reranking_logic` | Retrieval quality depends on blended scoring. | Hybrid ranking path integrates semantic + DARS logic. | Returned order reflects configured reranking policy. |
| `test_xml_schema_validation` | Prompt structure controls downstream model grounding. | XML envelope and memory tag composition are correct. | Brain LLM receives valid structured context. |
| `test_performance_bottleneck` | Async pathways must avoid avoidable blocking. | Concurrent gateway calls complete within expected envelope. | Gateway orchestration is suitable for multi-query usage. |

---

## 3. Experience Engine - Layer B (`tests/test_layer_b.py`)
Validates the LLM-Judge intercepts tracking experience feedback logic effectively back to Memory storage limits asynchronously.

| Test Case | Why it exists | What it validates | What pass means |
| :--- | :--- | :--- |
| `test_success_evaluator_yes` | Judge signal controls downstream learning updates. | Binary YES response handling from evaluator. | Positive feedback can trigger memory credit updates. |
| `test_laplacian_smoothing_calculator` | Raw success/failure ratios are too brittle early on. | Laplacian update formula produces expected utility. | Learning math is robust to sparse history. |
| `test_atomic_patch_execution` | Background learning must not corrupt payload updates. | Feedback loop calls patch path correctly. | Layer B can update memory metadata safely. |

---

## 4. Maintenance Manager - Layer C (`tests/test_layer_c.py`)
Verifies the background orchestration engine correctly executes pruning, compression, and retention cycles based on DARS scores, while respecting data integrity and performance limits.

| Test Case | Why it exists | What it validates | What pass means |
| :--- | :--- | :--- |
| `test_retention_policy` | High-value memories must survive maintenance cycles. | RETAIN branch leaves important data intact. | Triage does not over-prune useful knowledge. |
| `test_deletion_policy` | Low-value noise must be removable for hygiene. | DELETE branch actually removes target points. | Memory bloat control path works as designed. |
| `test_compression_shadow_integrity` | Compression must save tokens without killing recall. | Summary + backup + semantic retrieval invariants hold. | Shadow indexing preserves practical retrieval fidelity. |
| `test_volume_trigger` | Maintenance should not run continuously. | Threshold scheduler only runs above capacity trigger. | Background triage is controllable and resource-aware. |

---

## 5. Qdrant Storage Vault - Layer D Integration (`tests/test_integration.py`)
End-to-End vector search tests validating Qdrant native DB instantiation, vector mapping limits, and scroll extraction API batches.

| Test Category Suite | Why it exists | What it validates | What pass means |
| :--- | :--- | :--- |
| `TestCollectionManagement` | Storage initialization is a prerequisite for all layers. | Collection lifecycle and vector dimension contract. | Database substrate is valid and reachable. |
| `TestMemoryCRUD` | Core persistence correctness is non-negotiable. | Create/read/update/delete behaviors across points. | Memory lifecycle operations are reliable. |
| `TestSemanticSearch` | Retrieval is the core runtime capability. | KNN + scoring output format and ordering basics. | Search path is functional end-to-end. |
| `TestAtomicUpdates` | Multi-field metadata must evolve safely. | Recency/frequency/utility update primitives. | Layer B/C can mutate payload state safely. |
| `TestTriageScan` | Full-dataset maintenance requires streaming scans. | Scroll-based traversal and triage preparation path. | Large-collection maintenance is operationally feasible. |

---

## 6. DARS Metric Equations (`tests/test_scoring.py`)
Isolates pure calculation functions tracking `$S = \omega_r R + \omega_f F + \omega_u U + \omega_p P$` directly against mathematically known boundaries.

| Test Category Suite | Why it exists | What it validates | What pass means |
| :--- | :--- | :--- |
| `TestRecency` | Time-decay mistakes can bias all decisions. | Decay response across realistic and edge timestamps. | Recency signal is mathematically stable. |
| `TestFrequency` | Frequency normalization affects retention pressure. | Growth and cap behavior across access counts. | Frequency contributes predictably to DARS score. |
| `TestUtility` | Utility is central to learning credit assignment. | Success/failure histories map to expected utility. | Utility term behaves correctly across outcomes. |
| `TestDARSScore` | Final score formula must be internally consistent. | Weighted sum, clamping, and missing-key defaults. | DARS score remains bounded and dependable. |
| `TestRetentionClassification` | Wrong thresholding creates data loss/noise retention. | RETAIN/COMPRESS/DELETE boundary classification. | Policy decisioning follows documented thresholds. |

---

## 7. Vector Processing Embedder (`tests/test_embedding.py`)
Responsible for local embedding generations mapping input queries against `SentenceTransformer` vector dimensions.

| Key Functions Checked | Why it exists | What it validates | What pass means |
| :--- | :--- | :--- |
| Dimensionality Constraints | Qdrant collection schema requires fixed vector size. | Embedding outputs always match expected dimensions. | Stored points remain index-compatible. |
| Vector Quality Range | Similarity behavior must be numerically sane. | Identity and separation behavior in cosine space. | Embeddings carry meaningful semantic signal. |
| Batch Mapping Iteration | Runtime workloads are batched in practice. | Single vs batch embedding behavior consistency. | Inference pipeline can scale beyond single inputs. |

---

## 8. Schema Validation Contracts (`tests/test_schema.py`)
Checks Pydantic-style mapping bounds strictly guarding `MemoryPayload` injection pipelines.

| Test Category Suite | Why it exists | What it validates | What pass means |
| :--- | :--- | :--- |
| `TestMemoryPayload` | Payload drift silently breaks all layers. | Serialization, defaults, and compute behaviors. | Payload contract is stable for DB/storage operations. |
| `TestMemoryPoint` | Point identity integrity is fundamental. | ID generation format and uniqueness. | Storage/retrieval references remain reliable. |
| `TestDARSWeights` | Invalid weights corrupt all downstream decisions. | Validation of sum constraints and custom vectors. | Scoring remains mathematically coherent. |

---
## 9. Loophole Audit Gates (`tests/test_loophole_audit.py`)
Criticism-first verification suite intended to detect architecture drift before the next stage. These tests are deterministic and offline (mock-based), and they are expected to fail until contracts are fixed.

| Test Case | Why it exists | What it validates | Current Result |
| :--- | :--- | :--- |
| `test_score_calculator_emits_schema_compatible_keys` | Cross-layer field drift causes silent scoring errors. | Layer B emits Layer D-compatible keys. | PASSED |
| `test_learning_engine_feedback_loop_patches_schema_fields` | Patch contract must match storage schema. | Feedback updates use schema-native fields. | PASSED |
| `test_learning_engine_ingest_new_facts_forwards_predictive_value_to_storage` | New facts should not lose predictive signal. | Ingest path forwards computed predictive value. | PASSED |
| `test_success_evaluator_non_binary_returns_neutral` | Judge hallucinations must not poison learning. | Non-binary outputs fail-open to NEUTRAL. | PASSED |
| `test_reformulator_timeout_returns_raw_query` | Network failures must not block responses. | Timeout fallback returns original query. | PASSED |
| `test_reformulator_long_query_length_guard_fallback` | Expansion runaway must be constrained. | Length-guard fallback behavior on long queries. | PASSED |
| `test_prompt_constructor_escapes_xml_special_characters` | Prompt injection can corrupt model grounding. | XML content escaping for payload/query text. | PASSED |
| `test_increment_frequency_detects_conflict_when_update_not_applied` | Silent lock misses can drop updates undetected. | Conflict detection for optimistic frequency update. | PASSED |
| `test_update_utility_detects_conflict_when_update_not_applied` | Learning correctness requires lock integrity. | Conflict detection for utility update path. | PASSED |
| `test_docs_and_settings_have_single_epsilon_policy` | Docs drift creates wrong implementation assumptions. | Changelog/guide match runtime epsilon policy. | PASSED |

---
## 10. Verifier Phase-2 Lifecycle Audit (`tests/test_verifier_phase2_lifecycle_audit.py`)
Production-mimicking, dummy-data **adversarial verifier gate** validating memory-state evolution over time, payload rigidity under dirty inputs, idempotent maintenance re-entry, and cross-layer resilience. This suite is intentionally failure-seeking.

| Test Case | Why it exists | What it validates | Current Result |
| :--- | :--- | :--- | :--- |
| `test_phase2_time_travel_neutral_memory_should_be_delete_candidate_by_day3` | Real systems age; stale neutral memory must decay fast enough. | Whether stale neutral memory becomes delete-eligible by day-3. | FAILED |
| `test_phase2_payload_null_bytes_must_be_removed_from_prompt` | Dirty payloads can break downstream parser/tooling. | Null-byte sanitation in prompt construction. | FAILED |
| `test_phase2_payload_large_text_must_respect_prompt_size_budget` | Production can send very large context blocks. | Prompt-size budget enforcement under stress. | FAILED |
| `test_phase2_reentry_triage_should_continue_processing_despite_one_point_failure` | One bad record should not abort full maintenance cycle. | Best-effort per-point triage resilience in re-entry runs. | PASSED |
| `test_phase2_ingest_new_facts_must_call_embedder_with_string_not_list` | Type-shape drift can break real embedding backends. | Strict single-text embedder invocation contract. | FAILED |
| `test_phase2_store_memory_must_clamp_out_of_range_predictive_values` | Invalid predictive values can poison retention policy. | Input-bound clamping before persistence. | FAILED |
| `test_phase2_feedback_loop_should_be_best_effort_when_one_patch_conflicts` | Partial conflicts happen in production concurrency. | Whether one conflict stops all downstream updates. | FAILED |

### 10.1 Current Failure Report (for bug-fix planning)

Latest run command:

- `.venv\Scripts\pytest.exe tests/test_verifier_phase2_lifecycle_audit.py -q --tb=short`
- Result: **6 failed, 1 passed**

| Failed Test | What this test checks | Observed failure | What this failure means | Production risk if unresolved |
| :--- | :--- | :--- | :--- | :--- |
| `test_phase2_time_travel_neutral_memory_should_be_delete_candidate_by_day3` | A neutral, unused memory should decay enough by Day-3 to be delete-eligible. | Computed score at Day-3 is `0.396026` (expected `<= 0.3`). | Current decay/weight policy is too conservative for stale-neutral memories. | Stale memories may remain retrievable too long, increasing retrieval noise and outdated-context usage. |
| `test_phase2_payload_null_bytes_must_be_removed_from_prompt` | Prompt builder should sanitize null bytes (`\x00`) from memory/query content. | Null byte remains in final prompt string. | XML escaping alone is insufficient for control-character hardening. | Downstream model parsers, logging sinks, or middleware can behave unpredictably with null bytes. |
| `test_phase2_payload_large_text_must_respect_prompt_size_budget` | Prompt assembly should enforce a strict context-size budget under large payload stress. | Prompt length is `120893` (expected `<= 20000`). | No effective prompt-size guard/truncation policy is currently enforced. | Token explosion, latency spikes, higher cost, and context-window overflow failures in production. |
| `test_phase2_ingest_new_facts_must_call_embedder_with_string_not_list` | `LearningEngine.ingest_new_facts()` should use single-text embed call contract. | Actual call is `encode(['fact-1'])` instead of `encode('fact-1')`. | API-contract mismatch between caller and embedder for single-item ingestion path. | Type/shape drift with real embedding backends; possible runtime errors or silent vector-shape inconsistencies. |
| `test_phase2_store_memory_must_clamp_out_of_range_predictive_values` | Directly supplied predictive values must be clamped to `[0,1]` before write. | Stored payload predictive remains `3.7`. | Caller-provided predictive input is trusted without boundary enforcement. | Out-of-range predictive values can distort DARS score and triage policy decisions. |
| `test_phase2_feedback_loop_should_be_best_effort_when_one_patch_conflicts` | Feedback loop should continue patching remaining memories after one conflict. | Updates attempted only for `['m1', 'm2']`; `m3` was not attempted. | Patch loop aborts on first exception; no best-effort continuation strategy. | Partial state updates during contention can bias utility/frequency history and reduce learning consistency. |

### 10.2 Planning Notes for Remediation

- Treat all six failures as **production-readiness blockers** rather than test-only issues.
- Recommended code focus areas:
	- `core/layer_d/storage.py` (predictive clamping and score/decay policy behavior)
	- `core/layer_a/prompt_constructor.py` (control-character sanitation and size budgeting)
	- `core/layer_b/engine.py` (embedder API contract and best-effort patch continuation)
- Keep these tests adversarial; do not weaken assertions to achieve green status.

---
### How to interpret this report
- If a **baseline test** passes, implemented behavior is currently functional for that pathway.
- If a **loophole audit test** fails, it indicates a contract, safety, or consistency risk that may not be visible in normal happy-path testing.
- If a **Phase-2 lifecycle verifier test** fails, it indicates a production-risk vector (state-evolution, payload-rigidity, re-entry/idempotency, or conflict handling) that must be hardened before real-data rollout.
- Progression to next stage should prioritize closing FAILED loophole gates before relying on benchmark outcomes.

**Summary:** Baseline functional coverage is 102 tests, loophole audit is 10 tests, Phase-1 verifier is 6 tests, and Phase-2 lifecycle verifier is 7 adversarial tests intentionally designed to expose pre-production failures.
