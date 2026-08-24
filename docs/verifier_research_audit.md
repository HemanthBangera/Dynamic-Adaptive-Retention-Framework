# Verifier Research Audit (Core-Goal Focused)

This audit adds a focused verifier suite for research validation, not production hardening.

- Test file: `tests/test_verifier_research_audit.py`
- Scope: Layer goals + Layer-to-Layer handoff contracts (A→B→C→D)
- Latest run: **6 passed, 0 failed**

Command used:

- `.venv\Scripts\pytest.exe tests/test_verifier_research_audit.py -v`

---

## Why this suite exists

The existing baseline suite proves many implemented paths work. This verifier suite is intentionally narrower:

1. Validate **core research behavior contracts**.
2. Detect **silent drift** between layers.
3. Explain failures in terms of **research risk**, not only code defects.

---

## Added verifier tests and outcomes

| Test Case | Layer Focus | What the test does | Result |
| :--- | :--- | :--- | :--- |
| `test_reformulator_empty_generation_falls_back_to_raw_query` | Layer A | Mocks Gemini returning empty text and checks fail-open fallback to raw query. | PASSED |
| `test_ingest_new_facts_predictive_value_is_bounded` | Layer B → D | Verifies computed predictive score passed to storage remains in $[0,1]$. | PASSED |
| `test_triage_orchestrator_surfaces_maintenance_failures` | Layer C orchestration | Forces vault scroll failure and checks error is surfaced to caller. | PASSED |
| `test_gateway_handoff_uses_expanded_query_for_search_but_raw_for_prompt` | Layer A interaction | Confirms expanded query is used for retrieval and raw query is preserved in prompt. | PASSED |
| `test_decision_engine_skips_fresh_memories_during_grace_period` | Layer C policy | Verifies fresh memories are skipped under 24h grace policy. | PASSED |
| `test_feedback_loop_patches_all_retrieved_memories` | Layer B feedback loop | Confirms all retrieved memories are patched with schema-compatible updates. | PASSED |

---

## Issues found by this suite, and how they were resolved

These three contract violations were what the suite was written to expose. All three were
**observed on the first run and have since been fixed**; the suite now passes 6/6 against the
current code. This section is retained as the record of what was found and where it was corrected.

### 1) Empty reformulation was accepted instead of triggering fallback
- Test: `test_reformulator_empty_generation_falls_back_to_raw_query`
- Observed at audit time: reformulator returned an empty string `""` instead of the raw query.
- Cause: empty model output was treated as a valid expansion; the fallback guard fired on
  timeout, error and length drift, but not on empty content.
- Research risk if unfixed: retrieval could run on an empty query vector, reducing recall and
  destabilising benchmark outcomes.
- **Resolved** — `core/layer_a/reformulator.py:102` now treats a falsy result as a failure and
  returns the raw query (`reformulator.py:133`).

### 2) Predictive value could be negative on ingest
- Test: `test_ingest_new_facts_predictive_value_is_bounded`
- Observed at audit time: `predictive_value = -1.0` forwarded to `store_memory()`.
- Cause: cosine similarity was forwarded without clamping, though the schema documents the
  predictive score as $p \in [0,1]$.
- Research risk if unfixed: negative predictive values distort the DARS composite and bias
  retention decisions under adversarial vectors.
- **Resolved** — clamped to $[0,1]$ at `core/layer_d/storage.py:294` and `storage.py:363`.

### 3) Maintenance errors were logged but swallowed
- Test: `test_triage_orchestrator_surfaces_maintenance_failures`
- Observed at audit time: the run completed without raising; only a log entry was recorded.
- Cause: `run_maintenance()` caught a broad exception without re-raising.
- Research risk if unfixed: schedulers cannot detect failed maintenance cycles, allowing silent
  accumulation of stale memories.
- **Resolved** — the handler at `core/layer_c/triage.py:147` now re-raises after logging
  (`triage.py:149`).

---

## What the passes currently confirm

- Layer A query handoff intent is preserved (expanded for search, raw for user-context prompt).
- Layer C grace-period rule is functioning and avoids premature triage.
- Layer B feedback patching reaches all retrieved memories with schema-native fields.

---

## Interpretation for research progression

The suite was written to expose contract-level loopholes that could affect research validity. It
found three:

1. Empty reformulation acceptance.
2. Unbounded predictive handoff.
3. Non-propagating maintenance failures.

All three have been corrected at the code locations cited above, and the suite now passes 6/6. The
cross-layer contracts it covers — Layer A query handoff, Layer B feedback patching, Layer C grace
period and error propagation, and the bounded predictive handoff into Layer D — therefore hold
against the current implementation.

Note on scope: these tests verify architectural contracts, not retrieval quality. They establish
that the layers interact as specified; they do not establish that the DARS composite score improves
retrieval accuracy. That question requires the baseline and ablation experiments listed under Scope
for Future Work in the manuscript.
