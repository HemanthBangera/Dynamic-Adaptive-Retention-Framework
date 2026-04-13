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

## Failure analysis (why these 3 failed)

### 1) Empty reformulation fallback fails
- Test: `test_reformulator_empty_generation_falls_back_to_raw_query`
- Observed: reformulator returned empty string `""` instead of raw query.
- Why it might fail:
  - Empty model output is currently treated as valid output.
  - Fallback guard triggers on timeout/error/length mismatch, but not on empty-string content.
- Research risk:
  - Retrieval can run with empty query vectors, reducing recall quality and introducing unstable benchmark outcomes.

### 2) Predictive value can be negative on ingest
- Test: `test_ingest_new_facts_predictive_value_is_bounded`
- Observed: `predictive_value = -1.0` forwarded to `store_memory()`.
- Why it might fail:
  - Cosine similarity is computed directly and forwarded without clamp/normalization.
  - The schema intent documents predictive score as $p \in [0,1]$.
- Research risk:
  - Negative predictive values distort DARS scoring and can bias retention decisions under adversarial vectors.

### 3) Maintenance errors are logged but swallowed
- Test: `test_triage_orchestrator_surfaces_maintenance_failures`
- Observed: run completed without raising; only log entry recorded.
- Why it might fail:
  - `run_maintenance()` catches broad exception and does not re-raise.
- Research risk:
  - Supervisors/schedulers cannot detect failed maintenance cycles, causing silent stale-memory accumulation.

---

## What the passes currently confirm

- Layer A query handoff intent is preserved (expanded for search, raw for user-context prompt).
- Layer C grace-period rule is functioning and avoids premature triage.
- Layer B feedback patching reaches all retrieved memories with schema-native fields.

---

## Interpretation for research progression

This verifier run shows the implementation is functionally strong in several core paths, but has three contract-level loopholes that can affect research validity:

1. Empty reformulation acceptance.
2. Unbounded predictive handoff.
3. Non-propagating maintenance failures.

These are appropriate next targets for research-grade reliability before relying on comparative benchmark claims.
