# Verifier Research Audit (Core-Goal Focused)

This audit adds a focused verifier suite for research validation, not production hardening.

- Test file: `tests/test_verifier_research_audit.py`
- Scope: Layer goals + Layer-to-Layer handoff contracts (A→B→C→D)
- Latest run (2026-09-11, local Qdrant, full suite `pytest tests`): **6 passed, 0 failed**
  (as part of 187 passed, 0 skipped)

Command used:

- `python -m pytest tests/test_verifier_research_audit.py -v`

---

## Why this suite exists

The existing baseline suite proves many implemented paths work. This verifier suite is intentionally narrower:

1. Validate **core research behavior contracts**.
2. Detect **silent drift** between layers.
3. Explain failures in terms of **research risk**, not only code defects.

These tests verify architectural contracts, not retrieval quality. They establish
that the layers interact as specified; they do not establish that the DARS
composite score improves retrieval. That question is answered by the revision's
experiments (baselines, ablations, multi-session studies), not by this suite.

---

## Tests and outcomes

| Test Case | Layer Focus | What the test does | Result |
| :--- | :--- | :--- | :--- |
| `test_reformulator_empty_generation_falls_back_to_raw_query` | Layer A | Mocks the LLM returning empty text and checks fail-open fallback to the raw query. | PASSED |
| `test_ingest_new_facts_predictive_value_is_bounded` | Layer B → D | Ingests a fact against a goal vector exactly anti-aligned with it (cosine = −1) on a local vault and checks the persisted predictive value is in $[0,1]$. | PASSED |
| `test_triage_orchestrator_surfaces_maintenance_failures` | Layer C orchestration | Forces a vault scroll failure and checks the error is surfaced to the caller. | PASSED |
| `test_gateway_handoff_uses_expanded_query_for_search_but_raw_for_prompt` | Layer A interaction | Confirms the expanded query is used for retrieval and the raw query is preserved in the prompt. | PASSED |
| `test_decision_engine_skips_fresh_memories_during_grace_period` | Layer C policy | Verifies fresh memories are skipped under the grace-period policy. | PASSED |
| `test_feedback_loop_patches_all_retrieved_memories` | Layer B feedback loop | Confirms every retrieved memory receives the three version-guarded Layer D updates (utility, frequency, recency) and that no unguarded `patch_payload` write is used. | PASSED |

---

## History of this suite

### Issues found when the suite was first written (all fixed)

1. **Empty reformulation accepted instead of triggering fallback** — the reformulator
   returned `""`. Fixed: an empty result is treated as a failure and the raw query is
   returned (`core/layer_a/reformulator.py`).
2. **Predictive value could be negative on ingest** — cosine similarity was forwarded
   unclamped. Fixed: clamped to $[0,1]$ in `MemoryVault._initial_predictive`
   (`core/layer_d/storage.py`).
3. **Maintenance errors logged but swallowed** — fixed: `run_maintenance()` now
   triages every point best-effort and then raises one `RuntimeError` summarising
   the failures (`core/layer_c/triage.py`).

### Stale tests found in the 2026-09-11 audit (corrected)

Before the audit, this document and commit `871c8b7` reported 6/6 passing, but a
re-run showed **2 of 6 failing**, because two tests had drifted from the code:

- `test_ingest_new_facts_predictive_value_is_bounded` patched
  `DARSConfig.GOAL_VECTOR`, an attribute that no longer exists (the goal vector is now
  produced by `DARSConfig.get_goal_vector()`), and expected a `predictive_value`
  argument that `ingest_new_facts` no longer passes. Rewritten to exercise the real
  ingestion path on a local vault.
- `test_feedback_loop_patches_all_retrieved_memories` expected blind `patch_payload`
  calls, but the feedback loop uses the version-guarded `update_utility`,
  `increment_frequency` and `update_recency` (a deliberate fix, BUG #5). Rewritten to
  assert those calls.

Both corrections keep the original intent of the tests.
