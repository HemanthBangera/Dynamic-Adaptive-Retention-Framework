# Pre-registration addendum — confirmatory MSC study of write-side retention signals

**Status: FROZEN (v1.0, 2026-09-13 UTC).** Frozen before any run on the data it describes. Its SHA-256 is
recorded in `experiments/preregistration_addendum.md.sha256`. `benchmarks/dars_eval/confirm_msc.py confirm`
refuses to run unless this file says FROZEN and matches that hash. Any later change goes into
`experiments/deviations_post_freeze.md`; this file is never edited.

The original pre-registration (`experiments/preregistration.md`, SHA-256 `a3dec99a…a3965270`) is unchanged
and remains the protocol for H1–H6.

## 1. Why this addendum exists

The fourth audit of the finished test phase (deviations log, 2026-09-12) found a pattern across datasets. The
retention signals that the DARS vault updates on *retrieval* — last-access recency, access count and
retrieval-gated utility — were weak or even anti-predictive of later need. By contrast, signals recorded when a
fact is *written* — when it was created, how often it was restated — were strong:

| MSC test (E9) | AUROC for session-3 restatement |
|---|---|
| access count (LFU) | 0.243 |
| last access (LRU) | 0.664 |
| DARS, pre-registered selection | 0.891 |
| creation order (FIFO) | 0.866 |
| logistic model on creation session and mention count (fitted on dev) | 0.917 |

**This hypothesis was generated after seeing test results.** It therefore cannot be confirmed on those data. It
is tested here on data that no part of the study has touched.

## 2. Data

- **Source:** `nayohan/multi_session_chat`, Hugging Face revision `78b67491c43823fc169cab827ab3f82805e0235b`.
- **Splits:** `validation` (500 dialogues) and `test` (501 dialogues), all with sessions 0–4. Every earlier run
  loaded only `train` (`run_msc.load_msc_dialogues`).
- **Checks made before freezing, without computing any label or score:**
  - the schema is identical to train;
  - zero dialogues overlap with train or with each other (SHA-256 of normalised session-0 personas and of
    session-1 turns).
- **Clusters:** dialogues, identified as `validation:<id>` and `test:<id>`. The two splits are analysed
  together; per-split results are secondary.

## 3. Protocol

The E9 protocol, unchanged: `run_msc run` defaults (k = 3 similarity retrieval, 24-hour session gap, 60-second
turn step, feedback τ = 0.5, deduplication τ = 0.5, MSC goal vector, λ = 0.005 h⁻¹ in the vault), plus
`--record-writes`.

| Horizon | Streamed sessions | Label | Status |
|---|---|---|---|
| A | 1–2 | session-3 restatement | **primary** (identical to E9) |
| B | 1–3 | session-4 restatement | secondary |

```
python -m benchmarks.dars_eval.run_msc run --hf-split validation --split all --record-writes --label-session 3 --out benchmark_runs/revision/addendum/msc_validation_s3
python -m benchmarks.dars_eval.run_msc run --hf-split test       --split all --record-writes --label-session 3 --out benchmark_runs/revision/addendum/msc_test_s3
python -m benchmarks.dars_eval.run_msc run --hf-split validation --split all --record-writes --label-session 4 --out benchmark_runs/revision/addendum/msc_validation_s4
python -m benchmarks.dars_eval.run_msc run --hf-split test       --split all --record-writes --label-session 4 --out benchmark_runs/revision/addendum/msc_test_s4
python -m benchmarks.dars_eval.importance --facts <each run>/facts.jsonl --out benchmark_runs/revision/addendum/importance.jsonl
python -m benchmarks.dars_eval.confirm_msc confirm --primary --config experiments/addendum_config.json --runs <validation_s3> <test_s3> --importance <ratings> --out benchmark_runs/revision/addendum/confirm
python -m benchmarks.dars_eval.confirm_msc confirm --config experiments/addendum_config.json --runs <validation_s4> <test_s4> --importance <ratings> --out benchmark_runs/revision/addendum/confirm
```

Importance calls may be delivered through the Batch API into the same response cache, as in the main test phase.

## 4. Scores (all computed at the end of the stream; `benchmarks/dars_eval/retention_signals.py`)

| Score | Definition |
|---|---|
| `read_h4`, `read_h5` | the pre-registered E9 DARS score (vault components; R recomputed from last access), with the frozen H4 and H5 selections |
| `write_h4`, `write_h5` | the same weighted sum over **write-side components** (below), with configurations selected on the MSC dev split |
| `utility_read` | vault U = (successes + 1) / (successes + failures + 2), credited on retrieval |
| `utility_write` | U_w below |
| `generative_agents` | 0.995^(hours since last retrieval) plus the importance rating, each min–max normalised within the dialogue, equal weights. There is no query at retention time, so no relevance term |
| `memorybank` | exp(−days since last recall / S), where S = 1 + number of recalls |
| `metadata_model` | logistic model on (sessions since creation, mention count), fitted on dev |
| baselines | `recency_lru` (last access), `fifo` (creation time), `frequency_lfu` (access count), `mention_count` |

**Write-side components:**
- R_w = exp(−λ · hours since the fact was last stated in a persona summary);
- F_w = ln(1 + times stated) / ln 51, capped at 1;
- U_w = (distinct later sessions whose summary restated it + 1) / (later sessions it could have appeared in + 2);
- P is unchanged.

**Selected configurations,** from `benchmark_runs/revision/addendum/selection_dev.json` (SHA-256
`8b7871ea71035c97376d6cb5cf5b166088a3a1faac25c67be3897cafacc26192`), using the same grid (0.1-step simplex ×
λ ∈ {0.0005, 0.001, 0.005, 0.01, 0.05}) and the same selection rule as E9:

| Configuration | Values | Dev value |
|---|---|---|
| `write_h4` | λ = 0.05, weights (0.2, 0.1, 0.5, 0.2) | AUROC 0.922 |
| `write_h5` | λ = 0.01, weights (0.4, 0.0, 0.3, 0.3) | harmful deletion 0.076 |
| `read_h4` (frozen) | λ = 0.05, weights (0.2, 0, 0.8, 0) | AUROC 0.894 (re-derived exactly on the re-run dev rows) |
| `read_h5` (frozen) | λ = 0.0005, weights (0.3, 0, 0.7, 0) | harmful deletion 0.102 (re-derived exactly) |
| metadata model | intercept −0.9588; coefficients −2.9019 (sessions since creation), 2.7067 (mentions); `sklearn` `LogisticRegression()` defaults | — |

All values are in `experiments/addendum_config.json`, SHA-256
`da9c66e35a50ab9ea14cca5050b0ec43b7853278a3fd21fbfb31af27c2ad34c2`.

**Importance ratings:**
- `gpt-4.1-nano-2025-04-14`, temperature 0, seed 0, at most 4 tokens;
- the Park et al. (2023) importance prompt, verbatim apart from the memory text (`retention_signals.ga_importance_prompt`);
- the first integer 1–10 in the reply is the rating; an unparseable reply counts as 5, and their number is reported.

## 5. Primary family (horizon A, label `lex_0.5`, keep 50 %)

| ID | Endpoint | Comparison (a vs b) | Predicted |
|---|---|---|---|
| C1a | AUROC | `read_h4` vs `recency_lru` | a > b (replicates H4) |
| C1b | AUROC | `read_h4` vs `frequency_lfu` | a > b (replicates H4) |
| C2a | harmful deletion | `read_h5` vs `recency_lru` | a < b (replicates H5) |
| C2b | harmful deletion | `read_h5` vs `fifo` | a < b (replicates H5) |
| C3 | AUROC | `write_h4` vs `read_h4` | a > b (write-side thesis) |
| C4 | harmful deletion | `write_h5` vs `read_h5` | a < b (write-side thesis) |
| C5 | AUROC | `utility_write` vs `utility_read` | a > b (utility credited on writes, not retrievals) |
| C6a | AUROC | `write_h4` vs `generative_agents` | a > b |
| C6b | AUROC | `write_h4` vs `memorybank` | a > b |
| C6c | harmful deletion | `write_h5` vs `generative_agents` | a < b |
| C6d | harmful deletion | `write_h5` vs `memorybank` | a < b |

**Tests:**
- **AUROC differences:** paired two-stage clustered bootstrap over dialogues, 5,000 replicates, seed 0, two-sided
  p = 2·min(Pr*(Δ* ≤ 0), Pr*(Δ* ≥ 0)). Paired DeLong p-values are reported alongside but do not decide.
- **Harmful deletion:** each dialogue keeps its top 50 % of facts, with seeded random tie-breaks
  (`run_msc.eviction`); paired two-stage clustered bootstrap over dialogues, 10,000 replicates, seed 0, two-sided.
- **Multiplicity:** Holm–Bonferroni across the 11 comparisons at α = 0.05.
- **Outcome labels:**
  - **confirmed:** Holm p < 0.05, with the difference in the predicted direction;
  - **significant, opposite direction:** Holm p < 0.05, with the difference against the prediction;
  - **not confirmed:** otherwise.

**Interpretation, fixed in advance:**
- The write-side thesis is **supported** only if C3 **and** C4 are confirmed.
- It is **partly supported** if exactly one of them is.
- Otherwise it is **not supported**.
- Every outcome is reported, including a failure of the C1/C2 replications.

## 6. Secondary analyses (not adjusted)

- Horizon B with the same scores and configurations. There is no re-selection, because train has no fifth session.
- Each split separately.
- Labels `lex_0.6`, `lex_0.7` and `embed_0.8`.
- Keep 25 % and 75 %.
- AUROC and harmful deletion for every score in §4, with DeLong and clustered intervals.
- Write-side and read-side DARS with the submitted default weights (0.3, 0.2, 0.3, 0.2), λ = 0.005.

## 7. Integrity rules

- **One run per command.** If a run fails for a software reason, the bug is fixed without inspecting any score
  or label, the fix is logged in the deviations file, and the run is repeated.
- **Nothing is re-selected after contact with these data.**
- **No outcome is computed on these splits** by any command other than those in §3.
