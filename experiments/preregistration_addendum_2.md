# Pre-registration addendum 2 — confirmatory test of display order under conflicting memories

**Status: FROZEN (v1.0, 2026-09-13 UTC).** Frozen before any run on the data it describes. Its SHA-256 is recorded in
`experiments/preregistration_addendum_2.md.sha256` and in `experiments/deviations_post_freeze.md`. This file is never
edited; later changes go into the deviations log.

## 1. Why

In the exploratory FactConsolidation runs, the MemoryAgentBench reader answered best when the memory the retriever
ranked first was shown *last*, next to the question. These runs used sh_32k, B = 256, the MemoryAgentBench prompt and
serial prefixes (`benchmark_runs/revision/test/E13_audit/fc_order`, `fc_serial`).

| sh_32k, test questions | best-first (registered display) | best-last |
|---|---|---|
| BM25 | 0.886 | 0.671 |
| DARS `dars_rrf_k50` | 0.600 | 0.814 |
| DARS over BM25 (`bm25_dars_wrrf_k50_b0.5`) | 0.514 | 0.929 |

The explanation proposed: when two versions of a fact are shown, the reader favours the later one in the prompt. A
lifecycle ranking helps the answer only if its top-ranked memory — the current version — is placed last.

**This was found after looking at sh_32k.** It is tested here on FactConsolidation sources the study never used.

## 2. Data

- **Sources:** MemoryAgentBench `Conflict_Resolution`, `factconsolidation_sh_64k` and `factconsolidation_sh_262k`. No
  earlier run loaded them.
- **Overlap with sh_32k:** 1 of 100 question texts in each source; 49 % (64k) and 13 % (262k) of the fact statements.
  The overlapping question is kept and disclosed.
- **Questions:** test-split questions of each source (`splits.question_splits`, seed 20260911).

## 3. Protocol

- **Stream:** the E2 FactConsolidation stream, unchanged: facts in serial order one hour apart, feedback none,
  B = 256, λ = 0.001 h⁻¹, serial prefixes kept, MemoryAgentBench prompt.
- **Reader:** `gpt-4o-mini-2024-07-18`, seed 0. Answers delivered through the Batch API and replayed offline.

```
python -m benchmarks.dars_eval.run_stream --source <src> --out benchmark_runs/revision/addendum2/<src>/order_<order> \
  --methods similarity,bm25,dars_rrf_k50,bm25_dars_wrrf_k50_b0.5 --feedback none --budget 256 --decay-lambda 0.001 \
  --reader --display-order <order> --report-split test --n-boot 10000
```

`<src>` ∈ {factconsolidation_sh_64k, factconsolidation_sh_262k}; `<order>` ∈ {best_first, best_last}.

## 4. Primary family (reader substring exact match; Holm over 4 comparisons, α = 0.05)

| ID | Source | Comparison (a − b) | Predicted |
|---|---|---|---|
| D1a | sh_64k | `bm25_dars_wrrf_k50_b0.5` best-last − `bm25` best-first | > 0 |
| D1b | sh_262k | same | > 0 |
| D2a | sh_64k | `dars_rrf_k50` best-last − `dars_rrf_k50` best-first | > 0 |
| D2b | sh_262k | same | > 0 |

- **D1** compares DARS over BM25, shown best-last, against plain BM25 as MemoryAgentBench displays it.
- **D2** isolates the effect of display order on DARS.

**Tests:** paired bootstrap over the test questions (single context per source), 10,000 replicates, seed 0, two-sided.

**Outcome labels:**
- **confirmed:** Holm p < 0.05, with the difference in the predicted direction;
- **significant, opposite direction:** Holm p < 0.05, with the difference against the prediction;
- **not confirmed:** otherwise.

**Interpretation:**
- The display-order account is **supported** only if both D2 comparisons are confirmed.
- DARS over BM25 **improves on the benchmark's own display** only if both D1 comparisons are confirmed.

## 5. Secondary

- Every method in both orders, with precedence at B = 256 and recall.
- Similarity and BM25, best-last against best-first.
