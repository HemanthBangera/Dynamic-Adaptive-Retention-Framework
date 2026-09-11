# Pre-registration — DARS revision experiments

**Status: DRAFT (v0.1, 2026-09-11).** This document is frozen before any
held-out (test-split) evaluation. The SHA-256 of the frozen file is recorded in
the manifest of every test run; any later change goes into the deviations log
(§10) with its reason.

## 1. Purpose

To answer the reviewers of the *Scientific Reports* submission. The experiments
test whether the DARS score changes memory ranking and lifecycle decisions in
ways that improve retrieval or memory quality beyond standard semantic
retrieval. They also test when it does not.

## 2. Data, units and splits

| Source | Questions / items | Memory unit | Evidence label |
|---|---|---|---|
| EventQA-65K, EventQA-full (MemoryAgentBench, Accurate Retrieval) | 5×100 each, disjoint questions | ≤200-token sentence-bounded chunks | none (reader accuracy only) |
| RULER QA1 (197K) / QA2 (421K) | 100 / 100 | ≤200-token chunks within `Document N` blocks | gold SQuAD v2 paragraph / HotpotQA supporting paragraphs (100 % located) |
| LongMemEval-S\* | 5×60 | one unit per turn part, prefixed with session time and role | turn-level `has_answer` (290/300 labelled) |
| FactConsolidation sh/mh 6k/32k (Conflict Resolution) | 100 each | one unit per numbered fact | newest fact stating the answer (100 % labelled; non-newest labels flagged) |
| MSC (`nayohan/multi_session_chat`) | 1,001 complete dialogues | deduplicated persona facts | session-3 restatement (token F1 ≥ τ) |
| ALFWorld (`awawa-agi/alfworld-raw`) | 3,553 train / 140 in-dist / 134 out-of-dist | extracted instance / concept / strategy / goal memories | PDDL-derived need |

Every unit fits the embedder window (all-MiniLM-L6-v2, 254 wordpieces): 0
over-length units on all sources.

**Two structural notes, checked on the data:**
- **EventQA.** EventQA-65K and EventQA-full share no questions (0 of 500 in common), but they are the same five books truncated to different lengths. Each source is therefore analysed separately, with its own context clusters, and never pooled with the other.
- **RULER.** RULER QA1 and QA2 each have a single context, so their bootstrap is over questions.

**Splits.**
- MemoryAgentBench sources use a seeded 30/70 dev/test split at question level,
  within each context (seed 20260911; `benchmarks/dars_eval/splits.py`).
- MSC is split 30/70 by seeded dialogue ID.
- ALFWorld: the train split is used for streaming and tuning; the in- and
  out-of-distribution evaluation splits are the test sets.
- **FactConsolidation exception (disclosed, §9):** the `*_6k` sources are
  development-only. The `*_32k` sources are the held-out sources for H2, using
  their test-split questions.

## 3. Methods

**Baselines.**
- Semantic similarity (the same vault, `rank_mode=similarity`).
- BM25.
- Recency/LRU (last store-or-access time).
- Seeded random.
- No memory: the reader gets an empty memory list (EventQA/RULER reader floor).
- Full context, where it fits the reader's 128k window (EventQA-65K,
  FactConsolidation-32k).
  - What it includes: every memory unit in document order, one `Memory i:` block each. This is a chunked form of MemoryAgentBench's long-context agent.
  - Seeds: seed 0 only, because 65k-token prompts are bound by the tokens-per-minute limit.

**DARS.** `MemoryVault.search_and_rerank`, two-stage. The `fetch_k` nearest
memories are reranked by `rrf` / `wrrf` (β_s / (k + r_sim) + β_d / (k + r_dars))
/ `blend`; further budget slots follow similarity order.
- Weights (w_r, w_f, w_u, w_p) = (0.30, 0.20, 0.30, 0.20) unless tuned (§5).
- The predictive component P has three variants: as submitted (ALFWorld goal
  preset), domain-matched task goal, and none.

**Reader.** `gpt-4o-mini-2024-07-18` at temperature 0, seeds {0, 1, 2}.
- The prompt is MemoryAgentBench's system message, followed by
  `Memory i:\n<text>` blocks in rank order, then the `rag_agent` query template.
- Scoring uses MemoryAgentBench's `post_process`.

**Judge (Layer B).** `SuccessEvaluator` on `gpt-4.1-nano-2025-04-14`.

**Budgets.** Retrieved context is cut at B ∈ {1024, 2048, 5120, 8192, 16384}
tokens (gpt-4o-mini tokenizer, including memory headers). B = 5120 (MAB's
top-10 × 512) is primary for reader runs.

## 4. Protocols

- **E1 static.** A single common timestamp, so R, F and U are identical across
  memories.
- **E2 streams.**
  - FactConsolidation: facts arrive in serial order, one step apart.
  - LongMemEval: sessions at their real times; each question at its own date.
  - EventQA: questions in story order.
  - Feedback sources: none, oracle, oracle with noise ε ∈ {0.1, 0.2, 0.3, 0.5},
    judge, lexical.
- **E9 MSC:** sessions 0–2 are streamed; session 3 is held out.
- **E10 ALFWorld:** the train task stream, with environment-grounded feedback.

## 5. Hyperparameter selection (dev only)

The grid:
- fetch_k ∈ {20, 50, 100}
- β_d ∈ {0.25, 0.5, 1.0} (β_s = 1)
- α ∈ {0.5, 0.8}
- λ ∈ {0.0005, 0.001, 0.005, 0.01, 0.05} h⁻¹
- EventQA question step ∈ {1 h, 24 h}
- lexical τ ∈ {0.3, 0.5, 0.7}
- the weight simplex in steps of 0.1 (286 vectors; retrieval-only)

For each hypothesis the configuration with the best dev value of its primary
endpoint is carried to test. Ties go to the configuration closest to the
submitted defaults. The selected values are appended to §11 before the test run.

Weights are therefore treated as per-domain hyperparameters. The submitted
default weights (0.30, 0.20, 0.30, 0.20) are always reported alongside the tuned
weights. Two secondary analyses test whether tuning is needed: robustness under
Dirichlet perturbation, and cross-domain transfer (weights tuned on one dataset,
evaluated on the others).

Offline tuning (E9, E10, E1) recomputes scores from stored components and
timestamps. Streams whose trajectory depends on the ranking (E2) are re-run for
each λ.

## 6. Primary hypotheses and endpoints

| ID | Hypothesis | Endpoint | Comparison | Test |
|---|---|---|---|---|
| H1 | In the static single-session setting, DARS is not worse than semantic retrieval | evidence group recall at B = 5120 (RULER QA1/QA2, LongMemEval, FC-32k); reader substring EM at 5120 (EventQA, RULER) | DARS-best vs similarity | non-inferiority, margin 0.02 absolute: lower bound of the 95 % paired clustered-bootstrap CI of the difference > −0.02 |
| H2 | With timed updates, recency ranks the newest version of a fact above superseded versions | rank precedence: the newest version is ranked above every superseded version in the method's full ranking of the stored facts (FC sh_32k and mh_32k, test questions). Secondary: precedence at B = 256 (the newest version is shown and outranks any superseded version shown) | DARS-best (stream, no feedback) vs similarity | superiority; paired clustered bootstrap, two-sided α = 0.05 |
| H3 | After an environment-feedback stream, the DARS score (dev-tuned) ranks the correct object location above memories that are near-tied on similarity | MRR of the correct location concept for the location query (ALFWorld in- and out-of-distribution) | DARS-best vs similarity | superiority; paired bootstrap over tasks |
| H4 | DARS retention scores predict which memories will be needed later | AUROC for session-3 restatement (MSC test dialogues) | DARS score vs recency-only and frequency-only | paired DeLong |
| H5 | Under a memory budget, DARS eviction deletes fewer later-needed memories | harmful-deletion rate at M = 50 % (E2 LongMemEval, E9, E10) | DARS vs LRU and FIFO | paired bootstrap over contexts / dialogues / tasks |
| H6 | The LLM judge agrees with ground-truth success labels | Cohen's κ (gpt-4.1-nano) against the reference labels | κ ≥ 0.60 | point estimate and 95 % bootstrap CI reported; pass if the point estimate ≥ 0.60 |

**Multiplicity.** Holm–Bonferroni across all primary comparisons of H1–H5.
Secondary results are labelled exploratory.

## 7. Secondary (exploratory) analyses

- E3 remove-one-component and single-component ablations, weight-perturbation
  robustness, threshold-shift stability and λ sensitivity.
- E4 P variants.
- E6 compression fidelity (SemanticCompressor, LLMLingua-2, extractive),
  shadow indexing, grace period, three-tier vs delete-only.
- E7 Layer A (reformulation, XML vs bullets, memory ordering).
- E8 efficiency curves.
- E11 remaining MemoryAgentBench splits.
- Ranking dynamics (component spread, overlap with similarity, Kendall τ).

## 8. Statistics

- Two-stage clustered bootstrap: contexts, dialogues or tasks, then items;
  10,000 resamples.
- Paired bootstrap, exact McNemar, DeLong, Holm.
- Seeds are fixed and recorded.
- LLM responses are cached; a cache replay must reproduce every number.

## 9. Disclosures (looks at data before freezing)

1. The submitted results (EventQA-65K, first 5 questions × 5 contexts; RULER QA1,
   first 5 questions) were replayed for the harness-equivalence check (E0).
2. During harness development, a FactConsolidation `sh_6k` stream smoke test
   printed results for all questions, including that source's test split.
   `sh_6k` and `mh_6k` were therefore reclassified as development-only (§2).
3. The E1 smoke test used FactConsolidation `sh_6k` dev questions only.
4. No other test-split question has been evaluated by any revision configuration.

## 10. Deviations log

Draft-stage changes, made before freezing and before any test-split evaluation:

- **H3 wording.** Changed from "learned utility" to "the DARS score (dev-tuned)".
  The ALFWorld smoke stream on dev showed that the utility term's Laplace prior of
  0.5 ranks untested memories above tested ones when the success base rate is
  low (~0.25 with ~4 alternative locations). The hypothesis now tests the tuned
  score, not a single component.
- **ALFWorld protocol.** Each stream task also issues a location query ("Where can
  I find a <Object>?"). H3 candidates are ranked against the same query.
  Motivation: with only the goal query, location memories were rarely retrieved,
  so almost no location feedback was produced.

- **H2 endpoint.**
  - **Change.** The endpoint was "precedence at B = 256". It is now rank precedence over each method's full ranking of the stored facts. Precedence at B = 256 remains a secondary endpoint.
  - **Why.** On dev (FactConsolidation mh_32k, 18 labelled questions), evidence recall at 256 tokens was 0.03–0.13 for every method, because a multi-hop answer needs two facts. The budget-cut metric therefore mostly scored whether anything was retrieved, not the ordering of fact versions that H2 is about.
  - **Effect.** Rank precedence measures that ordering directly, for every labelled question.
  - **Status.** Decided on dev data only, before freezing. The ranking prefix, and therefore the memories shown to the reader, is unchanged.

- **Eviction tie-breaking.**
  - **Problem.** During dev tuning, score ties in eviction were broken by storage order (ALFWorld, MSC) or by context position (E2). Several weight vectors produce many ties; under ALFWorld weights (0, 0.5, 0.5, 0), for example, every never-accessed memory has F = 0 and U = 0.5. For such vectors this leaked creation order into the policy.
  - **Effect.** The ALFWorld H5 selection scored 0.017 harmful deletion with storage-order ties, but 0.341 with random ties, which equals LFU.
  - **Fix.** Every eviction ranking now breaks ties by a seeded random order: DARS and the baselines, in tune, analyze, evaluate and E2. The H5 selections were re-derived on dev.
  - **Status.** Found by the dev consistency check of the new `evaluate` command, before freezing.

## 11. Selected configurations (appended before the test run)

**Selection rule (§5).** Configurations were selected on dev data only: the best dev value of each hypothesis's primary endpoint, with ties going to the configuration closest to the submitted one. The submitted configuration is weights (0.3, 0.2, 0.3, 0.2), λ = 0.005 h⁻¹, rrf with fetch_k 15. The default-weight result is always reported alongside the selected one.

| ID | Selected configuration | Dev primary endpoint (selected) | Dev comparator(s) | Default weights (dev) |
|---|---|---|---|---|
| H1 | *pending: E1 reader dev (EventQA-65K)* | | | |
| H2 | dars_rrf_k50 (fetch_k 50, RRF k = 60), default weights, λ = 0.001 h⁻¹, stream without feedback, B = 256 | rank precedence: sh_32k 0.952 [0.86, 1.00] (n = 21); mh_32k 0.611 [0.39, 0.83] (n = 18) | similarity 0.333 / 0.556 | λ = 0.005: 0.762 / 0.611 |
| H3 | DARS score only (no fusion), weights (0, 1, 0, 0), λ = 0.0005 h⁻¹ | location MRR 0.502 (n = 351 tasks) | similarity 0.449 | rrf: 0.449 |
| H4 | weights (0.2, 0, 0.8, 0), λ = 0.05 h⁻¹; label lex_0.5 | AUROC 0.894 (5,812 facts, 300 dialogues) | recency-only 0.657; frequency-only 0.245 | 0.711 |
| H5 (E9 MSC) | weights (0.4, 0, 0.6, 0), λ = 0.005 h⁻¹ | harmful deletion at keep 50 %: 0.102 | LRU 0.397; FIFO 0.128 | 0.329 |
| H5 (E10 ALFWorld) | weights (0, 0.5, 0.5, 0), λ = 0.0005 h⁻¹ | harmful deletion at keep 50 %: 0.017 | LRU 0.999; FIFO 0.991 | 0.421 |
| H5 (E2 LongMemEval) | DARS eviction, weights (0.4, 0, 0.6, 0), λ = 0.005 h⁻¹; retrieval dars_blend_a0.5 with oracle feedback, B = 2048, keep 50 %. Candidates restricted to {default, E9-H5, E10-H5}: the stream depends on each configuration, so a full weight grid is infeasible | harmful deletion 0.045 [0.000, 0.112] (n = 86) | LRU 0.045; FIFO 0.052 (LFU 0.072 and random 0.232 also reported) | 0.050 |
| H6 | no selection: the Layer B `SuccessEvaluator` prompt on gpt-4.1-nano | | | |

**Reported alongside, though not pre-registered comparators.** They are listed so that the strongest simple baselines are visible:
- ALFWorld count prior: MRR 0.629.
- MSC:
  - FIFO AUROC 0.865;
  - utility-only AUROC 0.880;
  - mention count AUROC 0.751.
- ALFWorld LFU: harmful deletion 0.337.

**Robustness.** Two configurations are fragile under Dirichlet weight perturbation (E3):
- the H3 selection: MRR mean 0.407;
- the MSC default weights: AUROC p05 0.547.

Both results will be reported.
