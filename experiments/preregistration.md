# Pre-registration — DARS revision experiments

**Status: FROZEN (v1.0, 2026-09-12).** This document is frozen before any
held-out (test-split) evaluation. The SHA-256 of the frozen file is recorded in
`experiments/preregistration.sha256` and in the manifest of every test run; any
later change goes into the deviations log (§10) with its reason. The test phase
is run by `experiments/run_test_phase.sh`, which refuses to start unless this
file is frozen and unchanged.

## 1. Purpose

To answer the reviewers of the *Scientific Reports* submission. The experiments
test whether the DARS score changes memory ranking and lifecycle decisions in
ways that improve retrieval or memory quality beyond standard semantic
retrieval. They also test when it does not.

## 2. Data, units and splits

| Source | Questions / items | Memory unit | Evidence label |
|---|---|---|---|
| EventQA-65K, EventQA-full (MemoryAgentBench, Accurate Retrieval) | 5×100 each, disjoint questions | ≤200-token sentence-bounded chunks | none (reader accuracy only) |
| RULER QA1 (197K) / QA2 (421K) | one context of 100 questions each | ≤200-token chunks within `Document N` blocks | gold SQuAD v2 paragraph / HotpotQA supporting paragraphs (100 % located) |
| LongMemEval-S\* | 5×60 | one unit per turn part, prefixed with session time and role | turn-level `has_answer` (290/300 labelled) |
| FactConsolidation sh/mh 6k/32k (Conflict Resolution) | one context of 100 questions each | one unit per numbered fact | newest fact stating the answer (100 % labelled; non-newest labels flagged) |
| MSC (`nayohan/multi_session_chat`) | 1,001 complete dialogues | deduplicated persona facts | session-3 restatement (token F1 ≥ τ) |
| ALFWorld (`awawa-agi/alfworld-raw`) | 3,553 train / 140 in-dist / 134 out-of-dist | extracted instance / concept / strategy / goal memories | memories the task needs, from its walkthrough (true location concept, tool concept, task-type strategy) |
| DetectiveQA, ICL-Banking77 (E11) | 10 contexts / 71 questions; one context / 100 questions | book chunks; one labelled example per unit | none (reader accuracy only) |

Every unit fits the embedder window (all-MiniLM-L6-v2, 254 wordpieces): 0
over-length units on all sources.

**Two structural notes, checked on the data:**
- **EventQA.** EventQA-65K and EventQA-full share no questions (0 of 500 in common), but they are the same five books truncated to different lengths. Each source is therefore analysed separately, with its own context clusters, and never pooled with the other.
- **Single-context sources.** RULER QA1, RULER QA2 and each FactConsolidation source have a single context, so their bootstrap is over questions.

**Splits.**
- MemoryAgentBench sources use a seeded 30/70 dev/test split at question level,
  within each context (seed 20260911; `benchmarks/dars_eval/splits.py`).
- MSC is split 30/70 by seeded dialogue ID.
- ALFWorld: the train split is used for streaming and tuning (a seeded 10 % of
  train tasks is held out as the dev evaluation set); the in- and
  out-of-distribution evaluation splits are the test sets.
- **FactConsolidation exception (disclosed, §9):** the `*_6k` sources are
  development-only. The `*_32k` sources are the held-out sources for H2, using
  their test-split questions.

## 3. Methods

**Baselines.**
- Semantic similarity (the same vault, `rank_mode=similarity`).
- BM25.
- Recency/LRU (last store-or-access time); for eviction also LFU, FIFO and seeded random.
- No memory: the reader gets an empty memory list (EventQA/RULER reader floor).
- Full context, where it fits the reader's 128k window (EventQA-65K,
  FactConsolidation-32k).
  - What it includes: every memory unit in document order, one `Memory i:` block each. This is a chunked form of MemoryAgentBench's long-context agent.
  - Seeds: seed 0 only, because 65k-token prompts are bound by the tokens-per-minute limit.

**DARS.** `MemoryVault.search_and_rerank`, two-stage. The `fetch_k` nearest
memories are reranked by `rrf` / `wrrf` (β_s / (k + r_sim) + β_d / (k + r_dars))
/ `blend`; further budget slots follow similarity order.
- Weights (w_r, w_f, w_u, w_p) = (0.30, 0.20, 0.30, 0.20) unless tuned (§5).
- The predictive component P has five variants (E4): none, as submitted (ALFWorld
  goal preset), domain-matched task goal, dynamic (mean of the previous ten
  queries) and oracle (mean of the next ten queries). Unless stated otherwise,
  runs use the as-submitted variant on MemoryAgentBench and the domain-matched
  presets on MSC and ALFWorld.

**Streams in E9 and E10.** During the MSC and ALFWorld streams, retrieval is by
similarity for every policy, so all retention policies are compared on an
identical memory state. The DARS score is evaluated on the end-of-stream state
as a retention, eviction and re-ranking signal. The E2 streams, in contrast, are
closed-loop: each method's own ranking decides which memories are shown and
receive feedback.

**Reader.** `gpt-4o-mini-2024-07-18` at temperature 0, seeds {0, 1, 2}.
- The prompt is MemoryAgentBench's system message, followed by
  `Memory i:\n<text>` blocks in rank order, then the `rag_agent` query template.
- Scoring uses MemoryAgentBench's `post_process`.

**Judge (Layer B).** `SuccessEvaluator` on `gpt-4.1-nano-2025-04-14`; a reply is
a verdict only if it begins with the whole word YES or NO, otherwise NEUTRAL.

**Budgets.** Retrieved context is cut at B ∈ {1024, 2048, 5120, 8192, 16384}
tokens (gpt-4o-mini tokenizer, including memory headers). B = 5120 (MAB's
top-10 × 512) is primary for reader runs.

## 4. Protocols

- **E1 static.** A single common timestamp, so R, F and U are identical across
  memories.
- **E2 streams.**
  - FactConsolidation: facts arrive in serial order, one hour apart; questions
    follow the last fact, one hour apart.
  - LongMemEval: sessions at their real times; each question at its own date.
  - Feedback sources:
    - none and oracle (all streams);
    - oracle with noise ε ∈ {0.1, 0.2, 0.3, 0.5} (LongMemEval);
    - the Layer B judge and the lexical rater, with the reader answering every question (LongMemEval).
  - Memory budget M = 50 % with DARS, LRU, LFU, FIFO and random eviction (LongMemEval); M = 25 % and 75 % as a secondary sweep (§7).
- **E9 MSC:** sessions 0–2 are streamed (24 h apart); session 3 is held out.
- **E10 ALFWorld:** the train task stream (one hour per task), with environment-grounded feedback.

## 5. Hyperparameter selection (dev only)

The grid:
- fetch_k ∈ {20, 50, 100}
- β_d ∈ {0.25, 0.5, 1.0} (β_s = 1)
- α ∈ {0.5, 0.8}
- λ ∈ {0.0005, 0.001, 0.005, 0.01, 0.05} h⁻¹
- the weight simplex in steps of 0.1 (286 vectors; offline analyses)
- fusion modes for ALFWorld re-ranking: rrf, wrrf (β_d ∈ {0.25, 0.5, 1.0}),
  blend (α ∈ {0.5, 0.8}) and score-only

The lexical rater thresholds τ ∈ {0.3, 0.5, 0.7} (E5) are reported, not tuned.

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
each λ; for E2 eviction the candidate weights are restricted (§11).

## 6. Primary hypotheses and endpoints

| ID | Hypothesis | Endpoint | Comparison | Test |
|---|---|---|---|---|
| H1 | In the static single-session setting, DARS is not worse than semantic retrieval | evidence group recall at B = 5120 (RULER QA1/QA2, LongMemEval, FC-32k sh and mh); reader substring EM at 5120, averaged per question over seeds {0, 1, 2} (EventQA-65K, EventQA-full, RULER QA1/QA2) | DARS-best vs similarity | non-inferiority, margin 0.02 absolute (see Multiplicity) |
| H2 | With timed updates, recency ranks the newest version of a fact above superseded versions | rank precedence: the newest version is ranked above every superseded version in the method's full ranking of the stored facts (FC sh_32k and mh_32k, test questions). Secondary: precedence at B = 256 (the newest version is shown and outranks any superseded version shown) | DARS-best (stream, no feedback) vs similarity | superiority; paired clustered bootstrap, two-sided |
| H3 | After an environment-feedback stream, the DARS score (dev-tuned) ranks the correct object location above memories that are near-tied on similarity | MRR of the correct location concept for the location query (ALFWorld in- and out-of-distribution) | DARS-best vs similarity | superiority; paired bootstrap over tasks, two-sided |
| H4 | DARS retention scores predict which memories will be needed later | AUROC for session-3 restatement (MSC test dialogues) | DARS score vs recency-only and frequency-only | paired DeLong, two-sided |
| H5 | Under a memory budget, DARS eviction deletes fewer later-needed memories | harmful-deletion rate at M = 50 % (E2 LongMemEval, E9, E10 in- and out-of-distribution) | DARS vs LRU and FIFO | paired bootstrap over contexts / dialogues / tasks, two-sided |
| H6 | The LLM judge agrees with ground-truth success labels | Cohen's κ (gpt-4.1-nano) against the reference labels | κ ≥ 0.60 | point estimate and 95 % bootstrap CI reported; pass if the point estimate ≥ 0.60 |

**Multiplicity.** The primary family has 23 comparisons, adjusted together by the Holm–Bonferroni procedure at α = 0.05:
- **H1 (9):** evidence recall on RULER QA1, QA2, LongMemEval, FC-sh_32k and FC-mh_32k; reader EM on EventQA-65K, EventQA-full, RULER QA1 and QA2.
  - Each uses the one-sided bootstrap p-value p = Pr*(Δ* ≤ −0.02) for the difference Δ = DARS − similarity, doubled so that it enters the family on the two-sided scale.
  - Non-inferiority holds when the Holm-adjusted p is below 0.05.
- **H2 (2):** sh_32k, mh_32k.
- **H3 (2):** in- and out-of-distribution.
- **H4 (2):** vs recency-only and vs frequency-only.
- **H5 (8):** vs LRU and vs FIFO in each of E2-LongMemEval, E9, E10-in and E10-out.
- H2–H5 use two-sided paired p-values.

H6 is a threshold criterion outside the family. Secondary results are labelled exploratory and are not adjusted.

## 7. Secondary (exploratory) analyses

- E3 remove-one-component and single-component ablations, weight-perturbation
  robustness, threshold-shift stability, closed-form tier lifecycle and λ sensitivity.
- E4 P variants.
- E5 judge reliability beyond κ: accuracy, second judge (gpt-4o-mini), test–retest,
  paraphrased prompt, NEUTRAL rate, deterministic lexical rater.
  - Also, for each rater, κ against each half of the reference label (the answer is
    correct; the shown memories contain gold evidence) and a per-source breakdown.
    These use the verdicts already collected, and show which half of the definition
    a rater fails to track.
- E2 feedback-noise curve and feedback-source comparison (none / oracle / judge /
  lexical) on LongMemEval, including evidence recall and reader accuracy.
- **E2 eviction budget sweep (LongMemEval).**
  - Keep 25% and 75% of the store, with the H5-E2 configuration and the same policies: DARS with default weights, LRU, FIFO and LFU, plus random eviction averaged over five seeds.
  - H5 itself stays at keep 50%.
  - E9 and E10 already report keep 25/50/75% in their analyses, so all three eviction settings share the same budgets.
- E6 compression fidelity (SemanticCompressor, LLMLingua-2, extractive; fixed and
  matched ratios) and shadow indexing (kept vector vs re-embedded summary).
- **E7 Layer A.** A 2×2×2 factorial of reformulation (raw / reformulated), prompt format (XML / MemoryAgentBench) and memory ordering (best-first / best-last).
  - In the factorial the XML prompt carries the same memories as the MemoryAgentBench prompt (PromptConstructor's cap lifted).
  - The gateway as implemented is two further conditions: the 20,000-character cap, best-first, with raw and with reformulated queries.
  - Every condition reports how many memories reached the prompt; evidence recall counts only those.
- E8 efficiency at equal budgets with index-time compression (none, extractive,
  LLMLingua-2), reader at B = 1024, ranking latency.
- **E11, the remaining MemoryAgentBench competencies:**
  - **DetectiveQA (Long-Range Understanding):** 10 contexts, 71 questions; ≤200-token book chunks. The retrieval query is the question and its options, after the fixed worked example.
  - **ICL-Banking77 (Test-Time Learning):** one labelled example per memory unit (5,897 units); the retrieval query is the utterance.
  - **Protocol:** methods similarity, BM25, dars_rrf_k15 and no memory; B = 5120; reader seeds {0, 1, 2}; MemoryAgentBench scoring; test questions only; no tuning.
  - **Out of scope:** InfBench-Sum (whole-book summaries) and RecSys-Redial (recommendation from a 521k-token dialogue log), because neither is question answering from a fixed-budget memory.
- Ranking dynamics (component spread, overlap with similarity, Kendall τ,
  per-component influence).

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
   All development runs of E1–E11 used development questions, dialogues or tasks
   only; the ALFWorld evaluation splits and the MSC test dialogues have not been run.

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

- **Random-eviction reproducibility.**
  - **Problem.** In E2, the random eviction baseline drew one key per memory in the vault's scroll order. That order follows random point ids, so the baseline changed from run to run.
  - **Fix.** Keys are now indexed by memory unit, and the baseline's dev value was re-measured.
  - **Status.** Found by re-running dev before freezing.

- **Scope of E2, E6 and E11 (v0.2).**
  - The EventQA story-order stream listed in v0.1 was dropped. It is replaced by the LongMemEval feedback-source stream (reader, judge, lexical and oracle feedback), whose gold evidence allows an oracle; the EventQA question-step grid was dropped with it.
  - The E6 grace-period and three-tier-vs-delete-only analyses listed in v0.1 were not implemented. The grace period and tiers are examined analytically in E3 instead.
  - E11 was scoped to DetectiveQA and ICL-Banking77.
  - A LongMemEval eviction sweep at keep 25% and 75% was added as a secondary analysis (§7), so that E2, E9 and E10 report the same budgets.
  - The multiplicity family and the H1 reader endpoint (per-question mean over seeds) were made explicit.

- **Duplicate concurrent LLM requests.**
  - **Problem.** Identical requests issued concurrently each made their own API call. At temperature 0 with a fixed seed the API is only near-deterministic, so a run could hold two different answers to one request while the cache stored one of them. A cache replay then differed slightly from the original run (E5 dev: inter-judge κ 0.2258 → 0.2283). In E5, 6.3% of items shared a prompt with another item.
  - **Fix.** The transport now coalesces in-flight duplicates: the first caller makes the call, later callers await its result, and the ledger reports how many were `coalesced`. Runs and their replays therefore agree.
  - **Status.** Found before freezing, while replaying the E5 development run. The cached replay is the canonical artifact for any run made before this fix.

- **Transient network errors.**
  - **Problem.** A brief connection outage aborted two development runs (E7 and the LongMemEval feedback-source stream) after the OpenAI SDK's own retries were exhausted.
  - **Fix.** The transport now retries dropped connections and 5xx responses with exponential backoff (`DARS_OPENAI_TRANSIENT_RETRIES`, default 8) and counts them in the ledger. Authentication and request errors still fail immediately, and no failed call is ever scored.
  - **Effect on results:** none. The retried request is identical and its answer is cached as usual. Stages of the test phase are resumable, so this only avoids losing work.

- **E7 prompt-format confound.**
  - **Problem.** The E7 dev log showed PromptConstructor's 20,000-character cap dropping memories from the XML prompt at B = 5120. The XML-vs-MemoryAgentBench contrast therefore compared different amounts of context, not formats alone. Under best-last ordering the cap dropped the highest-ranked memories, because it cuts the display order.
  - **Fix.** The factorial's XML arm now lifts the cap, so it is a pure format contrast. The gateway as implemented is kept as two separate conditions (capped, best-first). Every row records how many memories reached the prompt, and evidence recall counts only those. PromptConstructor's default behaviour is unchanged.
  - **Status.** Found in the dev log before freezing. The dev run was stopped and repeated with the corrected design.

## 11. Selected configurations (appended before the test run)

**Selection rule (§5).** Configurations were selected on dev data only: the best dev value of each hypothesis's primary endpoint, with ties going to the configuration closest to the submitted one. The submitted configuration is weights (0.3, 0.2, 0.3, 0.2), λ = 0.005 h⁻¹, rrf with fetch_k 15. The default-weight result is always reported alongside the selected one.

| ID | Selected configuration | Dev primary endpoint (selected) | Dev comparator(s) | Default weights (dev) |
|---|---|---|---|---|
| H1 | dars_rrf_k15, the submitted fusion (RRF k = 60 over the 15 nearest memories, default weights, submitted goal vector). All DARS variants tied with similarity on dev evidence recall at B = 5120. H1 has two endpoints, so the tie was broken by dev reader EM (the secondary endpoint), then by closeness to the submitted configuration | recall@5120 equal to similarity on every labelled source (QA1 1.000, QA2 0.900, LME 0.777, FC-sh 1.000, FC-mh 0.500); reader EM@5120: EventQA-65K 0.887, QA1 0.867, QA2 0.633, LME 0.411 | similarity EM 0.880 / 0.867 / 0.633 / 0.356 (BM25 0.913 / 0.900 / 0.633 / 0.456 also reported) | is the default |
| H2 | dars_rrf_k50 (fetch_k 50, RRF k = 60), default weights, λ = 0.001 h⁻¹, stream without feedback, B = 256 | rank precedence: sh_32k 0.952 [0.86, 1.00] (n = 21); mh_32k 0.611 [0.39, 0.83] (n = 18) | similarity 0.333 / 0.556 | λ = 0.005: 0.762 / 0.611 |
| H3 | DARS score only (no fusion), weights (0, 1, 0, 0), λ = 0.0005 h⁻¹ | location MRR 0.502 (n = 351 tasks) | similarity 0.449 | rrf: 0.449 |
| H4 | weights (0.2, 0, 0.8, 0), λ = 0.05 h⁻¹; label lex_0.5 | AUROC 0.894 (5,812 facts, 300 dialogues) | recency-only 0.657; frequency-only 0.245 | 0.711 |
| H5 (E9 MSC) | weights (0.3, 0, 0.7, 0), λ = 0.0005 h⁻¹ (re-derived after the tie-break fix, §10) | harmful deletion at keep 50 %: 0.102 | LRU 0.388; FIFO 0.131 | 0.328 |
| H5 (E10 ALFWorld) | weights (0, 0, 0, 1): predictive relevance only (the domain-matched ALFWorld goal). λ has no effect because w_r = 0 (0.0005 recorded). Re-derived after the tie-break fix, §10 | harmful deletion at keep 50 %: 0.090 | LRU 0.999; FIFO 0.991 | 0.421 |
| H5 (E2 LongMemEval) | DARS eviction with the default weights (0.3, 0.2, 0.3, 0.2), λ = 0.005 h⁻¹; retrieval dars_blend_a0.5 with oracle feedback, B = 2048, keep 50 %. Candidates were restricted to {default, E9-H5 (0.3, 0, 0.7, 0) at λ = 0.0005, E10-H5 P-only}, because the stream depends on each configuration and a full weight grid is infeasible. Re-derived after the tie-break fix, §10 | harmful deletion 0.050 [0.006, 0.116] (n = 86) | LRU 0.045; FIFO 0.052 (also reported: LFU 0.233, random 0.091 from one seed (the test phase averages five seeds), E9-H5 weights 0.070, P-only 0.205) | is the default |
| H6 | no selection: the Layer B `SuccessEvaluator` prompt on gpt-4.1-nano | | | |

**Reported alongside, though not pre-registered comparators.** They are listed so that the strongest simple baselines are visible:
- ALFWorld count prior: MRR 0.629.
- MSC:
  - FIFO AUROC 0.865;
  - utility-only AUROC 0.880;
  - mention count AUROC 0.751.
- ALFWorld LFU: harmful deletion 0.341 (seeded random tie-break).

**Robustness.** Two configurations are fragile under Dirichlet weight perturbation (E3):
- the H3 selection: MRR mean 0.407;
- the MSC default weights: AUROC p05 0.547.

Both results will be reported.
