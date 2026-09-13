# Deviations after the pre-registration was frozen

The pre-registration (`experiments/preregistration.md`, v1.0, SHA-256
`a3dec99a953600c37efae8e572528c4f5059b95c87869460a0a4d1df3a965270`) is kept **byte-identical**
after freezing, so its hash continues to verify. Anything that changed afterwards is recorded
here instead, in date order, with what it affects and why. Nothing in this file changes a
hypothesis, an endpoint, a comparator or a selected configuration.

---

## 2026-09-12 — ALFWorld out-of-distribution walkthroughs were not parsed

**What was wrong.** The first test-phase run of E10 produced zero candidate location memories
for all 134 out-of-distribution tasks, and the `evaluate` step then failed on an empty
bootstrap sample. The cause was a difference between the splits of `awawa-agi/alfworld-raw`:

- train and in-distribution walkthroughs name objects readably — `take cd 1 from desk 1`;
- out-of-distribution walkthroughs use simulator identifiers —
  `take cd_bar__minus_00_dot_40_bar__plus_00_dot_86_bar__minus_00_dot_66 from desk_bar__minus_00_dot_57_…`.

`_parse_walkthrough` matched only the readable form, so for that split the target object,
source receptacle, tool and action were all `None`. Every downstream quantity was therefore
empty or wrong: no location query, no candidates, and an under-counted set of needed memories.

**Fix.** `data/groupB/extractor.py` now normalises simulator identifiers to the readable form
before parsing (`_canonical_name`, `_normalize_step`). The type is the leading segment, except
where a trailing alphabetic segment carries it: `sink_bar__…_bar_sinkbasin` is a `sinkbasin`,
which is also the name the train split uses. Of the 46 identifier types in that split, this is
the only one whose leading segment disagrees with the readable vocabulary, and getting it wrong
would have made every sink comparison silently false.

**What it affects.** E10 on the out-of-distribution split only: H3 (location MRR) and H5
(harmful deletion), i.e. two of the 23 pre-registered comparisons. The development and
in-distribution splits use readable names and are unchanged, as are E1–E9 and E11.

**Status.** Found by the test-phase run itself, before any out-of-distribution result was
recorded or reported. The E10 stage was re-run from the beginning after the fix. No
out-of-distribution number produced before the fix is used anywhere.

**Also changed.** `run_alfworld.evaluate` now reports a split that has no task with more than
one candidate location memory as `not_evaluable`, instead of raising. An unscorable split is a
result to be reported, not a crash.

**Tests.** `tests/test_alfworld_walkthrough_unit.py`.

---

## 2026-09-12 — the remaining test stages moved from the laptop to EC2

**Why.** The laptop could not hold a multi-day run: a transient network error killed two stages, and
Windows then killed the driver under memory pressure (15.6 GB total, ~3 GB free, against EventQA
contexts of 400–736k tokens). The remaining nine stages were moved to an `r7i.2xlarge` instance
(8 vCPU, 61 GB, us-east-1a) running the same pinned environment. Nothing about the protocol, the
hypotheses, the endpoints or the selected configurations changed; the frozen pre-registration is
unmodified and its hash still verifies on the new host.

**What ran where.** Stages `e2_h2`, `e2_h5`, `e2_budget_sweep`, `e2_dynamics`, `e2_noise`, `e9`, `e10`,
`e3`, `e4` and the E1 readers for RULER QA1/QA2, LongMemEval and FactConsolidation sh/mh ran on the
laptop (Windows 11, i5-12500H). The rest ran on EC2 (Ubuntu 24.04). Every run manifest records
`platform`, `processor`, `cpu_count`, the package versions, the dataset revisions and the
pre-registration hash, so which host produced each number is recoverable from the artifacts.

**Environment on the Linux host.** `experiments/requirements-vm.txt` pins every package this project
imports to the exact version in the frozen laptop lock file (31 packages: torch, sentence-transformers
5.4.1, transformers 5.6.2, numpy 2.3.4, qdrant-client 1.17.1, openai 3.13.0, datasets 4.8.5, tiktoken
0.12.0, scipy 1.17.0, matplotlib 3.10.9, llmlingua 0.2.2, …), and `requirements-vm-installed.txt`
records what was actually installed. Two deliberate differences:

- `torch==2.11.0` is installed from the PyTorch CPU index, giving the same CPU build as the laptop's
  `2.11.0+cpu` instead of the CUDA build PyPI supplies by default.
- The laptop's *whole* global `pip freeze` is not installed. It contains packages this project never
  imports whose pins contradict one another (`awscli==1.45.36` requires `botocore==1.43.36` while the
  file pins `botocore==1.43.42`; `streamlit==1.51.0` requires `pyarrow<22` while the file pins
  `pyarrow==23.0.1`) and others with no CPython 3.14 wheel that would have to be compiled against
  system graphics headers. None of them is importable from this codebase, so installing them would add
  risk without adding fidelity. Python itself is the same version, 3.14.3.

**Why this cannot move the numbers.** The embedding cache and the full LLM response cache were copied to
the instance, so embeddings and model responses are replayed from disk rather than recomputed; the
remaining compute is retrieval arithmetic and statistics on pinned NumPy. The test suite is required to
pass with the same count on both hosts (334) before any stage is run.

---

## 2026-09-12 — reader and judge calls delivered through the OpenAI Batch API

**Why.** The account is Tier 1, where `gpt-4o-mini` allows 10,000 synchronous requests per day.
That allowance was exhausted on 12 September with roughly 19,000 reader and judge calls still to
make, and the headers showed the next reset more than 32 hours away
(`x-ratelimit-remaining-requests: 0`, `x-ratelimit-reset-requests: 32h44m`). Waiting would have
spread the remaining stages over about three days. The Batch API draws on a separate allowance —
it accepted work while the synchronous cap was exhausted — so the remaining calls go through it.

**What is identical.** The request bodies are byte-for-byte what the synchronous path would have
sent: the same model snapshot (`gpt-4o-mini-2024-07-18`), the same system and user messages,
`temperature` 0, the same explicit seed and the same token limit. Each batched response is stored
in the same content-addressed cache under the same key (a SHA-256 of the request), so a stage
replayed afterwards consumes exactly these answers and any later replay reproduces them.

**What differs.** Only the delivery channel and its latency. Responses carry a `service_tier`
field from the batch queue, and each cached record is tagged `"via": "batch"`, so which channel
answered any given call is recoverable from the artifacts. Batch pricing is half the synchronous
rate, so the reported cost is correspondingly lower.

**Mechanics.** `DARS_LLM_COLLECT` makes the transport write a request out instead of calling the
API, returning a placeholder; a collect pass therefore writes to a scratch directory and its own
metrics are discarded. `scripts/batch_run.py` submits the collected requests, respecting the
2,000,000 enqueued-token limit, and writes the answers into the cache. The stage is then run
normally and every call is a cache hit. `experiments/aws/batch_stage.sh` does the three steps for
one stage, using the same commands as `experiments/run_test_phase.sh`.

**Not used for E2's feedback-source stream**, where each reader answer determines the next
retrieval and the requests therefore cannot be known in advance. That stage runs synchronously.

**Tests.** `tests/test_batch_run_unit.py` (8), including that a batched response replays as a
cache hit with the correct text, usage and cost, and that a batch rejected for the enqueued-token
limit is requeued rather than lost.

---

## 2026-09-12 — secondary analysis: reader-level check of the H2 precedence result

**What.** H2's pre-registered endpoint is *rank precedence*: does the current version of a fact outrank
its superseded versions. After seeing the test result (DARS 0.976 vs similarity 0.357 vs BM25 0.238) we
added a secondary, exploratory check: run the same FactConsolidation streams with the reader enabled
(feedback off, B = 256, λ = 0.001) and measure answer accuracy. Artifacts:
`benchmark_runs/revision/test/E2/staleness_factconsolidation_{sh,mh}_32k`.

**Why.** A ranking metric is only interesting if it changes what the agent does. This converts H2 from
a statement about ordering into a statement about behaviour.

**Result, which limits the claim.** Precedence did not translate into accuracy on sh_32k: BM25 0.886,
similarity 0.757, DARS 0.600, despite DARS's far higher precedence. MemoryAgentBench prompts carry the
facts' serial numbers, so a reader shown *both* versions resolves the conflict itself; DARS buys
precedence by showing fewer memories (recall 0.857 against 1.000) and sometimes shows neither. Multi-hop
is uninformative at this budget (accuracy 0.04–0.09).

**How it is reported.** H2 stands as pre-registered and is reported as a ranking result. This secondary
check is reported alongside it, and the manuscript states plainly that the precedence advantage does not
improve answers on this benchmark, with the serial-number mechanism as the explanation. The paper's
claim for the retention score therefore rests on the retention experiments (H3, H4, H5), where an error
deletes a memory and no reader can compensate.

**Exploratory status.** Added after seeing the H2 test result, so it is labelled secondary and
exploratory; it is not part of the Holm-corrected primary family.

---

## 2026-09-13 — secondary analysis: how accurate must feedback be?

**What.** `benchmarks/dars_eval/feedback_threshold.py` degrades the feedback that produced each
memory's success/failure counts and recomputes H4 (AUROC) and H5 (harmful deletion) as a function of
the verdict error rate ε. Artifacts: `benchmark_runs/revision/test/E12/feedback_threshold.json`.

**Why it is exact and offline.** In E9 and E10 retrieval is similarity for every policy (§3 of the
pre-registration), so a wrong verdict cannot change which memories were retrieved — only the verdict
recorded against them. Verdicts are independent and utility depends on the counts alone, so flipping a
fraction ε is exactly a binomial resample: s' = (s − Bin(s, ε)) + Bin(f, ε). No stream is re-run and no
API call is made. Recency, FIFO and LFU ignore verdicts, so they are flat reference lines.

**Result (MSC, test split).** Eviction stops beating FIFO at ε ≈ 0.13, and restatement prediction stops
beating recency-only at ε ≈ 0.41. The judge measured in E5 errs at ε ≈ 0.37 (accuracy 0.634). So the
deployed judge is past the point where utility-based eviction is worth having, though still marginally
useful for ranking.

**Caveat.** Flips are independent and symmetric here; a real judge's errors are correlated and biased,
so ε is a first-order proxy for judge error rather than an identity. Stated as such in the manuscript.

**Validation.** ALFWorld's selected H5 configuration is P-only (w_u = 0), and its curve is flat at
0.096 across the whole ε grid — confirming the resample perturbs utility and nothing else.

**Exploratory status.** Added after seeing the E5 and E9 test results; labelled secondary, not part of
the Holm-corrected primary family. Tests: `tests/test_feedback_threshold_unit.py` (6).

---

## 2026-09-12T21:17Z — fourth audit: how the primary results must be reported, and what is still owed

**Nature of this entry.** An adversarial audit of the finished test phase, checking every primary and secondary
result against its code path. Nothing here changes a hypothesis, endpoint, comparator, selected configuration or
the Holm-corrected table; it changes how those results are *described*, and it lists analyses that the
pre-registration promised but the test phase did not run. Every item below was verified on the artifacts before
this entry was written; every analysis it announces is exploratory unless it belongs to the confirmatory addendum
(`experiments/preregistration_addendum.md`).

### Findings that change the reporting of primary results

- **H1 evidence recall is non-inferior by construction (5 of 9 H1 comparisons).** `dars_rrf_k15` reranks the 15
  nearest memories and fills further budget slots in similarity order (`rankers.rank`), while every source shows
  36–269 units at B = 5120 (`E1/*/summary.json`, `n_units@5120`). The shown set therefore equals similarity's
  and the difference is exactly zero. These rows stay in the table as registered, flagged "by construction".
  Where the sets differ (B = 1024), DARS recall is lower (LongMemEval 0.473 vs 0.499; RULER QA2 0.771 vs 0.807).
- **H3's selected configuration is frequency-only, with similarity as tie-break.** Weights (0, 1, 0, 0), mode
  `score_only`; ties in S follow similarity order (`run_alfworld._fused_order`). The same ranking with random
  tie-breaks gives p = 0.051 instead of 0.0096 on test_in. The count prior (MRR 0.626 / 0.650) exceeds the
  selected configuration (0.510 / 0.485); utility-only equals similarity (p = 0.96).
- **H4's pre-registered comparators are the weakest available.** Frequency-only has AUROC 0.243 (below chance).
  FIFO (0.866) and utility-only (0.875) are within 0.025 of the selected score (0.891). A two-feature logistic
  model on creation session and mention count, fitted on the MSC dev split, reaches test AUROC 0.917. DeLong
  intervals ignore dialogue clustering; a dialogue-clustered bootstrap keeps the differences significant
  (DARS − utility +0.015 [0.013, 0.017]; DARS − FIFO +0.024 [0.018, 0.031]).
- **H5 on ALFWorld is carried by P alone** (weights (0, 0, 0, 1)), a static similarity to the domain goal. LRU and
  FIFO delete almost every needed memory; the informative comparators are LFU (0.330 / 0.374) and random. Default
  weights lose to LFU, and to random at keep 75 %.
- **H5 on LongMemEval was measured without a reader** (`reader: None`), so it is a proxy until validated.
- **Feedback in the LongMemEval stream lowers ranking quality even when perfect**: MRR 0.278 (none) vs 0.158
  (oracle) for rrf_k50, 0.356 vs 0.316 for blend; injected noise *raises* MRR. The noise curve therefore cannot be
  reported as robustness. The stream credits one verdict to every shown memory (`run_stream`), whereas MSC and
  ALFWorld credit each memory separately, so feedback granularity is confounded with dataset.
- **P harms query-time ranking in every variant, including the oracle** (E4 test).
- **E6 shadow indexing:** only the evidence memory was re-embedded (`run_layerc`), so the comparison favours
  re-embedding.
- **E3 tiers are degenerate**: default weights place 98 % of MSC facts in COMPRESS and 0 % in DELETE, and 80 % of
  ALFWorld memories in DELETE with 0 % in RETAIN; threshold shifts of ±0.05–0.10 give κ between −0.01 and 0.33 on MSC.

### Pre-registered analyses not run on the test split

- Cross-domain weight transfer (§5): not implemented.
- Dirichlet weight perturbation and λ sensitivity (§5, §7): run on dev only.
- Exact McNemar (§8): not used. H1 reader outcomes are per-question means over three seeds, so they are not
  binary; the paired clustered bootstrap is the test applied. The Methods will say so.

These will be run on the existing test artifacts and reported as secondary analyses.

### Analyses announced here, before they are run (exploratory)

Label-threshold sensitivity of the selected H4/H5 configurations; tie-break sensitivity for all ALFWorld rankers;
recall and MRR at budgets where DARS changes the shown set; minimum detectable effects for inconclusive H1
endpoints; dev-to-test drift of every selection; ALFWorld needed-memory composition and count-prior / LFU eviction;
LongMemEval evidence-age profile; need persistence per dataset; clustered AUROC; per-memory oracle credit
(`oracle_unit`) in the LongMemEval ranking and eviction streams; whole-context re-embedding in E6; LongMemEval
eviction with the reader; FactConsolidation reader check without serial prefixes; Layer A reformulation on
LongMemEval; a BM25 first stage for DARS; comparison with MemoryAgentBench's published numbers; Generative-Agents
and MemoryBank baselines; verdict-subsampling extension of the feedback-reliability analysis.

### Label-free checks on data not yet used

The MSC Hugging Face `validation` (500) and `test` (501) splits, never loaded by any run (`run_msc` uses
`split="train"`), have the train schema, five sessions per dialogue, and zero overlap with the train dialogues
used here or with each other (hashes of normalised session-0 personas and session-1 turns). No label or score
was computed on them.

### Data safety

All LLM responses delivered through the Batch API existed only on the EC2 volume. The caches (46,740 responses)
and `benchmark_runs/revision` were archived to S3 and to local storage with SHA-256 checksums
(`20260912T210925Z`) before any further work.

---

## 2026-09-13T04:59Z — confirmatory addendum frozen

`experiments/preregistration_addendum.md` v1.0 was frozen with SHA-256 `1cb19107d358f582eb9826edbd4afe1bd609bb054a5e0e2e888afba48d801159`; its configuration file
`experiments/addendum_config.json` has SHA-256 `da9c66e35a50ab9ea14cca5050b0ec43b7853278a3fd21fbfb31af27c2ad34c2`. The write-side configurations and the metadata model
were selected on the MSC dev dialogues (`benchmark_runs/revision/addendum/msc_dev_writes`, a re-run of the E9 dev
stream with write histories recorded).

**Consistency checks before freezing.**
- The re-run dev rows equal the original E9 dev rows except for the new fields and a floating-point difference
  in P of at most 1.3e-7, from the embedding cache introduced after the original dev run.
- The frozen read-side H4 and H5 selections re-derive exactly from the new rows.
- `run_msc run` with default options still writes byte-identical `facts.jsonl` (SHA-256 checked on 20 dev
  dialogues before and after the change).

**Rehearsal.** The confirmatory pipeline was rehearsed end to end on the dev rows only, with placeholder importance
ratings, to check that it runs. No validation or test dialogue of MSC has been streamed, labelled or scored.

**Correction, 2026-09-13T04:59Z.** The status line of the addendum gave the freeze date as 2026-09-12; the freeze happened on
2026-09-13 (UTC). The date was corrected before any confirmatory run, which changes the addendum's SHA-256 from
`1cb19107d358f582eb9826edbd4afe1bd609bb054a5e0e2e888afba48d801159` to `fc523c7d1ec5e471773b61b39c7cd07e0b37b370459afecbddd4f3da9f37ccf7`. The content is otherwise identical. `fc523c7d1ec5e471773b61b39c7cd07e0b37b370459afecbddd4f3da9f37ccf7` is the hash of record.

---

## 2026-09-13T05:42Z — designs of the reader-level exploratory runs (announced before they run)

All on test questions, reader `gpt-4o-mini-2024-07-18` (seed 0), answers delivered through the Batch API and
replayed offline (`experiments/aws/audit_batch.sh`):

- **Eviction with the reader (LongMemEval).** The registered H5 protocol (retrieval `dars_blend_a0.5`, set-level
  oracle feedback, B = 2048) at keep 25 / 50 / 75 % for DARS, LRU, FIFO, LFU and random (seed 0), plus an
  unlimited store. Outcome: reader substring exact match, and its association with harmful deletion.
- **Serial numbers in FactConsolidation (sh_32k, mh_32k).** A 2 × 2 design: serial prefix in the memory text
  (kept / removed) × MemoryAgentBench's instruction to resolve conflicts by serial number (kept / removed),
  with similarity, BM25, `dars_rrf_k50` and DARS over a BM25 first stage (`bm25_dars_wrrf_k50_b0.5`), feedback
  none, B = 256, λ = 0.001. The kept/kept cell repeats the earlier reader check. The MemoryAgentBench query
  template itself instructs the reader to pick the fact with the larger serial number; removing both is the
  condition in which only the retriever can supply recency.
- **Layer A on LongMemEval.** The E7 factorial unchanged, on `longmemeval_s*` (evidence recall 0.72 at
  B = 5120), where query reformulation has room to help.
- **Whole-context shadow-indexing test (E6).** Every memory of each E6 context compressed and re-embedded; the
  semantic compressor runs on gpt-4.1-nano, cached reader answers are reused.

Result summaries go to `revision_submission/PROJECT_BRIEF.md`; all are exploratory.

---

## 2026-09-13T05:53Z — confirmatory addendum run (primary family), completed once

Run exactly as `experiments/preregistration_addendum.md` §3 specifies. There were no software failures and no re-runs.
- **Streams:** MSC `validation` and `test`, horizons A (label session 3) and B (label session 4).
- **Facts, horizon A:** 15,251, from 1,001 dialogues.
- **Importance ratings:** 23,960 distinct texts. Delivered through the Batch API and replayed from cache, with 0 API calls on replay. 583 replies (2.4 %) were unparseable, cut off by the 4-token limit, and counted as 5 as specified. Ratings file SHA-256 `04093bd447436f794cce256df25d3827fef8c0d5d4eed7da4c3447a8e143f45f`.
- **Output:** `benchmark_runs/revision/addendum/confirm/confirm_primary.json`, SHA-256 `8e5a86bd3613b9fddbdccdc955d4daf08227eeddcdc139dea45f5c0fdb126baf`.

**Outcome of the primary family.** All 11 comparisons were confirmed after Holm correction. By the rule fixed in §5 (C3 and C4 both confirmed), the write-side thesis is **supported**.

**Reported with it, as specified in §6:**
- Per split, the H5 replications (C2a, C2b) are not confirmed on the `test` split alone. All other comparisons are confirmed in both splits.
- The label base rate in these splits (0.643) is higher than in the train-split study (0.448).
- The metadata model matches or exceeds write-side DARS at the stricter labels, and in eviction.

---

## 2026-09-13T06:33Z — follow-up announced after the serial-number factorial: display order under conflict

**The trigger.** In the FactConsolidation serial factorial, with prefixes and the MemoryAgentBench prompt:
- DARS over a BM25 first stage shows both versions of a fact, with the newest ranked *first*, and answers 0.514;
- BM25 alone shows the newest version *later* and answers 0.886.

**The question.** Does the reader favour the last-shown memory when memories conflict?

**The design.** `run_stream --display-order best_last` passes the same shown memories to the reader in
reverse rank order, so the highest-ranked memory is next to the query. It is run on sh_32k and mh_32k with
similarity, BM25, `dars_rrf_k50` and `bm25_dars_wrrf_k50_b0.5` (B = 256, λ = 0.001, feedback none,
MemoryAgentBench prompt, prefixes kept). The best_first cell repeats the registered reader check.
Exploratory.

**Also recorded: the H5 proxy check.** Pooled over the LongMemEval eviction-with-reader streams, reader
accuracy was 0.200 [0.150, 0.249] when any evidence group had been evicted before the question (650
question-streams), against 0.331 [0.288, 0.384] otherwise (2,410).

---

## 2026-09-13T06:43Z — addendum 2 frozen (display order under conflict)

`experiments/preregistration_addendum_2.md` v1.0 was frozen with SHA-256 `14bb17f2694721e59e74bf92aa67edd5751cc68d41aa72af087fa343cbaf4d30` before any run on
`factconsolidation_sh_64k` or `factconsolidation_sh_262k`, which no earlier run had loaded.
- **Analysis script:** `benchmarks/dars_eval/confirm_display_order.py`. It refuses to run unless the addendum
  is frozen and its hash matches.
- **Rehearsal:** the analysis was rehearsed only on the already-seen sh_32k runs, where it reproduces
  0.929 vs 0.886 (D1, not significant at n = 70) and 0.814 vs 0.600 (D2).

---

## 2026-09-13T06:57Z — addendum 2 run once; importance-rating protocol for the exploratory prior-art baselines

**Addendum 2 outcome** (`benchmark_runs/revision/addendum2/confirm_display_order.json`, SHA-256 `f7b7fd59c1eba2606c49815f5f79fe2b9036cbd47d3255cf799746275f8ee242`).
The run followed `experiments/preregistration_addendum_2.md`; the replay made 0 API calls.
- **D2a, D2b confirmed:** DARS answered better when its top-ranked memory was shown last. sh_64k: 0.557 → 0.900
  (+0.343). sh_262k: 0.557 → 0.771 (+0.214, Holm p = 0.010).
- **D1a, D1b not confirmed:** DARS over BM25 shown best-last did not beat BM25 as MemoryAgentBench displays it.
  sh_64k: 0.900 vs 0.857. sh_262k: 0.757 vs 0.800.
- **By the pre-registered rule,** the display-order account is supported, and the claim that DARS over BM25 improves on
  the benchmark's display is not.

**Importance ratings for the LongMemEval and ALFWorld prior-art baselines** (exploratory; the MSC addendum's ratings are
unchanged).
- **4-token limit:** 73 % of replies were unparseable, because long conversational turns draw a sentence before the
  number. The results computed from those ratings were discarded before use and the files kept as
  `importance_lme_alfworld_4tok_invalid.*`.
- **32-token limit, with a parser that ignores the stated scale** (`retention_signals.parse_rating_verbose`):
  29 % were still unparseable, so this attempt was superseded as well.
- **96-token limit:** 11 of 13,314 unparseable. These are the ratings used.
- **Applied the same way to every compared score.** The ALFWorld memory texts come from a re-run of the E10 stream
  with texts saved. It reproduces the pre-registered P-only harmful deletion exactly, and differs from the registered
  stream by 4 of 37,822 adjudications (embedding-cache precision).

---

## 2026-09-13T14:08Z — whole-context shadow-indexing test (E6) run; outcome

**What was run.** The whole-context condition announced on 2026-09-12 (`run_layerc --split test --whole-context`).
Every memory of each test context was compressed and re-embedded, so the evidence memory competes against equally
compressed text instead of full-length vectors.
- **Scope:** RULER QA1 (1,489 memories), QA2 (3,719) and the five LongMemEval contexts (2,305–2,483 each).
- **Semantic compression:** 16,160 gpt-4.1-nano calls, 0 failures, 0 rate-limit retries, US$0.70.
- **Reader calls:** all 2,064 replayed from cache (0 API calls).
- **Run window:** 06:20–09:04 UTC on the EC2 host.
- **Output:** `benchmark_runs/revision/test/E13_audit/e6_whole_context/`.

**Consistency with the registered E6.** Every field of the registered E6 summary is reproduced exactly from the new
item file, once computed with the registered 10,000 bootstrap resamples. The run itself used the script default of
2,000, so `summary.json` was recomputed offline with 10,000 (SHA-256
`d95c420ccf1b7ad84af15f4eef0681fa32984f7b63c8b068f368b73f6372ab31`; items
`2d23160e187771b282373b53b43ea0fa28fa0c3f6f0cb21dbd35d59b57fcce38`). The 2,000-resample file is kept as
`summary_nboot2000.json`. Conclusions are the same under both.

**Outcome** (recall@10 of the evidence memory; kept original vector 0.747 in every condition; n = 344 questions,
clustered by context; exploratory):

| Compressor | Re-embedded, evidence only | Re-embedded, whole context | Kept − whole context | Whole context − evidence only |
|---|---|---|---|---|
| Semantic (nano) | 0.823 | 0.802 | −0.055 [−0.139, +0.012], p = 0.117 | −0.020, p = 0.138 |
| LLMLingua-2 | 0.875 | 0.811 | −0.064 [−0.156, +0.007], p = 0.094 | −0.064, p = 0.036 |
| Extractive | 0.698 | 0.680 | +0.067 [+0.011, +0.119], p = 0.026 | −0.017, p = 0.235 |

- **Holm across the three fixed-rate compressors:** adjusted p = 0.187, 0.187 and 0.077. No difference between
  keeping the original vector and re-embedding the compressed text is significant.
- **The target-only test overstated re-embedding** by 0.020, 0.064 and 0.017. LLMLingua-2 at the matched rate
  shows the largest overstatement: −0.093, p = 0.001.

**What may be claimed.** Shadow indexing keeps a compressed memory's ranking exactly unchanged, by construction.
It did not make memories more retrievable than re-embedding their compressed text:
- **Abstractive compressors:** the point estimates favour re-embedding, not significantly.
- **Extractive compression:** the point estimate favours the kept vector, significant only before Holm correction.

The manuscript, response letter and SI state this and no stronger retrievability claim.

---

## 2026-09-13T15:43Z — publication archive built and audited; code state recorded

No new analysis of study data. This entry records how the archive was assembled, what its checks found, the
corrections they caused, and the code state behind every result.

**Assembly** (`scripts/build_archive.py`, with unit tests).
- **Contents:** the code; run artifacts; the pre-registration, addenda and this log; the embedding cache (vectors
  keyed by hashes); and the LLM response cache, with every prompt replaced by its SHA-256.
- **Merged caches.** The laptop and EC2 caches each held entries the other lacked.
  - LLM responses: 84,426 + 73,900, of which 20,524 are shared and identical in response, usage and request, giving
    137,802.
  - Embeddings: 173,104 + 265,793, of which 104,477 are shared and byte-identical, giving 334,420.
  - Compression caches: the laptop's is a subset of the EC2 copy, with no conflicting entry.
- **Working trees:** laptop and EC2 were brought to identical file checksums (1,056 files) before the build.

**Checks on the audited build** (tarball SHA-256 `3568685e55a8f5b5d7512135b57d4acda0bbe89cbb881a43ad4b4de8f7158ca1`).
- **Redacted cache:** 0 of 137,802 cache records still carry prompt text.
- **Secret scan.** It covers provider key patterns, private keys, the local key values, and every key-shaped value
  in the git history.
  - **First build:** it failed. The Gemini key committed in `df0ae5c` is also present in the tracked
    `claude_review.md` (line 281), including at `HEAD` and on the public remote. The working-tree copy was replaced
    by a placeholder; rotating the key remains with the authors.
  - **Rebuild:** 0 blocking findings. The 6 remaining matches are synthetic placeholders in unit tests.
- **Novel text** (EventQA, DetectiveQA and the other novel-derived MemoryAgentBench sources, whose novels carry no
  stated licence). Every archived text file was scanned for 12-word spans of the novels.
  - 73 files contain such a span, all of them model outputs; the longest span is 26 words.
  - Other matches are only with the benchmark's own questions, answers and event lists (MIT licence).

**Replay audit** (outputs in `benchmark_runs/revision/test/E13_audit/replay_audit/`; summary in
`replay_audit.json`).
- **Set-up:** the tarball was unpacked into a fresh directory with a fresh virtual environment, a placeholder API key
  and network access blocked.
- **What was replayed:**
  - E1 RULER QA1 with the reader;
  - E11 DetectiveQA, whose prompts were rebuilt from the dataset;
  - E5;
  - the four addendum-2 streams and their analysis;
  - the MSC confirmatory data build (test and validation splits);
  - both addendum-1 analyses;
  - the primary and exploratory tables, and the SI.
- **API calls:** all eight replayed manifests record 0 API calls.
- **Byte-identical:** E5, E11 and addendum 2.
- **E1:** summary and rankings are identical; similarity scores differ by at most 6.5e-8.
- **Addendum-1 analyses on the archived data:** statistics differ by at most 6.9e-7.
- **Tables and SI:** regenerated identically.
- **MSC data regenerated on Linux** (the archived data were built on Windows): one similarity tie resolved
  differently, exchanging retrieval-credited counts between 2 of 15,251 facts in one dialogue. No write-side or
  label field changed. Recomputing the primary confirmatory family from these data confirmed 11 of 11, as archived,
  with no decision changed overall or per split. The largest difference is 1.6e-4; one per-split interval bound that
  the paper does not report moves in the third decimal.

**Corrections made because of these checks.**
- **"The test suite runs without cloud credentials" was inaccurate.** Tests of the LLM-backed layers need a provider
  key.
  - Without a key, 378 pass and 27 are skipped.
  - With a placeholder key and the archived cache, all 405 pass with network access blocked.
  - `test_grace_period_bypassed_by_priority_tag` reaches the LLM compressor but lacked the `requires_llm` marker;
    the marker was added.
  - README, Methods, SI Note S12 and the response letter were corrected.
- **"No benchmark text is archived" was imprecise.** The artifacts contain short excerpts from permissively licensed
  sources, and model outputs can quote the novels. The Data availability statement and SI Note S15 now state exactly
  what the archive holds, and add FactConsolidation's source (MQuAKE, MIT) and EventQA's origin.
- **Also fixed during this work:**
  - nine SI tables whose run keys contained "|" were malformed; the generators now escape the character;
  - Fig. 7c now shows all three compressors.

**Code state (D5).** Every run after 2026-09-11 used commit `7b26e6b` plus working-tree changes (manifests record
`dirty: true`). On the laptop, excluding this log and its hash file:
- SHA-256 of `git diff --binary 7b26e6b`: `0d49cc0eff9650f82d679f4e20613863dcd3b153757efffc5f47705e6e511634`
  (49 tracked files changed);
- SHA-256 of the sorted per-file checksums of the 764 untracked, non-ignored files:
  `308d524cd5ebe65ace56b2722c8569766337fba850089f0a24581edc228069c4`.

The frozen pre-registration (`a3dec99a…`), addendum 1 (`fc523c7d…`), addendum 2 (`14bb17f2…`) and the addendum
configuration (`da9c66e3…`) still match their recorded hashes.

---

## 2026-09-13T16:25Z — submission-format corrections (no analysis)

Checks on the built documents found four presentation defects. None changes a number.

- **Figure order.** Figures were not cited in numerical order, and Figs 1 and 7 were not cited in the text. The
  figures were renumbered by first citation, and citations were added:
  - Fig. 1 architecture; Fig. 2 static retrieval; Fig. 3 display order; Fig. 4 write- versus read-anchored signals;
    Fig. 5 mechanisms; Fig. 6 feedback reliability; Fig. 7 layers.
  - `make_figures_revised.py` now writes all seven figures under these numbers.
  - One Layer A citation had pointed to the static-retrieval figure.
- **Display items.** Five inline tables in the main text and Methods would have exceeded the journal's limit of eight
  display items. Their contents are now given as text, with the same numbers. Table 1 is the only table.
- **Cross-references.** "Table Table 1" and "Fig. Fig." were left over from filling cross-reference placeholders.
- **Word conversion.** `scripts/build_docx.py` turned every hard-wrapped Markdown line into its own Word paragraph,
  and counted words only under top-level headings, which under-counted the main text.
  - Wrapped lines now join into their paragraph or list item.
  - Code fences render as monospace lines.
  - The count includes subsections.
  - Unit tests were added.

**Code state after these corrections (supersedes the D5 hashes of 15:43Z).** On the laptop, excluding this log and its
hash file:
- SHA-256 of `git diff --binary 7b26e6b`: `0d49cc0eff9650f82d679f4e20613863dcd3b153757efffc5f47705e6e511634` (49 tracked files changed);
- SHA-256 of the sorted per-file checksums of the 766 untracked, non-ignored files: `9cb7345acf4a38ce1d9809cbb5405f9703e666cda7f4d0201246be78ea19f11a`.

---

## 2026-09-13T16:20Z — timestamp correction

The previous entry's heading and the two hash-log lines that follow it read 16:25Z and 16:27Z. Those times were typed
rather than read from the clock, and they are wrong. The entry was written at about 16:10Z, and its closing code-state
paragraph at 16:11Z (file modification time 16:11:13Z). The content is unchanged and the recorded hashes stay valid.

---

## 2026-09-13T19:13Z — reporting corrections found while reviewing the response letter (no analysis)

- **Efficiency (R2.7).**
  - The letter listed budgets of 1,024–16,384 tokens for every method, plus accuracy-versus-budget curves and per-query
    cost. In fact, retrieval methods were compared at those budgets in E1, with reader accuracy at 5,120. Compression
    (E8) used 512, 1,024 and 2,048 tokens, with reader accuracy at 1,024. No per-query cost was computed; the total
    API cost is in SI Note S13. The letter and SI Note S8 now say this.
  - E8's manifest note named the laptop CPU, but E8 ran on EC2, as its provenance records. The note in
    `run_efficiency.py` no longer names a host; SI Note S8 gives the recorded platform.
- **Per-memory feedback credit.** "Removed most of the harm" held only for blending (0.350 against 0.316, with 0.356
  without feedback). With rank fusion it halved the harm (0.218 against 0.158, with 0.278 without feedback). The
  manuscript and letter now give both.
- **Test counts.** The manuscript and letter said 405 tests. The re-checked archive build ran 407 (the two document-
  builder tests were added later): all pass with the archived cache, and without a key 380 pass and 27 are skipped. SI
  Note S12 now reads these counts from that check (`E13_audit/replay_audit/out/final_archive_check.log`).
- **ALFWorld task counts.** The Methods now give the 3,553 training tasks as 3,198 streamed plus 355 for development,
  matching the letter.
