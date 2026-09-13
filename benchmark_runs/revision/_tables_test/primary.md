Primary comparisons (Holm–Bonferroni over 23 comparisons, α = 0.05).

| ID | Comparison | Effect [95% CI] | n | p | Holm p | Result | Note |
|---|---|---|---|---|---|---|---|
| H1 | ruler_qa1_197K: evidence recall@5120, dars_rrf_k15 − similarity (margin 0.02) | +0.000 [+0.000, +0.000] | 70 | < 1e-04 | < 2e-03 | non-inferior | shown set identical to similarity by construction for 100% of questions; the difference is zero by design |
| H1 | ruler_qa2_421K: evidence recall@5120, dars_rrf_k15 − similarity (margin 0.02) | +0.000 [+0.000, +0.000] | 70 | < 1e-04 | < 2e-03 | non-inferior | shown set identical to similarity by construction for 100% of questions; the difference is zero by design |
| H1 | longmemeval_s: evidence recall@5120, dars_rrf_k15 − similarity (margin 0.02) | +0.000 [+0.000, +0.000] | 204 | < 1e-04 | < 2e-03 | non-inferior | shown set identical to similarity by construction for 100% of questions; the difference is zero by design |
| H1 | factconsolidation_sh_32k: evidence recall@5120, dars_rrf_k15 − similarity (margin 0.02) | +0.000 [+0.000, +0.000] | 70 | < 1e-04 | < 2e-03 | non-inferior | shown set identical to similarity by construction for 100% of questions; the difference is zero by design |
| H1 | factconsolidation_mh_32k: evidence recall@5120, dars_rrf_k15 − similarity (margin 0.02) | +0.000 [+0.000, +0.000] | 70 | < 1e-04 | < 2e-03 | non-inferior | shown set identical to similarity by construction for 100% of questions; the difference is zero by design |
| H1 | eventqa_65536: reader EM@5120 (mean of seeds), dars_rrf_k15 − similarity (margin 0.02) | +0.000 [-0.022, +0.019] | 350 | 0.0742 | 0.148 | not shown |  |
| H1 | eventqa_full: reader EM@5120 (mean of seeds), dars_rrf_k15 − similarity (margin 0.02) | +0.022 [-0.008, +0.051] | 350 | 0.0062 | 0.0434 | non-inferior |  |
| H1 | ruler_qa1_197K: reader EM@5120 (mean of seeds), dars_rrf_k15 − similarity (margin 0.02) | +0.014 [+0.000, +0.043] | 70 | < 1e-04 | < 2e-03 | non-inferior |  |
| H1 | ruler_qa2_421K: reader EM@5120 (mean of seeds), dars_rrf_k15 − similarity (margin 0.02) | -0.048 [-0.114, +0.010] | 70 | 1 | 1 | not shown |  |
| H2 | factconsolidation_sh_32k: rank precedence, dars_rrf_k50 (λ 0.001) − similarity | +0.619 [+0.476, +0.762] | 42 | < 1e-04 | < 2e-03 | supported |  |
| H2 | factconsolidation_mh_32k: rank precedence, dars_rrf_k50 (λ 0.001) − similarity | +0.100 [+0.020, +0.180] | 50 | 0.011 | 0.048 | supported |  |
| H3 | ALFWorld test_in: location MRR, selected − similarity | +0.077 [+0.018, +0.137] | 140 | 0.0096 | 0.048 | supported |  |
| H3 | ALFWorld test_out: location MRR, selected − similarity | +0.116 [+0.059, +0.175] | 131 | < 1e-04 | < 2e-03 | supported | 3 of 134 tasks excluded (no alternative location memory) |
| H4 | MSC: AUROC, selected − recency_lru | +0.227 [+0.218, +0.237] | 13550 | < 1e-15 | < 2e-14 | supported | 13550 facts in 701 dialogues; DeLong treats facts as independent |
| H4 | MSC: AUROC, selected − frequency_lfu | +0.648 [+0.638, +0.657] | 13550 | < 1e-15 | < 2e-14 | supported | 13550 facts in 701 dialogues; DeLong treats facts as independent |
| H5 | LongMemEval: harmful deletion, DARS − recency_lru | +0.016 [+0.000, +0.038] | 204 | 0.0416 | 0.125 | not significant |  |
| H5 | MSC: harmful deletion, selected − recency_lru | -0.285 [-0.296, -0.274] | 701 | < 1e-04 | < 2e-03 | supported |  |
| H5 | ALFWorld test_in: harmful deletion, selected − recency_lru | -0.904 [-0.932, -0.873] | 140 | < 1e-04 | < 2e-03 | supported |  |
| H5 | ALFWorld test_out: harmful deletion, selected − recency_lru | -0.862 [-0.894, -0.830] | 134 | < 1e-04 | < 2e-03 | supported |  |
| H5 | LongMemEval: harmful deletion, DARS − fifo | +0.024 [+0.004, +0.051] | 204 | 0.0078 | 0.0468 | significant, opposite direction |  |
| H5 | MSC: harmful deletion, selected − fifo | -0.028 [-0.036, -0.020] | 701 | < 1e-04 | < 2e-03 | supported |  |
| H5 | ALFWorld test_in: harmful deletion, selected − fifo | -0.898 [-0.927, -0.867] | 140 | < 1e-04 | < 2e-03 | supported |  |
| H5 | ALFWorld test_out: harmful deletion, selected − fifo | -0.862 [-0.894, -0.830] | 134 | < 1e-04 | < 2e-03 | supported |  |

H6 — judge κ = 0.263 [0.036, 0.473] (n = 968, NEUTRAL 0.0%; threshold 0.60): fail
