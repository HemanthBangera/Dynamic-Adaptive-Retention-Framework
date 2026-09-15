Primary comparisons (Holm–Bonferroni over 19 comparisons, α = 0.05).

| ID | Comparison | Effect [95% CI] | n | p | Holm p | Result |
|---|---|---|---|---|---|---|
| H1 | ruler_qa1_197K: evidence recall@5120, dars_rrf_k15 − similarity (margin 0.02) | +0.000 [+0.000, +0.000] | 30 | 0 | 0 | non-inferior |
| H1 | ruler_qa2_421K: evidence recall@5120, dars_rrf_k15 − similarity (margin 0.02) | +0.000 [+0.000, +0.000] | 30 | 0 | 0 | non-inferior |
| H1 | longmemeval_s: evidence recall@5120, dars_rrf_k15 − similarity (margin 0.02) | +0.000 [+0.000, +0.000] | 86 | 0 | 0 | non-inferior |
| H1 | factconsolidation_sh_32k: evidence recall@5120, dars_rrf_k15 − similarity (margin 0.02) | +0.000 [+0.000, +0.000] | 30 | 0 | 0 | non-inferior |
| H1 | factconsolidation_mh_32k: evidence recall@5120, dars_rrf_k15 − similarity (margin 0.02) | +0.000 [+0.000, +0.000] | 30 | 0 | 0 | non-inferior |
| H1 | eventqa_65536: reader EM@5120 (mean of seeds), dars_rrf_k15 − similarity (margin 0.02) | +0.007 [-0.040, +0.067] | 150 | 0.356 | 1 | not shown |
| H1 | eventqa_full: reader EM@5120 (mean of seeds), dars_rrf_k15 − similarity (margin 0.02) | — | — | — | — | missing |
| H1 | ruler_qa1_197K: reader EM@5120 (mean of seeds), dars_rrf_k15 − similarity (margin 0.02) | +0.000 [+0.000, +0.000] | 30 | 0 | 0 | non-inferior |
| H1 | ruler_qa2_421K: reader EM@5120 (mean of seeds), dars_rrf_k15 − similarity (margin 0.02) | +0.000 [-0.100, +0.100] | 30 | 0.697 | 1 | not shown |
| H2 | factconsolidation_sh_32k: rank precedence, dars_rrf_k50 (λ 0.001) − similarity | +0.619 [+0.381, +0.810] | 21 | 0 | 0 | supported |
| H2 | factconsolidation_mh_32k: rank precedence, dars_rrf_k50 (λ 0.001) − similarity | +0.056 [+0.000, +0.167] | 18 | 0.709 | 1 | not significant |
| H3 | ALFWorld dev: location MRR, selected − similarity | +0.053 [+0.019, +0.089] | 351 | 0.0032 | 0.0192 | supported |
| H4 | MSC: AUROC, selected − recency_lru | +0.236 [+0.222, +0.251] |  | 6.95e-221 | 4.86e-220 | supported |
| H4 | MSC: AUROC, selected − frequency_lfu | +0.649 [+0.634, +0.664] |  | 0 | 0 | supported |
| H5 | LongMemEval: harmful deletion, DARS − recency_lru | +0.006 [-0.029, +0.046] | 86 | 0.943 | 1 | not significant |
| H5 | MSC: harmful deletion, selected − recency_lru | -0.287 [-0.304, -0.270] | 300 | 0 | 0 | supported |
| H5 | ALFWorld dev: harmful deletion, selected − recency_lru | -0.908 [-0.926, -0.890] | 355 | 0 | 0 | supported |
| H5 | LongMemEval: harmful deletion, DARS − fifo | -0.002 [-0.041, +0.041] | 86 | 0.961 | 1 | not significant |
| H5 | MSC: harmful deletion, selected − fifo | -0.030 [-0.041, -0.019] | 300 | 0 | 0 | supported |
| H5 | ALFWorld dev: harmful deletion, selected − fifo | -0.901 [-0.919, -0.882] | 355 | 0 | 0 | supported |

H6 — judge κ = 0.202 [0.052, 0.330] (n = 1022, NEUTRAL 0.0%; threshold 0.60): fail
