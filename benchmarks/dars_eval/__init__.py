"""
DARS revision evaluation harness.

Modules
-------
provenance     – git / dataset / model / package versions recorded in every run manifest
memory_units   – per-dataset memory units that always fit the embedder's input window
labels         – deterministic evidence labels (RULER gold documents, LongMemEval
                 has_answer turns, FactConsolidation newest facts)
stats          – clustered bootstrap, paired tests, Holm correction, kappa, AUROC/DeLong
"""
