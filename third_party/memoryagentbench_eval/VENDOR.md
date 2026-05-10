# Vendored MemoryAgentBench evaluation utilities

## Origin

- **Repository:** https://github.com/HUST-AI-HYZ/MemoryAgentBench  
- **Files:** `utils/eval_other_utils.py`, `utils/templates.py`  
- **License:** MIT (per upstream repository)  
- **DensePhrases attribution:** `eval_other_utils.py` header cites Princeton DensePhrases eval utilities.

## Why vendored

Keeps token chunking, DRQA-style metrics, query/memorize templates, and `post_process` routing aligned with the official benchmark while avoiding a full second checkout as a runtime dependency.

## Pinning for papers

Record the **git commit SHA** of upstream `main` at download time in each benchmark run manifest (`upstream_mab_commit` field). Re-download intentionally when bumping the pin.

## NLTK data

Chunking uses `nltk.sent_tokenize` and may download `punkt` / `punkt_tab` on first use. CI or air-gapped runs should pre-bundle NLTK data or set `NLTK_DATA`.
