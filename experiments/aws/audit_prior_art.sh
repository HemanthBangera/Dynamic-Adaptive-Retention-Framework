#!/usr/bin/env bash
# K3 — prior-art eviction baselines on LongMemEval (registered H5 protocol otherwise): MemoryBank and
# Generative Agents (recency + gpt-4.1-nano importance). No LLM call at run time: ratings come from
# E13_audit/importance_lme_alfworld.jsonl.
set -uo pipefail
REPO="$HOME/dars/Dynamic-Adaptive-Retention-Framework"
cd "$REPO" || exit 1
export PATH="$REPO/.venv/bin:$HOME/.local/bin:$PATH"
export DARS_EMBED_CACHE=benchmark_runs/_emb_cache/minilm.sqlite
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONIOENCODING=utf-8 DARS_LLM_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
OUT=benchmark_runs/revision/test/E13_audit
IMP=$OUT/importance_lme_alfworld.jsonl
for keep in 0.25 0.5 0.75; do
  for ev in memorybank generative_agents; do
    python -m benchmarks.dars_eval.run_stream --source "longmemeval_s*" --out "$OUT/lme_evict_prior_art/keep$keep/$ev" \
      --methods dars_blend_a0.5 --feedback oracle --budget 2048 --memory-budget $keep --eviction $ev \
      --importance-file "$IMP" --report-split test --n-boot 10000 >> "$OUT/lme_evict_prior_art.log" 2>&1 &
  done
done
wait
echo "prior-art eviction finished $(date -u +%FT%TZ)" >> "$OUT/lme_evict_prior_art.log"
