#!/usr/bin/env bash
# Fourth-audit exploratory streams that need no LLM call (announced in experiments/deviations_post_freeze.md).
# Results go to benchmark_runs/revision/test/E13_audit/. Each run writes a .done marker so the script resumes.
set -uo pipefail
REPO="$HOME/dars/Dynamic-Adaptive-Retention-Framework"
cd "$REPO" || exit 1
export PATH="$REPO/.venv/bin:$HOME/.local/bin:$PATH"
export DARS_EMBED_CACHE=benchmark_runs/_emb_cache/minilm.sqlite
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONIOENCODING=utf-8 DARS_LLM_OFFLINE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
OUT=benchmark_runs/revision/test/E13_audit
N_BOOT=10000
mkdir -p "$OUT/_done"
LOG="$OUT/audit_cpu.log"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

step() {   # step <name> <command...>
  local name="$1"; shift
  [ -f "$OUT/_done/$name" ] && { say "skip $name"; return 0; }
  say "run  $name"
  if "$@" >> "$OUT/logs_$name.log" 2>&1; then touch "$OUT/_done/$name"; say "done $name"; else say "FAIL $name"; fi
}

LME="longmemeval_s*"

# L19 — per-memory oracle credit vs the registered one-verdict-for-all oracle (ranking)
step lme_rank_oracle_unit python -m benchmarks.dars_eval.run_stream --source "$LME" --out "$OUT/lme_rank_oracle_unit" \
  --methods dars_rrf_k50,dars_wrrf_k50_b0.5,dars_blend_a0.5 --feedback none oracle oracle_unit \
  --budget 2048 --report-split test --n-boot $N_BOOT &

# L4 — DARS over a BM25 first stage: FactConsolidation precedence (H2 setting) and LongMemEval ranking
for src in factconsolidation_sh_32k factconsolidation_mh_32k; do
  step "fc_bm25_first_stage_$src" python -m benchmarks.dars_eval.run_stream --source $src --out "$OUT/fc_bm25_first_stage/$src" \
    --methods bm25,similarity,dars_rrf_k50,bm25_dars_rrf_k50,bm25_dars_wrrf_k50_b0.5,bm25_dars_blend_a0.5 \
    --feedback none --budget 256 --decay-lambda 0.001 --report-split test --n-boot $N_BOOT &
done
wait
step lme_bm25_first_stage python -m benchmarks.dars_eval.run_stream --source "$LME" --out "$OUT/lme_bm25_first_stage" \
  --methods bm25,bm25_dars_rrf_k50,bm25_dars_wrrf_k50_b0.5,bm25_dars_blend_a0.5 --feedback none oracle_unit \
  --budget 2048 --report-split test --n-boot $N_BOOT &

# L19 — eviction under per-memory credit, same protocol as H5 and the budget sweep otherwise
for keep in 0.25 0.5 0.75; do
  for ev in dars lru fifo lfu; do
    step "lme_evict_unit_${keep}_$ev" python -m benchmarks.dars_eval.run_stream --source "$LME" \
      --out "$OUT/lme_evict_oracle_unit/keep${keep}/$ev" --methods dars_blend_a0.5 --feedback oracle_unit \
      --budget 2048 --memory-budget $keep --eviction $ev --report-split test --n-boot $N_BOOT &
  done
  wait
done

# P1 — cross-domain transfer to LongMemEval eviction (registered protocol: set-level oracle, keep 50 %).
# The decay rate stays at the registered 0.005 because the vault uses one rate for retrieval and eviction;
# changing it would also change retrieval. Only the eviction weights transfer.
declare -A W=( [msc_h4]="0.2 0 0.8 0" [msc_h5]="0.3 0 0.7 0" [alfworld_h3]="0 1 0 0" [alfworld_h5]="0 0 0 1" )
declare -A L=( [msc_h4]=0.05 [msc_h5]=0.0005 [alfworld_h3]=0.0005 [alfworld_h5]=0.0005 )
for name in msc_h4 msc_h5 alfworld_h3 alfworld_h5; do
  step "lme_transfer_$name" python -m benchmarks.dars_eval.run_stream --source "$LME" --out "$OUT/lme_transfer/$name" \
    --methods dars_blend_a0.5 --feedback oracle --budget 2048 --memory-budget 0.5 --eviction dars \
    --eviction-weights ${W[$name]} --report-split test --n-boot $N_BOOT &
done
wait
say "audit_cpu finished"
