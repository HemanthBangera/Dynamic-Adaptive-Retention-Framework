#!/usr/bin/env bash
# Fourth-audit exploratory stages that need gpt-4o-mini reader calls, delivered through the Batch API.
# Announced in experiments/deviations_post_freeze.md before running. Same three steps as batch_stage.sh:
#   collect  run each command with DARS_LLM_COLLECT (reader requests written out, scratch outputs)
#   batch    submit the collected requests; answers land in the response cache
#   replay   run the commands for real with DARS_LLM_OFFLINE=1, so every reader call must be a cache hit
# Usage: bash experiments/aws/audit_batch.sh <lme_evict_reader | fc_serial | fc_order | lme_layera>
set -uo pipefail
STAGE="${1:?usage: audit_batch.sh <stage>}"
REPO="$HOME/dars/Dynamic-Adaptive-Retention-Framework"
cd "$REPO" || exit 1
export PATH="$REPO/.venv/bin:$HOME/.local/bin:$PATH"
export DARS_EMBED_CACHE=benchmark_runs/_emb_cache/minilm.sqlite
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
OUT=benchmark_runs/revision/test/E13_audit
COLLECT="$HOME/collect/audit_$STAGE"
SCRATCH="/tmp/audit_$STAGE"
LOG="$OUT/audit_batch_$STAGE.log"
N_BOOT=10000
mkdir -p "$OUT" "$COLLECT"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
LME="longmemeval_s*"

# commands <root> <nboot> : one line per job, run 4 at a time
commands() {
  local root="$1" nb="$2"
  case "$STAGE" in
    lme_evict_reader)
      echo "python -m benchmarks.dars_eval.run_stream --source '$LME' --out $root/unlimited --methods dars_blend_a0.5 --feedback oracle --budget 2048 --reader --report-split test --n-boot $nb"
      for keep in 0.25 0.5 0.75; do
        for ev in dars lru fifo lfu random; do
          echo "python -m benchmarks.dars_eval.run_stream --source '$LME' --out $root/keep$keep/$ev --methods dars_blend_a0.5 --feedback oracle --budget 2048 --memory-budget $keep --eviction $ev --reader --report-split test --n-boot $nb"
        done
      done ;;
    fc_serial)
      for src in factconsolidation_sh_32k factconsolidation_mh_32k; do
        for prefix in with without; do
          for prompt in mab no_serial_rule; do
            flags="--fc-prompt $prompt"
            [ "$prefix" = without ] && flags="$flags --no-serial-prefix"
            echo "python -m benchmarks.dars_eval.run_stream --source $src --out $root/$src/prefix_${prefix}__prompt_$prompt --methods similarity,bm25,dars_rrf_k50,bm25_dars_wrrf_k50_b0.5 --feedback none --budget 256 --decay-lambda 0.001 --reader $flags --report-split test --n-boot $nb"
          done
        done
      done ;;
    addendum2)
      for src in factconsolidation_sh_64k factconsolidation_sh_262k; do
        for order in best_first best_last; do
          echo "python -m benchmarks.dars_eval.run_stream --source $src --out $root/$src/order_$order --methods similarity,bm25,dars_rrf_k50,bm25_dars_wrrf_k50_b0.5 --feedback none --budget 256 --decay-lambda 0.001 --reader --display-order $order --report-split test --n-boot $nb"
        done
      done ;;
    fc_order)
      for src in factconsolidation_sh_32k factconsolidation_mh_32k; do
        for order in best_first best_last; do
          echo "python -m benchmarks.dars_eval.run_stream --source $src --out $root/$src/order_$order --methods similarity,bm25,dars_rrf_k50,bm25_dars_wrrf_k50_b0.5 --feedback none --budget 256 --decay-lambda 0.001 --reader --display-order $order --report-split test --n-boot $nb"
        done
      done ;;
    lme_layera)
      echo "python -m benchmarks.dars_eval.run_layera --sources '$LME' --split test --method dars_rrf_k15 --out $root --n-boot $nb" ;;
    *) echo "unknown stage $STAGE" >&2; exit 2 ;;
  esac
}

run_all() {   # run_all <root> <nboot> <env prefix> : 4 jobs at a time, one collect file per job
  local root="$1" nb="$2" mode="$3" i=0
  while IFS= read -r cmd; do
    i=$((i + 1))
    if [ "$mode" = collect ]; then
      ( DARS_LLM_COLLECT="$COLLECT/job$i.jsonl" DARS_LLM_COLLECT_MODEL=gpt-4o-mini-2024-07-18 \
          bash -c "$cmd" >> "$LOG.collect" 2>&1 || say "collect job $i failed: $cmd" ) &
    else
      ( DARS_LLM_OFFLINE=1 bash -c "$cmd" >> "$LOG.replay" 2>&1 || say "replay job $i failed: $cmd" ) &
    fi
    [ $((i % 4)) -eq 0 ] && wait
  done < <(commands "$root" "$nb")
  wait
}

if [ ! -f "$COLLECT/.collected" ]; then
  say "collect $STAGE"
  rm -rf "$SCRATCH" "$COLLECT"/*.jsonl
  run_all "$SCRATCH" 10 collect
  touch "$COLLECT/.collected"
fi
# batch_run.py takes one request file: merge the per-job files (duplicates are skipped by the scheduler)
cat "$COLLECT"/job*.jsonl > "$COLLECT/all_requests.merged" 2>/dev/null || true
n=$(wc -l < "$COLLECT/all_requests.merged")
say "batch $STAGE ($n requests)"
if [ "$n" -gt 0 ]; then
  python scripts/batch_run.py run --in "$COLLECT/all_requests.merged" --state "$COLLECT/state.json" >> "$LOG" 2>&1     || { say "batch failed"; exit 1; }
fi
say "replay $STAGE"
REPLAY_ROOT="$OUT/$STAGE"
[ "$STAGE" = addendum2 ] && REPLAY_ROOT=benchmark_runs/revision/addendum2
run_all "$REPLAY_ROOT" $N_BOOT replay
[ "$STAGE" = addendum2 ] && python -m benchmarks.dars_eval.confirm_display_order --root "$REPLAY_ROOT" >> "$LOG" 2>&1
say "$STAGE DONE"
