#!/usr/bin/env bash
# Run one test-phase stage through the Batch API instead of the rate-capped synchronous API.
#
#   1. collect - run the stage with DARS_LLM_COLLECT and a scratch --out, so the reader
#      requests are written out and no synchronous reader call is made. Auxiliary
#      (gpt-4.1-nano) calls still run live: that model has no daily cap.
#   2. batch   - submit the collected requests, wait, and write the answers into the cache.
#   3. replay  - run the real stage through experiments/run_test_phase.sh, which now finds
#      every reader response in the cache and writes its .done marker as usual.
#
# The request bodies are identical either way, so this changes only how the answers are
# delivered. Usage: bash experiments/aws/batch_stage.sh <stage>
set -uo pipefail
STAGE="${1:?usage: batch_stage.sh <stage>}"
REPO="$HOME/dars/Dynamic-Adaptive-Retention-Framework"
cd "$REPO" || exit 1
export PATH="$REPO/.venv/bin:$HOME/.local/bin:$PATH"
export DARS_EMBED_CACHE=benchmark_runs/_emb_cache/minilm.sqlite
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export DARS_LLM_COLLECT_MODEL=gpt-4o-mini-2024-07-18

COLLECT_DIR="$HOME/collect"
SCRATCH="/tmp/collect_$STAGE"
JSONL="$COLLECT_DIR/$STAGE.jsonl"
OUT=benchmark_runs/revision/test
mkdir -p "$COLLECT_DIR"
rm -rf "$SCRATCH"; rm -f "$JSONL"

E1_METHODS=similarity,bm25,recency,random,dars_rrf_k15,dars_rrf_k50,dars_wrrf_k50_b0.5,dars_blend_a0.8,no_memory
E1_READER=similarity,bm25,recency,dars_rrf_k15,no_memory

# The collect pass only needs the request bodies, so bootstrap resamples are pointless there.
collect() {
  case "$STAGE" in
    e1_eventqa65k)
      python -m benchmarks.dars_eval.run_static --source eventqa_65536 --split test --out "$SCRATCH" \
        --methods "$E1_METHODS,full_context" --reader "$E1_READER,full_context" \
        --reader-budget 5120 --reader-seeds 0 1 2 --full-context-seed0 --n-boot 10 ;;
    e1_eventqa_full)
      python -m benchmarks.dars_eval.run_static --source eventqa_full --split test --out "$SCRATCH" \
        --methods "$E1_METHODS" --reader "$E1_READER" \
        --reader-budget 5120 --reader-seeds 0 1 2 --n-boot 10 ;;
    e11_detective)
      python -m benchmarks.dars_eval.run_static --source detective_qa --split test --out "$SCRATCH" \
        --methods similarity,bm25,dars_rrf_k15,no_memory --reader similarity,bm25,dars_rrf_k15,no_memory \
        --reader-budget 5120 --reader-seeds 0 1 2 --budgets 5120 --n-boot 10 ;;
    e11_icl)
      python -m benchmarks.dars_eval.run_static --source icl_banking77_5900shot_balance --split test \
        --out "$SCRATCH" --methods similarity,bm25,dars_rrf_k15,no_memory \
        --reader similarity,bm25,dars_rrf_k15,no_memory \
        --reader-budget 5120 --reader-seeds 0 1 2 --budgets 5120 --n-boot 10 ;;
    e5)
      python -m benchmarks.dars_eval.run_judge --out "$SCRATCH" --methods dars_rrf_k15,similarity --seeds 0 \
        --inputs "$OUT/E1/ruler_qa1_197K" "$OUT/E1/ruler_qa2_421K" "$OUT/E1/longmemeval_s" \
        "$OUT/E1/factconsolidation_sh_32k" "$OUT/E1/factconsolidation_mh_32k" --n-boot 10 ;;
    e6)
      python -m benchmarks.dars_eval.run_layerc --out "$SCRATCH" --split test --n-boot 10 ;;
    e7)
      python -m benchmarks.dars_eval.run_layera --out "$SCRATCH" --split test --method dars_rrf_k15 --n-boot 10 ;;
    e8)
      python -m benchmarks.dars_eval.run_efficiency --out "$SCRATCH" --split test \
        --methods similarity,bm25,dars_rrf_k15 --reader-methods similarity,dars_rrf_k15 --n-boot 10 ;;
    *) echo "no collect recipe for stage '$STAGE'" >&2; return 2 ;;
  esac
}

echo "=== [1/3] collect $STAGE  $(date -u +%FT%TZ)"
DARS_LLM_COLLECT="$JSONL" collect
rc=$?
[ $rc -ge 2 ] && exit $rc
n=$( [ -f "$JSONL" ] && wc -l < "$JSONL" || echo 0 )
echo "collected $n request(s) for $STAGE"
# Collect-only lets every stage be collected first, so one scheduler can keep the
# enqueued-token pipe full across all of them instead of draining it per stage.
if [ "${DARS_COLLECT_ONLY:-0}" = "1" ]; then rm -rf "$SCRATCH"; exit 0; fi

if [ "$n" -gt 0 ]; then
  echo "=== [2/3] batch $STAGE  $(date -u +%FT%TZ)"
  python scripts/batch_run.py run --in "$JSONL" --interval 15 --timeout 21600 || exit 1
fi

echo "=== [3/3] replay $STAGE for real  $(date -u +%FT%TZ)"
bash experiments/run_test_phase.sh "$STAGE"
rc=$?
rm -rf "$SCRATCH"
echo "=== $STAGE finished rc=$rc  $(date -u +%FT%TZ)"
exit $rc
