#!/usr/bin/env bash
# ============================================================================
#  DARS revision — pre-registered TEST phase (experiments/preregistration.md)
# ============================================================================
#  Runs every held-out evaluation once, with the configurations frozen in §11 of
#  the pre-registration.  Refuses to start unless the pre-registration is marked
#  FROZEN and its SHA-256 matches experiments/preregistration.sha256 (written at
#  freeze).  Every run manifest also records the pre-registration hash.
#
#  Stages are resumable: a finished stage leaves a marker in $OUT/_stages and is
#  skipped next time; LLM responses are cached on disk, so a stage interrupted by
#  the OpenAI request limits (e.g. 10k requests/day on Tier 1) resumes at no cost.
#
#  Usage:  bash experiments/run_test_phase.sh [stage ...]    (default: every stage)
# ============================================================================
set -euo pipefail
set -f                                   # no pathname expansion ("longmemeval_s*" is a source name)
cd "$(dirname "$0")/.."

PREREG=experiments/preregistration.md
SHA_FILE=experiments/preregistration.sha256
OUT=benchmark_runs/revision/test
N_BOOT=10000

# ── Guards ───────────────────────────────────────────────────────────────────
grep -q '^\*\*Status: FROZEN' "$PREREG" || { echo "Pre-registration is not frozen." >&2; exit 1; }
[ -f "$SHA_FILE" ] || { echo "Missing $SHA_FILE (written at freeze)." >&2; exit 1; }
actual=$(python -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$PREREG")
expected=$(cut -d' ' -f1 "$SHA_FILE")
[ "$actual" = "$expected" ] || { echo "Pre-registration changed since freeze ($actual != $expected)." >&2; exit 1; }

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export DARS_EMBED_CACHE=benchmark_runs/_emb_cache/minilm.sqlite
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONIOENCODING=utf-8
mkdir -p "$OUT/_stages" "$OUT/logs"

run_stage() {                            # run_stage NAME COMMAND [ARGS...]
  local name=$1; shift
  if [ -f "$OUT/_stages/$name.done" ]; then echo "[skip] $name"; return 0; fi
  echo "[run ] $name  $(date -u +%FT%TZ)"
  "$@" > "$OUT/logs/$name.log" 2>&1
  date -u +%FT%TZ > "$OUT/_stages/$name.done"
  echo "[done] $name"
}

# ── E1: static retrieval at equal budgets, reader at B = 5120 (H1) ───────────
E1_METHODS=similarity,bm25,recency,random,dars_rrf_k15,dars_rrf_k50,dars_wrrf_k50_b0.5,dars_blend_a0.8,no_memory
E1_READER=similarity,bm25,recency,dars_rrf_k15,no_memory
e1() {                                   # e1 SOURCE FULL_CONTEXT(yes|no)
  local src=$1 full=$2 safe=${1//\*/}
  local methods=$E1_METHODS reader=$E1_READER
  if [ "$full" = yes ]; then methods=$methods,full_context; reader=$reader,full_context; fi
  python -m benchmarks.dars_eval.run_static --source "$src" --split test --out "$OUT/E1/$safe" \
    --methods "$methods" --reader "$reader" --reader-budget 5120 --reader-seeds 0 1 2 \
    --full-context-seed0 --n-boot $N_BOOT
}

# ── E2: multi-session streams ────────────────────────────────────────────────
STREAM_METHODS=similarity,bm25,recency,random,dars_rrf_k50,dars_wrrf_k50_b0.5,dars_blend_a0.5
e2_h2() {                                # FactConsolidation precedence (H2): selected λ and submitted λ
  for src in factconsolidation_sh_32k factconsolidation_mh_32k; do
    python -m benchmarks.dars_eval.run_stream --source $src --out "$OUT/E2/${src}_b256_l0.001" \
      --methods $STREAM_METHODS --feedback none oracle --budget 256 --decay-lambda 0.001 \
      --report-split test --n-boot $N_BOOT
    python -m benchmarks.dars_eval.run_stream --source $src --out "$OUT/E2/${src}_b256_l0.005" \
      --methods dars_rrf_k50,dars_wrrf_k50_b0.5,dars_blend_a0.5 --feedback none oracle --budget 256 \
      --decay-lambda 0.005 --report-split test --n-boot $N_BOOT
  done
}
e2_h5() {                                # LongMemEval eviction at keep 50 % (H5): DARS (default) vs LRU, FIFO
  for ev in dars lru fifo lfu; do
    python -m benchmarks.dars_eval.run_stream --source "longmemeval_s*" --out "$OUT/E2/lme_budget50/$ev" \
      --methods dars_blend_a0.5 --feedback oracle --budget 2048 --memory-budget 0.5 --eviction $ev \
      --report-split test --n-boot $N_BOOT
  done
  for seed in 0 1 2 3 4; do              # random baseline: mean of five seeds (one draw is noisy at n = 86)
    python -m benchmarks.dars_eval.run_stream --source "longmemeval_s*" --out "$OUT/E2/lme_budget50/random_s$seed" \
      --methods dars_blend_a0.5 --feedback oracle --budget 2048 --memory-budget 0.5 --eviction random \
      --seed $seed --report-split test --n-boot $N_BOOT
  done
}
e2_budget_sweep() {                      # LongMemEval eviction at keep 25 % and 75 % (secondary; H5 stays at 50 %)
  for keep in 0.25 0.75; do
    for ev in dars lru fifo lfu; do
      python -m benchmarks.dars_eval.run_stream --source "longmemeval_s*" --out "$OUT/E2/lme_budget${keep#0.}/$ev" \
        --methods dars_blend_a0.5 --feedback oracle --budget 2048 --memory-budget $keep --eviction $ev \
        --report-split test --n-boot $N_BOOT
    done
    for seed in 0 1 2 3 4; do
      python -m benchmarks.dars_eval.run_stream --source "longmemeval_s*" --out "$OUT/E2/lme_budget${keep#0.}/random_s$seed" \
        --methods dars_blend_a0.5 --feedback oracle --budget 2048 --memory-budget $keep --eviction random \
        --seed $seed --report-split test --n-boot $N_BOOT
    done
  done
}
e2_dynamics() {                          # ranking dynamics and per-component influence (R2.2)
  python -m benchmarks.dars_eval.run_stream --source "longmemeval_s*" --out "$OUT/E2/lme_dynamics" \
    --methods $STREAM_METHODS --feedback none oracle --budget 2048 --report-split test --n-boot $N_BOOT
}
e2_noise() {                             # robustness to feedback noise (R2.8)
  python -m benchmarks.dars_eval.run_stream --source "longmemeval_s*" --out "$OUT/E2/lme_feedback_noise" \
    --methods dars_rrf_k50 --feedback none oracle oracle_noisy:0.1 oracle_noisy:0.2 oracle_noisy:0.3 \
    oracle_noisy:0.5 --budget 2048 --report-split test --n-boot $N_BOOT
}
e2_feedback_sources() {                  # end-to-end Layer B loop with the reader: none / oracle / judge / lexical
  python -m benchmarks.dars_eval.run_stream --source "longmemeval_s*" --out "$OUT/E2/lme_feedback_sources" \
    --methods dars_rrf_k50 --feedback none oracle judge lexical --reader --budget 2048 \
    --report-split test --n-boot $N_BOOT
}

# ── E9 MSC (H4, H5) and E10 ALFWorld (H3, H5): frozen configurations ────────
e9() {
  python -m benchmarks.dars_eval.run_msc run --out "$OUT/E9" --split test
  python -m benchmarks.dars_eval.run_msc evaluate --run "$OUT/E9" --split test \
    --h4-lambda 0.05 --h4-weights 0.2 0 0.8 0 --h5-lambda 0.0005 --h5-weights 0.3 0 0.7 0 --n-boot $N_BOOT
  python -m benchmarks.dars_eval.run_msc analyze --run "$OUT/E9" --split test --n-boot $N_BOOT
}
e10() {
  python -m benchmarks.dars_eval.run_alfworld run --out "$OUT/E10" --eval-test
  for split in test_in test_out; do
    python -m benchmarks.dars_eval.run_alfworld evaluate --run "$OUT/E10" --split $split \
      --h3-lambda 0.0005 --h3-mode score_only --h3-weights 0 1 0 0 \
      --h5-lambda 0.0005 --h5-weights 0 0 0 1 --n-boot $N_BOOT
    python -m benchmarks.dars_eval.run_alfworld analyze --run "$OUT/E10" --split $split --n-boot $N_BOOT
  done
}

# ── E3 thresholds on the end-of-stream test snapshots ────────────────────────
e3() {
  python -m benchmarks.dars_eval.thresholds --out "$OUT/E3/default" \
    --snapshot MSC="$OUT/E9/facts.jsonl" --snapshot ALFWorld="$OUT/E10/memories.jsonl"
  python -m benchmarks.dars_eval.thresholds --out "$OUT/E3/msc_h4" --weights 0.2 0 0.8 0 --decay-lambda 0.05 \
    --snapshot MSC="$OUT/E9/facts.jsonl"
  python -m benchmarks.dars_eval.thresholds --out "$OUT/E3/msc_h5" --weights 0.3 0 0.7 0 --decay-lambda 0.0005 \
    --snapshot MSC="$OUT/E9/facts.jsonl"
  local now
  now=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['now'])" "$OUT/E10/manifest.json")
  python -m benchmarks.dars_eval.thresholds --out "$OUT/E3/alfworld_h5" --weights 0 0 0 1 --decay-lambda 0.0005 \
    --now "$now" --snapshot ALFWorld="$OUT/E10/memories.jsonl"
}

# ── Secondary experiments ────────────────────────────────────────────────────
e4() { python -m benchmarks.dars_eval.run_p_variants --out "$OUT/E4" --split test --n-boot $N_BOOT; }
e5() {                                   # judge reliability (H6): H1 method and similarity, reader seed 0
  python -m benchmarks.dars_eval.run_judge --out "$OUT/E5" --methods dars_rrf_k15,similarity --seeds 0 \
    --inputs "$OUT/E1/ruler_qa1_197K" "$OUT/E1/ruler_qa2_421K" "$OUT/E1/longmemeval_s" \
    "$OUT/E1/factconsolidation_sh_32k" "$OUT/E1/factconsolidation_mh_32k" --n-boot $N_BOOT
}
e6() { python -m benchmarks.dars_eval.run_layerc --out "$OUT/E6" --split test --n-boot $N_BOOT; }
e7() { python -m benchmarks.dars_eval.run_layera --out "$OUT/E7" --split test --method dars_rrf_k15 --n-boot $N_BOOT; }
e8() {
  python -m benchmarks.dars_eval.run_efficiency --out "$OUT/E8" --split test \
    --methods similarity,bm25,dars_rrf_k15 --reader-methods similarity,dars_rrf_k15 --n-boot $N_BOOT
}

# ── E11: remaining MemoryAgentBench competencies (secondary, prereg §7) ──────
e11() {                                  # e11 SOURCE
  local src=$1
  python -m benchmarks.dars_eval.run_static --source "$src" --split test --out "$OUT/E11/$src" \
    --methods similarity,bm25,dars_rrf_k15,no_memory --reader similarity,bm25,dars_rrf_k15,no_memory \
    --reader-budget 5120 --reader-seeds 0 1 2 --budgets 5120 --n-boot $N_BOOT
}

# ── Stage order: CPU-only primary analyses first, then reader (LLM) stages ───
declare -A STAGE=(
  [e2_h2]="e2_h2" [e2_h5]="e2_h5" [e2_budget_sweep]="e2_budget_sweep" [e2_dynamics]="e2_dynamics"
  [e2_noise]="e2_noise"
  [e9]="e9" [e10]="e10" [e3]="e3" [e4]="e4"
  [e1_ruler_qa1]="e1 ruler_qa1_197K no" [e1_ruler_qa2]="e1 ruler_qa2_421K no"
  [e1_lme]="e1 longmemeval_s* no" [e1_fc_sh]="e1 factconsolidation_sh_32k yes"
  [e1_fc_mh]="e1 factconsolidation_mh_32k yes" [e1_eventqa65k]="e1 eventqa_65536 yes"
  [e1_eventqa_full]="e1 eventqa_full no"
  [e5]="e5" [e6]="e6" [e7]="e7" [e8]="e8" [e2_feedback_sources]="e2_feedback_sources"
  [e11_detective]="e11 detective_qa" [e11_icl]="e11 icl_banking77_5900shot_balance"
)
ORDER=(e2_h2 e2_h5 e2_budget_sweep e2_dynamics e2_noise e9 e10 e3 e4
       e1_ruler_qa1 e1_ruler_qa2 e1_lme e1_fc_sh e1_fc_mh e1_eventqa65k e1_eventqa_full
       e5 e6 e7 e8 e2_feedback_sources e11_detective e11_icl)
[ $# -gt 0 ] && ORDER=("$@")

{ echo "start $(date -u +%FT%TZ)"; echo "preregistration_sha256 $actual"; git rev-parse HEAD 2>/dev/null || true; } \
  >> "$OUT/_stages/RUNS.log"
for s in "${ORDER[@]}"; do
  [ -n "${STAGE[$s]:-}" ] || { echo "Unknown stage: $s" >&2; exit 1; }
  # shellcheck disable=SC2086
  run_stage "$s" ${STAGE[$s]}
done
echo "Test phase complete: $(date -u +%FT%TZ)"
