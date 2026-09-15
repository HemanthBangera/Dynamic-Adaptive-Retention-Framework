#!/usr/bin/env bash
# Replay each batched stage as soon as all of its responses are in the cache.
#
# The batch scheduler fills the cache; this turns a full cache into real artifacts by running
# the stage through experiments/run_test_phase.sh, which writes the usual .done marker. A stage
# is only started when every one of its collected requests has an answer, so the replay makes no
# API calls and cannot be half-finished by a missing response.
set -uo pipefail
REPO="$HOME/dars/Dynamic-Adaptive-Retention-Framework"
cd "$REPO" || exit 1
export PATH="$REPO/.venv/bin:$HOME/.local/bin:$PATH"
export DARS_EMBED_CACHE=benchmark_runs/_emb_cache/minilm.sqlite
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}

STAGES="e1_eventqa65k e1_eventqa_full e5 e6 e7 e8 e11_detective e11_icl"
COLLECT="$HOME/collect"
MARKERS="benchmark_runs/revision/test/_stages"
LOG="benchmark_runs/revision/replay.log"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

missing_count() {   # how many of a stage's collected requests still have no cached answer
  python - "$1" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from scripts.batch_run import cache_path
cache = Path("benchmark_runs/_llm_cache")
ids = [json.loads(l)["custom_id"] for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
print(sum(1 for k in ids if not cache_path(cache, k).is_file()))
PY
}

while true; do
  pending=0
  for s in $STAGES; do
    [ -f "$MARKERS/$s.done" ] && continue
    pending=$((pending + 1))
    jsonl="$COLLECT/$s.jsonl"
    [ -f "$jsonl" ] || continue
    missing=$(missing_count "$jsonl" 2>/dev/null)
    [ "${missing:-1}" = "0" ] || continue
    say "replaying $s - all responses cached"
    if bash experiments/run_test_phase.sh "$s" >> benchmark_runs/revision/test_phase.log 2>&1; then
      say "$s DONE"
    else
      say "$s FAILED during replay - see test_phase.log"
    fi
  done
  [ "$pending" = "0" ] && { say "ALL BATCHED STAGES REPLAYED"; break; }
  sleep 120
done

say "running the pre-registered analysis over whatever is complete"
python -m benchmarks.dars_eval.make_tables --layout test --allow-missing \
  --out benchmark_runs/revision/_tables_test >> "$LOG" 2>&1 || true
bash "$HOME/dars/aws_scripts/sync_status.sh" >/dev/null 2>&1 || true
say "replay watcher finished"
