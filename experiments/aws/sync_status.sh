#!/usr/bin/env bash
# Publish progress and artifacts to S3 so they can be read without SSH and without the laptop.
set -uo pipefail
BUCKET="${BUCKET:-dars-revision-445527450587}"
REPO="$HOME/dars/Dynamic-Adaptive-Retention-Framework"
cd "$REPO" || exit 0
export PATH="$HOME/.local/bin:$PATH"
MARKERS="benchmark_runs/revision/test/_stages"
LOG="benchmark_runs/revision/test_phase.log"
TOTAL=23
done_n=$(ls "$MARKERS"/*.done 2>/dev/null | wc -l)
last=$(ls -t "$MARKERS"/*.done 2>/dev/null | head -1 | xargs -r basename)
running=$(grep -E '^\[run \]' "$LOG" 2>/dev/null | tail -1)
state=$( [ "${1:-}" = "final" ] && echo COMPLETE || echo RUNNING )
grep -q '^FAILED:' "$LOG" 2>/dev/null && state=FAILED
calls=$(.venv/bin/python - <<'PY' 2>/dev/null || echo "?"
import json, glob, os
t = 0
for p in glob.glob(os.path.join("benchmark_runs","revision","test","**","manifest.json"), recursive=True):
    try:
        u = json.load(open(p, encoding="utf-8")).get("llm_usage") or {}
    except Exception:
        continue
    t += u["api_calls"] if isinstance(u, dict) and "api_calls" in u else sum(
        v.get("api_calls", 0) for v in u.values() if isinstance(v, dict))
print(t)
PY
)
{
  echo "state:        $state"
  echo "updated:      $(date -u +%FT%TZ)"
  echo "stages done:  $done_n / $TOTAL"
  echo "last marker:  ${last:-none}"
  echo "now running:  ${running:-unknown}"
  echo "api calls:    $calls  (test phase, cumulative)"
  echo "instance:     $(hostname) $(uptime -p)"
  echo
  echo "--- last 25 log lines ---"
  tail -25 "$LOG" 2>/dev/null
} > /tmp/status.txt
aws s3 cp /tmp/status.txt "s3://$BUCKET/status.txt" --only-show-errors
aws s3 sync benchmark_runs/revision/test "s3://$BUCKET/artifacts/test" --only-show-errors --exclude '*.tmp'
