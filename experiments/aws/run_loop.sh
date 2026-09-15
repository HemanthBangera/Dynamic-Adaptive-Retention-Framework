#!/usr/bin/env bash
# Run the pre-registered test phase to completion, unattended.
#
# Two kinds of interruption must be told apart:
#   * out of quota  - Tier 1 allows 10,000 gpt-4o-mini requests/day. Hitting that is normal
#                     and expected; we wait for the reset and lose nothing, because stages are
#                     resumable and every completed response is on disk.
#   * a real bug    - three consecutive failures that produce no new .done marker while quota
#                     IS available. Then we stop and say so instead of looping for days.
set -uo pipefail
REPO="$HOME/dars/Dynamic-Adaptive-Retention-Framework"
cd "$REPO" || exit 1
export PATH="$REPO/.venv/bin:$HOME/.local/bin:$PATH"    # the venv supplies python
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
MARKERS="benchmark_runs/revision/test/_stages"
LOG="benchmark_runs/revision/test_phase.log"
count_markers() { ls "$MARKERS"/*.done 2>/dev/null | wc -l; }
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

# Requests left on the daily cap: prints a number, or nothing if it cannot tell.
rpd_remaining() {
  set -a; . "$HOME/dars/.env" 2>/dev/null; set +a
  curl -s -D - -o /dev/null --max-time 30 \
    -H "Authorization: Bearer ${OPEN_AI_API_KEY:-}" -H "Content-Type: application/json" \
    -d '{"model":"gpt-4o-mini-2024-07-18","messages":[{"role":"user","content":"ping"}],"max_tokens":1}' \
    https://api.openai.com/v1/chat/completions 2>/dev/null \
    | grep -i '^x-ratelimit-remaining-requests:' | awk '{print $2}' | tr -d '\r'
}

barren=0
while true; do
  before=$(count_markers)
  if bash experiments/run_test_phase.sh >> "$LOG" 2>&1; then
    say "TEST PHASE COMPLETE"
    break
  fi
  after=$(count_markers)
  if [ "$after" -gt "$before" ]; then
    barren=0                                   # progress was made; that was a pause, not a bug
  else
    rem=$(rpd_remaining)
    if [ "${rem:-x}" = "0" ]; then
      # Out of quota for today. Idling an r7i.2xlarge until the reset costs ~$10/day for
      # nothing, so shut down instead: the dars-poll schedule boots us again later, systemd
      # restarts this loop, and the stage markers mean we pick up exactly where we stopped.
      say "daily request cap reached; stopping the instance until quota returns"
      bash "$HOME/dars/aws_scripts/sync_status.sh" >/dev/null 2>&1 || true
      TOKEN=$(curl -sS -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' 2>/dev/null)
      IID=$(curl -sS -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null)
      aws ec2 stop-instances --instance-ids "$IID" --region us-east-1 >/dev/null 2>&1         && say "stop requested (quota pause)" || say "stop request failed; will retry after a wait"
      sleep 900
      continue
    fi
    barren=$((barren + 1))
    say "failure with quota available and no new marker ($barren/3)"
  fi
  if [ "$barren" -ge 3 ]; then
    # A real bug, not a rate limit. Record it, publish it, and shut down rather than bill
    # while idle; the EBS volume keeps all state for debugging on the next boot.
    say "FAILED: three consecutive failures with quota available"
    bash "$HOME/dars/aws_scripts/sync_status.sh" >/dev/null 2>&1 || true
    TOKEN=$(curl -sS -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' 2>/dev/null)
    IID=$(curl -sS -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null)
    aws ec2 stop-instances --instance-ids "$IID" --region us-east-1 >/dev/null 2>&1 || true
    exit 1
  fi
  sleep 900
done

say "running the pre-registered analysis"
python -m benchmarks.dars_eval.make_tables --layout test --out benchmark_runs/revision/_tables_test >> "$LOG" 2>&1 || true
python -m benchmarks.dars_eval.make_figures --layout test >> "$LOG" 2>&1 || true
bash "$HOME/dars/aws_scripts/sync_status.sh" final || true
# Stop the instance from inside, so an idle box does not bill while nobody is watching.
TOKEN=$(curl -sS -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')
IID=$(curl -sS -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 stop-instances --instance-ids "$IID" --region us-east-1 >/dev/null 2>&1 && say "instance stop requested"
