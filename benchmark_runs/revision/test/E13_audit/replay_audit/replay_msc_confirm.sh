#!/usr/bin/env bash
set -uo pipefail
cd ~/replay_audit/dars_archive
export PATH="$PWD/.venv/bin:$PATH" DARS_EMBED_CACHE=benchmark_runs/_emb_cache/minilm.sqlite HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONIOENCODING=utf-8 DARS_LLM_OFFLINE=1 OPEN_AI_API_KEY=offline-replay-audit-placeholder
export HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 ALL_PROXY=http://127.0.0.1:9 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
S=~/replay_audit/out
python -m benchmarks.dars_eval.run_msc run --split all --hf-split validation --label-session 3 --record-writes --out $S/msc_validation_s3 > $S/msc_validation_s3.log 2>&1
python -m benchmarks.dars_eval.confirm_msc confirm --primary --config experiments/addendum_config.json \
  --runs $S/msc_validation_s3 $S/msc_test_s3 --importance benchmark_runs/revision/addendum/importance.jsonl \
  --out $S/addendum_confirm_from_replayed_data > $S/confirm_from_replayed_data.log 2>&1
echo "EXIT $?"
