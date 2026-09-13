#!/usr/bin/env bash
# Replay audit of the publication archive: unpack it into a fresh directory, build a fresh virtual
# environment, and regenerate a set of results with no API key and no network, from the redacted cache.
set -uo pipefail
ARCH_TGZ=$HOME/archive_build/dars_archive.tgz
R=$HOME/replay_audit
S=$R/out
A=$R/pristine
LOG=$R/replay_audit.log
rm -rf "$R"; mkdir -p "$R" "$S" "$A"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

say "archive sha256 $(sha256sum "$ARCH_TGZ" | cut -c1-64)"
tar xzf "$ARCH_TGZ" -C "$R"
tar xzf "$ARCH_TGZ" -C "$A" --strip-components=1 dars_archive/benchmark_runs/revision dars_archive/experiments
cd "$R/dars_archive" || exit 1
say "unpacked: $(find . -type f | wc -l) files"

say "fresh venv"
~/.local/bin/uv venv --python 3.14.3 .venv >> "$LOG" 2>&1
~/.local/bin/uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch==2.11.0 >> "$LOG" 2>&1
~/.local/bin/uv pip install --python .venv/bin/python -r experiments/requirements-vm.txt >> "$LOG" 2>&1 || { say "venv FAILED"; exit 1; }

# No real key, no network: every LLM answer must come from the archived (redacted) cache.
export PATH="$PWD/.venv/bin:$PATH" DARS_EMBED_CACHE=benchmark_runs/_emb_cache/minilm.sqlite
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONIOENCODING=utf-8 DARS_LLM_OFFLINE=1
export OPEN_AI_API_KEY=offline-replay-audit-placeholder OPENAI_API_KEY=offline-replay-audit-placeholder
export HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 ALL_PROXY=http://127.0.0.1:9 NO_PROXY=
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
unset DARS_ENV_FILE
NB=10000
T=benchmark_runs/revision/test

run() {  # run NAME CMD...
  local name=$1; shift
  say "start $name"
  if "$@" > "$S/$name.log" 2>&1; then say "done  $name"; else say "FAILED $name (see out/$name.log)"; fi
}
e1()   { python -m benchmarks.dars_eval.run_static --source ruler_qa1_197K --split test --out "$S/E1/ruler_qa1_197K" \
           --methods similarity,bm25,recency,random,dars_rrf_k15,dars_rrf_k50,dars_wrrf_k50_b0.5,dars_blend_a0.8,no_memory \
           --reader similarity,bm25,recency,dars_rrf_k15,no_memory --reader-budget 5120 --reader-seeds 0 1 2 \
           --full-context-seed0 --n-boot $NB; }
e11()  { python -m benchmarks.dars_eval.run_static --source detective_qa --split test --out "$S/E11/detective_qa" \
           --methods similarity,bm25,dars_rrf_k15,no_memory --reader similarity,bm25,dars_rrf_k15,no_memory \
           --reader-budget 5120 --reader-seeds 0 1 2 --budgets 5120 --n-boot $NB; }
add2() { for src in factconsolidation_sh_64k factconsolidation_sh_262k; do for order in best_first best_last; do
           python -m benchmarks.dars_eval.run_stream --source $src --out "$S/addendum2/$src/order_$order" \
             --methods similarity,bm25,dars_rrf_k50,bm25_dars_wrrf_k50_b0.5 --feedback none --budget 256 \
             --decay-lambda 0.001 --reader --display-order $order --report-split test --n-boot $NB || return 1
         done; done
         python -m benchmarks.dars_eval.confirm_display_order --root "$S/addendum2"; }
msc()  { python -m benchmarks.dars_eval.run_msc run --split all --hf-split test --label-session 3 --record-writes \
           --out "$S/msc_test_s3"; }
rest() {
  python -m benchmarks.dars_eval.confirm_msc confirm --primary --config experiments/addendum_config.json \
    --runs benchmark_runs/revision/addendum/msc_validation_s3 benchmark_runs/revision/addendum/msc_test_s3 \
    --importance benchmark_runs/revision/addendum/importance.jsonl --out "$S/addendum_confirm" || return 1
  python -m benchmarks.dars_eval.confirm_msc confirm --config experiments/addendum_config.json \
    --runs benchmark_runs/revision/addendum/msc_validation_s4 benchmark_runs/revision/addendum/msc_test_s4 \
    --importance benchmark_runs/revision/addendum/importance.jsonl --out "$S/addendum_confirm" || return 1
  python -m benchmarks.dars_eval.run_judge --out "$S/E5" --methods dars_rrf_k15,similarity --seeds 0 \
    --inputs $T/E1/ruler_qa1_197K $T/E1/ruler_qa2_421K $T/E1/longmemeval_s $T/E1/factconsolidation_sh_32k \
    $T/E1/factconsolidation_mh_32k --n-boot $NB || return 1
  python -m benchmarks.dars_eval.make_tables --layout test --out "$S/tables" || return 1
  python -m benchmarks.dars_eval.make_exploratory_tables || return 1
  python scripts/build_si.py --out "$S/supplementary_information.md"
}

run pytest python -m pytest -q -p no:cacheprovider &
run e1 e1 &
run e11 e11 &
run addendum2 add2 &
run msc_test_s3 msc &
run rest rest &
wait

say "compare"
C="python $HOME/replay_compare.py"
{
  $C "$A/$T/E1/ruler_qa1_197K" "$S/E1/ruler_qa1_197K" --label E1_ruler_qa1
  $C "$A/$T/E11/detective_qa" "$S/E11/detective_qa" --label E11_detective_qa
  $C "$A/benchmark_runs/revision/addendum2" "$S/addendum2" --label addendum2
  $C "$A/benchmark_runs/revision/addendum/msc_test_s3" "$S/msc_test_s3" --label addendum1_msc_test_s3
  $C "$A/benchmark_runs/revision/addendum/confirm" "$S/addendum_confirm" --label addendum1_confirm
  $C "$A/$T/E5" "$S/E5" --label E5_judge
  $C "$A/benchmark_runs/revision/_tables_test" "$S/tables" --only primary.json --ignore inputs --label primary_table
  $C "$A/benchmark_runs/revision/_tables_test" benchmark_runs/revision/_tables_test --only exploratory.json --label exploratory_table
  python - "$A" "$S" <<'PY'
import sys
from pathlib import Path
a, s = Path(sys.argv[1]), Path(sys.argv[2])
norm = lambda p: p.read_bytes().replace(b"\r\n", b"\n")
pairs = [("primary.md", a / "benchmark_runs/revision/_tables_test/primary.md", s / "tables/primary.md"),
         ("exploratory.md", a / "benchmark_runs/revision/_tables_test/exploratory.md",
          Path("benchmark_runs/revision/_tables_test/exploratory.md")),
         ("supplementary_information.md", Path.home() / "si_reference.md", s / "supplementary_information.md")]
for name, x, y in pairs:
    print(f"[text] {name}: {'identical (line endings aside)' if norm(x) == norm(y) else 'DIFFERS'}")
PY
  grep -h "passed\|failed" "$S/pytest.log" | tail -1
  grep -h "api_calls" -r "$S" --include=manifest.json | tr -d ' ,' | sort | uniq -c
} 2>&1 | tee -a "$LOG" > "$R/compare.txt"
say "AUDIT FINISHED"
