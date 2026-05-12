"""
Hyperparameter grid for MemoryAgentBench runs (subprocess per configuration).

Example:
  python -m benchmarks.memory_agent_bench.grid_search \\
    --split Accurate_Retrieval --source eventqa_65536 --max-samples 5 \\
    --output-root ./benchmark_runs/grid_eventqa
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Grid search MemoryAgentBench × DARS")
    p.add_argument("--split", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--max-samples", type=int, default=5)
    p.add_argument("--chunk-size", type=int, default=4096)
    p.add_argument("--output-root", default="./benchmark_runs/grid")
    p.add_argument("--fetch-k", default="15,25,35", help="Comma-separated fetch_k values")
    p.add_argument("--top-n", default="3,5,7", help="Comma-separated top_n values")
    p.add_argument("--overlap", default="0,128,256", help="Comma-separated chunk_overlap_tokens")
    p.add_argument("--no-narrative-grid", action="store_true", help="Add --no-narrative to each cell")
    p.add_argument("--hf-revision", default="main")
    args = p.parse_args(argv)

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    fks = [int(x.strip()) for x in args.fetch_k.split(",") if x.strip()]
    tns = [int(x.strip()) for x in args.top_n.split(",") if x.strip()]
    ovs = [int(x.strip()) for x in args.overlap.split(",") if x.strip()]

    exe = sys.executable
    project_root = Path(__file__).resolve().parents[2]
    index: list[dict] = []

    for fk, tn, ov in itertools.product(fks, tns, ovs):
        label = f"fk{fk}_tn{tn}_ov{ov}"
        out = root / label
        cmd = [
            exe,
            "-m",
            "benchmarks.memory_agent_bench",
            "run",
            "--split",
            args.split,
            "--source",
            args.source,
            "--max-samples",
            str(args.max_samples),
            "--chunk-size",
            str(args.chunk_size),
            "--fetch-k",
            str(fk),
            "--top-n",
            str(tn),
            "--chunk-overlap-tokens",
            str(ov),
            "--output-dir",
            str(out),
            "--hf-revision",
            args.hf_revision,
            "--no-failure-detail",
        ]
        if args.no_narrative_grid:
            cmd.append("--no-narrative")
        subprocess.run(cmd, check=True, cwd=project_root)
        summary_path = out / "metrics_summary.json"
        mean_em = None
        if summary_path.exists():
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            em = data.get("exact_match") or data.get("eventqa_recall")
            if isinstance(em, dict):
                mean_em = em.get("mean")
        index.append({"label": label, "fetch_k": fk, "top_n": tn, "overlap": ov, "mean_em": mean_em})

    (root / "grid_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Wrote {root / 'grid_index.json'}")


if __name__ == "__main__":
    main()
