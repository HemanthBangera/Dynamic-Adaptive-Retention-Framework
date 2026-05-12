#!/usr/bin/env python3
"""
Export CSV files for LaTeX pgfplots (figures 1.7, 1.8) from benchmark_runs artifacts.
Run from repo root (DARS-Mini-Project/) or from this script's directory.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve().parent
    # scripts/ -> diagrams/ -> reports_latex/ -> DARS-Mini-Project/
    return here.parent.parent.parent


def export_metrics_bar(repo: Path, out_csv: Path) -> int:
    """One row per metrics_summary.json under benchmark_runs."""
    rows: list[dict[str, object]] = []
    idx = 0
    for p in sorted(repo.glob("benchmark_runs/**/metrics_summary.json")):
        run_dir = p.parent
        label = run_dir.name
        tex_label = label.replace("_", "-")
        short = tex_label if len(tex_label) <= 24 else (tex_label[:22] + "...")
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        em = data.get("exact_match", {}).get("mean", "")
        f1 = data.get("f1", {}).get("mean", "")
        n = data.get("exact_match", {}).get("n", "")
        manifest = run_dir / "run_manifest.json"
        path_mode = ""
        split = ""
        source = ""
        if manifest.is_file():
            try:
                m = json.loads(manifest.read_text(encoding="utf-8"))
                path_mode = str(m.get("path_mode", ""))
                split = str(m.get("split", ""))
                source = str(m.get("metadata_source", ""))
            except (OSError, json.JSONDecodeError):
                pass
        rows.append(
            {
                "plot_index": idx,
                "label": tex_label[:48],
                "short_label": short,
                "em_mean": em,
                "f1_mean": f1,
                "n": n,
                "path_mode": path_mode,
                "split": split,
                "source": source,
            }
        )
        idx += 1
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # Placeholder so LaTeX always compiles
        rows = [
            {
                "plot_index": 0,
                "label": "no-runs",
                "short_label": "none",
                "em_mean": 0.0,
                "f1_mean": 0.0,
                "n": 0,
                "path_mode": "",
                "split": "",
                "source": "",
            }
        ]
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "plot_index",
                "label",
                "short_label",
                "em_mean",
                "f1_mean",
                "n",
                "path_mode",
                "split",
                "source",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def export_scatter(repo: Path, out_csv: Path, max_rows: int = 120) -> int:
    """Scatter: token_savings_ratio vs f1 from each results.json (capped)."""
    out_rows: list[tuple[float, float, str, str]] = []
    for p in sorted(repo.glob("benchmark_runs/**/results.json")):
        run_dir = p.parent
        try:
            arr = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(arr, list):
            continue
        for row in arr:
            if len(out_rows) >= max_rows:
                break
            try:
                x = float(row.get("token_savings_ratio", 0.0))
                y = float(row.get("f1", 0.0))
            except (TypeError, ValueError):
                continue
            pm = str(row.get("path_mode", ""))
            qid = str(row.get("qa_pair_id", row.get("query_id", "")))
            out_rows.append((x, y, pm, qid))
        if len(out_rows) >= max_rows:
            break
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not out_rows:
        out_rows = [(0.5, 0.5, "n/a", "placeholder")]
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["token_savings_ratio", "f1", "path_mode", "qa_pair_id"])
        for t in out_rows:
            w.writerow(t)
    return len(out_rows)


def main() -> None:
    repo = _repo_root()
    data_dir = Path(__file__).resolve().parent.parent / "data"
    n1 = export_metrics_bar(repo, data_dir / "fig_1_7_metrics.csv")
    n2 = export_scatter(repo, data_dir / "fig_1_8_scatter.csv")
    print(f"Wrote fig_1_7_metrics.csv ({n1} runs)")
    print(f"Wrote fig_1_8_scatter.csv ({n2} points)")


if __name__ == "__main__":
    main()
