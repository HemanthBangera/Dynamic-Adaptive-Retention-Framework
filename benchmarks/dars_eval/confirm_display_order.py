"""
Addendum 2: confirmatory test of display order under conflicting memories (FactConsolidation sh_64k, sh_262k).

    python -m benchmarks.dars_eval.confirm_display_order --root benchmark_runs/revision/addendum2
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from benchmarks.dars_eval.provenance import collect_provenance
from benchmarks.dars_eval.stats import holm, paired_bootstrap_diff

ADDENDUM = Path("experiments/preregistration_addendum_2.md")
SOURCES = ("factconsolidation_sh_64k", "factconsolidation_sh_262k")
FAMILY = (  # id, source, (method, order) a, (method, order) b
    ("D1a", SOURCES[0], ("bm25_dars_wrrf_k50_b0.5", "best_last"), ("bm25", "best_first")),
    ("D1b", SOURCES[1], ("bm25_dars_wrrf_k50_b0.5", "best_last"), ("bm25", "best_first")),
    ("D2a", SOURCES[0], ("dars_rrf_k50", "best_last"), ("dars_rrf_k50", "best_first")),
    ("D2b", SOURCES[1], ("dars_rrf_k50", "best_last"), ("dars_rrf_k50", "best_first")),
)


def em_by_question(root: Path, source: str, method: str, order: str) -> Dict[int, float]:
    out = {}
    for line in (root / source / f"order_{order}" / "per_question.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["method"] == method and r.get("reader"):
            out[r["question"]] = float(r["reader"]["metrics"].get("substring_exact_match", 0.0))
    return out


def analyse(root: Path, n_boot: int = 10_000) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for cid, src, a, b in FAMILY:
        ea, eb = em_by_question(root, src, *a), em_by_question(root, src, *b)
        keys = sorted(set(ea) & set(eb))
        r = paired_bootstrap_diff([ea[k] for k in keys], [eb[k] for k in keys], None, n_boot=n_boot)
        rows.append({"id": cid, "source": src, "a": a, "b": b, "mean_a": float(np.mean([ea[k] for k in keys])),
                     "mean_b": float(np.mean([eb[k] for k in keys])), **r})
    for r, p in zip(rows, holm([r["p_value"] for r in rows])):
        r["holm_p"] = p
        r["result"] = ("confirmed" if p < 0.05 and r["diff"] > 0 else
                       "significant, opposite direction" if p < 0.05 else "not confirmed")
    return {"family": rows}


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Addendum 2 primary family")
    p.add_argument("--root", default="benchmark_runs/revision/addendum2")
    args = p.parse_args(argv)
    text = ADDENDUM.read_text(encoding="utf-8")
    expected = Path(str(ADDENDUM) + ".sha256").read_text(encoding="utf-8").split()[0]
    if "Status: FROZEN" not in text or hashlib.sha256(ADDENDUM.read_bytes()).hexdigest() != expected:
        raise SystemExit("addendum 2 is not frozen or does not match its recorded hash; refusing to run")
    report = analyse(Path(args.root))
    report["addendum_sha256"] = expected
    report["provenance"] = collect_provenance()
    out = Path(args.root) / "confirm_display_order.json"
    out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    for r in report["family"]:
        print(f"{r['id']} {r['source']:28s} {r['a'][0]}|{r['a'][1]} {r['mean_a']:.3f} vs {r['b'][0]}|{r['b'][1]} {r['mean_b']:.3f} "
              f"diff {r['diff']:+.3f} [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] p={r['p_value']:.4g} holm={r['holm_p']:.4g} {r['result']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
