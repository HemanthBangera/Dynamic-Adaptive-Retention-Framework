"""
Pre-registered primary analysis of the DARS revision (experiments/preregistration.md §6).

The script:
- reads the test-phase artifacts written by ``experiments/run_test_phase.sh``
  (``benchmark_runs/revision/test``);
- computes the 23 primary comparisons of H1–H5;
- adjusts them together by Holm–Bonferroni;
- computes H6 (the judge's Cohen's κ with a clustered bootstrap CI).

Outputs are ``<out>/primary.json`` and ``<out>/primary.md``.

``--layout dev`` runs the identical analysis on the development artifacts. That run
validates this script before the test phase; its family is smaller, because the dev
layout has one ALFWorld split and no EventQA-full reader run.

Usage
-----
python -m benchmarks.dars_eval.make_tables --layout test
python -m benchmarks.dars_eval.make_tables --layout dev --allow-missing
"""

from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from benchmarks.dars_eval.provenance import PROJECT_ROOT, collect_provenance
from benchmarks.dars_eval.stats import (
    _groups,
    cohen_kappa,
    holm,
    paired_bootstrap_diff,
    paired_bootstrap_noninferiority,
)

RUNS = PROJECT_ROOT / "benchmark_runs" / "revision"
MARGIN = 0.02
ALPHA = 0.05
H1_METHOD, BASELINE, H2_METHOD = "dars_rrf_k15", "similarity", "dars_rrf_k50"
H1_FETCH_K = 15
H1_RECALL_SOURCES = ("ruler_qa1_197K", "ruler_qa2_421K", "longmemeval_s",
                     "factconsolidation_sh_32k", "factconsolidation_mh_32k")
H1_EM_SOURCES = ("eventqa_65536", "eventqa_full", "ruler_qa1_197K", "ruler_qa2_421K")
H2_SOURCES = ("factconsolidation_sh_32k", "factconsolidation_mh_32k")
JUDGE_KEY = "gpt-4.1-nano-2025-04-14|layer_b|s0"


class Missing(Exception):
    """An input artifact of the layout does not exist."""


def layout(name: str) -> Dict[str, Any]:
    if name == "test":
        t = RUNS / "test"
        return {
            "split": "test",
            "e1": {s: t / "E1" / s for s in sorted(set(H1_RECALL_SOURCES) | set(H1_EM_SOURCES))},
            "h2": {s: {"dars": t / "E2" / f"{s}_b256_l0.001", "similarity": t / "E2" / f"{s}_b256_l0.001"}
                   for s in H2_SOURCES},
            "e9": t / "E9" / "evaluate_test.json",
            "e10": {sp: t / "E10" / f"evaluate_{sp}.json" for sp in ("test_in", "test_out")},
            "lme_evict": {p: t / "E2" / "lme_budget50" / p for p in ("dars", "lru", "fifo")},
            "e5": t / "E5",
        }
    if name == "dev":
        e1 = {s: RUNS / "E1" / "dev_reader" / f"{s}_b5120"
              for s in ("ruler_qa1_197K", "ruler_qa2_421K", "longmemeval_s", "eventqa_65536")}
        e1.update({s: RUNS / "E1" / "dev" / f"{s}_dev" for s in ("factconsolidation_sh_32k", "factconsolidation_mh_32k")})
        e1["eventqa_full"] = RUNS / "E1" / "dev_reader" / "eventqa_full_b5120"          # not run on dev
        return {
            "split": "dev",
            "e1": e1,
            "h2": {s: {"dars": RUNS / "E2" / "dev" / "lambda" / f"{s}_b256_l0.001",
                       "similarity": RUNS / "E2" / "dev" / f"{s}_b256"} for s in H2_SOURCES},
            "e9": RUNS / "E9" / "dev_default" / "evaluate_dev.json",
            "e10": {"dev": RUNS / "E10" / "dev_default" / "evaluate_dev.json"},
            "lme_evict": {p: RUNS / "E2" / "dev" / "lme_budget50" / p for p in ("dars", "lru", "fifo")},
            "e5": RUNS / "E5" / "dev",
        }
    raise ValueError(name)


# ── Readers ──────────────────────────────────────────────────────────────────


def _need(path: Path) -> Path:
    if not path.exists():
        raise Missing(str(path))
    return path


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in _need(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(_need(path).read_text(encoding="utf-8"))


def by_question(run_dir: Path, method: str, feedback: Optional[str] = None) -> Dict[Tuple[int, int], Dict[str, Any]]:
    out = {}
    for r in read_jsonl(run_dir / "per_question.jsonl"):
        if r["method"] == method and (feedback is None or r.get("feedback") == feedback):
            out[(r["context"], r["question"])] = r
    return out


def paired(a: Dict, b: Dict, value) -> Tuple[List[float], List[float], List[int]]:
    keys = sorted(k for k in a if k in b and value(a[k]) is not None and value(b[k]) is not None)
    return [value(a[k]) for k in keys], [value(b[k]) for k in keys], [k[0] for k in keys]


def recall_at_5120(rec: Dict[str, Any]) -> Optional[float]:
    return rec.get("budgets", {}).get("5120", {}).get("group_recall")


def by_construction_share(a: Dict, b: Dict, budget: str = "5120") -> float:
    """Share of questions whose shown set at ``budget`` must equal similarity's.

    The DARS method reranks only its ``fetch_k`` nearest memories and fills further slots in
    similarity order, so when both methods show at least ``fetch_k`` memories the shown sets are
    identical and any recall difference is exactly zero (fourth audit, deviations log).
    """
    keys = [k for k in a if k in b and budget in a[k].get("budgets", {}) and budget in b[k].get("budgets", {})]
    if not keys:
        return float("nan")
    hits = 0
    for k in keys:
        na, nb = a[k]["budgets"][budget]["n_units"], b[k]["budgets"][budget]["n_units"]
        same = set(a[k]["ranked"][:na]) == set(b[k]["ranked"][:nb])
        hits += int(same and min(na, nb) >= H1_FETCH_K)
    return hits / len(keys)


def mean_em_at_5120(rec: Dict[str, Any]) -> Optional[float]:
    vals = [r["metrics"].get("substring_exact_match", 0.0) for r in rec.get("reader", []) if r.get("budget") == 5120]
    return float(np.mean(vals)) if vals else None


# ── Primary comparisons ─────────────────────────────────────────────────────


def primary(lay: Dict[str, Any], n_boot: int, allow_missing: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def add(hyp: str, comparison: str, fn, direction: str) -> None:
        try:
            res = fn()
        except Missing as exc:
            if not allow_missing:
                raise
            rows.append({"hypothesis": hyp, "comparison": comparison, "missing": str(exc)})
            return
        rows.append({"hypothesis": hyp, "comparison": comparison, "direction": direction, **res})

    # H1 — non-inferiority of DARS-best to similarity (one-sided p doubled onto the two-sided scale)
    def h1(src: str, value, label: str):
        def fn():
            a = by_question(lay["e1"][src], H1_METHOD)
            b = by_question(lay["e1"][src], BASELINE)
            va, vb, cl = paired(a, b, value)
            if not va:
                raise Missing(f"{lay['e1'][src]}: no paired {label} values")
            r = paired_bootstrap_noninferiority(va, vb, MARGIN, cl, n_boot=n_boot)
            out = {"effect": r["diff"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "n": r["n"],
                   "p_one_sided": r["p_one_sided"], "p": min(1.0, 2 * r["p_one_sided"]), "p_resolution": 1.0 / n_boot}
            if value is recall_at_5120:
                out["by_construction_share"] = by_construction_share(a, b)
            return out
        return fn

    for src in H1_RECALL_SOURCES:
        add("H1", f"{src}: evidence recall@5120, {H1_METHOD} − similarity (margin {MARGIN})",
            h1(src, recall_at_5120, "recall"), "non-inferior")
    for src in H1_EM_SOURCES:
        add("H1", f"{src}: reader EM@5120 (mean of seeds), {H1_METHOD} − similarity (margin {MARGIN})",
            h1(src, mean_em_at_5120, "EM"), "non-inferior")

    # H2 — rank precedence, DARS-best stream (no feedback) − similarity
    for src in H2_SOURCES:
        def fn(src=src):
            a = by_question(lay["h2"][src]["dars"], H2_METHOD, "none")
            b = by_question(lay["h2"][src]["similarity"], BASELINE, "none")
            va, vb, cl = paired(a, b, lambda r: r.get("precedence_rank"))
            r = paired_bootstrap_diff(va, vb, cl, n_boot=n_boot)
            return {"effect": r["diff"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "n": r["n"], "p": r["p_value"],
                    "p_resolution": 1.0 / n_boot}
        add("H2", f"{src}: rank precedence, {H2_METHOD} (λ 0.001) − similarity", fn, "greater")

    # H3, H5 (ALFWorld) from the evaluate reports
    for split, path in lay["e10"].items():
        def h3(path=path):
            rep_ = read_json(path)["h3"]
            d = rep_["paired_vs_similarity"]["selected"]
            excluded = rep_["tasks"] - rep_["reachable_with_alternatives"]
            return {"effect": d["diff"], "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"], "n": d["n"], "p": d["p_value"],
                    "p_resolution": 1e-4,
                    "note": f"{excluded} of {rep_['tasks']} tasks excluded (no alternative location memory)" if excluded else ""}
        add("H3", f"ALFWorld {split}: location MRR, selected − similarity", h3, "greater")
    # H4 (MSC), DeLong
    for comp in ("recency_lru", "frequency_lfu"):
        def h4(comp=comp):
            rep_ = read_json(lay["e9"])
            d = rep_["h4"]["paired"][comp]
            se = abs(d["diff"] / d["z"]) if d.get("z") else 0.0
            return {"effect": d["diff"], "ci_lo": d["diff"] - 1.96 * se, "ci_hi": d["diff"] + 1.96 * se,
                    "auc_selected": d["auc_a"], "auc_comparator": d["auc_b"], "p": d["p_value"],
                    "n": rep_["facts"], "p_resolution": 1e-15,
                    "note": f"{rep_['facts']} facts in {rep_['dialogues']} dialogues; DeLong treats facts as independent"}
        add("H4", f"MSC: AUROC, selected − {comp}", h4, "greater")
    # H5 — harmful deletion, DARS − comparator (lower is better)
    for comp in ("recency_lru", "fifo"):
        def h5_lme(comp=comp):
            a = by_question(lay["lme_evict"]["dars"], "dars_blend_a0.5", "oracle")
            b = by_question(lay["lme_evict"]["lru" if comp == "recency_lru" else "fifo"], "dars_blend_a0.5", "oracle")
            va, vb, cl = paired(a, b, lambda r: r.get("evidence_groups_evicted"))
            r = paired_bootstrap_diff(va, vb, cl, n_boot=n_boot)
            return {"effect": r["diff"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "n": r["n"], "p": r["p_value"],
                    "p_resolution": 1.0 / n_boot}
        add("H5", f"LongMemEval: harmful deletion, DARS − {comp}", h5_lme, "less")

        def h5_msc(comp=comp):
            d = read_json(lay["e9"])["h5"]["paired"][comp]
            return {"effect": d["diff"], "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"], "n": d["n"], "p": d["p_value"],
                    "p_resolution": 1e-4}
        add("H5", f"MSC: harmful deletion, selected − {comp}", h5_msc, "less")

        for split, path in lay["e10"].items():
            def h5_alf(comp=comp, path=path):
                d = read_json(path)["h5"]["paired_vs_selected"][comp]
                return {"effect": d["diff"], "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"], "n": d["n"], "p": d["p_value"],
                        "p_resolution": 1e-4}
            add("H5", f"ALFWorld {split}: harmful deletion, selected − {comp}", h5_alf, "less")
    return rows


def adjust(rows: List[Dict[str, Any]]) -> None:
    """Holm–Bonferroni over every available primary comparison; direction-aware decisions."""
    live = [r for r in rows if "p" in r]
    for r, p_adj in zip(live, holm([r["p"] for r in live])):
        r["p_holm"] = p_adj
        if r["direction"] == "non-inferior":
            r["decision"] = "non-inferior" if p_adj < ALPHA else "not shown"
        else:
            favourable = r["effect"] > 0 if r["direction"] == "greater" else r["effect"] < 0
            if p_adj < ALPHA:
                r["decision"] = "supported" if favourable else "significant, opposite direction"
            else:
                r["decision"] = "not significant"


# ── H6 ───────────────────────────────────────────────────────────────────────


def h6(e5_dir: Path, n_boot: int, seed: int = 0) -> Dict[str, Any]:
    items = read_jsonl(e5_dir / "items.jsonl")
    verdicts = np.array([it["judge"][JUDGE_KEY] for it in items])
    keep = verdicts != "NEUTRAL"
    pred = verdicts[keep] == "YES"
    ref = np.array([it["reference"] for it in items])[keep]
    clusters = np.array([f"{it['source']}:{it['context']}" for it in items])[keep]
    groups = _groups(clusters)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        picks = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[g][rng.integers(0, len(groups[g]), len(groups[g]))] for g in picks])
        boots.append(cohen_kappa(pred[idx], ref[idx]))
    kappa = cohen_kappa(pred, ref)
    return {"kappa": kappa, "ci_lo": float(np.quantile(boots, 0.025)), "ci_hi": float(np.quantile(boots, 0.975)),
            "n": int(keep.sum()), "neutral_rate": float(1 - keep.mean()), "threshold": 0.60,
            "decision": "pass" if kappa >= 0.60 else "fail"}


# ── Output ───────────────────────────────────────────────────────────────────


def format_p(p: float, resolution: Optional[float]) -> str:
    """A p-value that rounds to zero is reported as below the test's resolution, never as 0."""
    if resolution and p < resolution:
        return f"< {resolution:.0e}"
    return f"{p:.3g}"


def to_markdown(rows: List[Dict[str, Any]], h6_res: Optional[Dict[str, Any]], family: int) -> str:
    lines = [f"Primary comparisons (Holm–Bonferroni over {family} comparisons, α = {ALPHA}).", "",
             "| ID | Comparison | Effect [95% CI] | n | p | Holm p | Result | Note |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "missing" in r:
            lines.append(f"| {r['hypothesis']} | {r['comparison']} | — | — | — | — | missing |")
            continue
        n = r.get("n", "")
        res = r.get("p_resolution")
        note = r.get("note", "")
        share = r.get("by_construction_share")
        if share is not None and share == share:
            note = (f"shown set identical to similarity by construction for {share:.0%} of questions"
                    + ("; the difference is zero by design" if share == 1.0 else ""))
        lines.append(f"| {r['hypothesis']} | {r['comparison']} | {r['effect']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] "
                     f"| {n} | {format_p(r['p'], res)} | {format_p(r['p_holm'], res * family if res else None)} | {r['decision']} | {note} |")
    if h6_res:
        lines += ["", f"H6 — judge κ = {h6_res['kappa']:.3f} [{h6_res['ci_lo']:.3f}, {h6_res['ci_hi']:.3f}] "
                      f"(n = {h6_res['n']}, NEUTRAL {h6_res['neutral_rate']:.1%}; threshold 0.60): {h6_res['decision']}"]
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Pre-registered primary analysis (H1–H6)")
    p.add_argument("--layout", choices=("test", "dev"), default="test")
    p.add_argument("--out", default=None)
    p.add_argument("--allow-missing", action="store_true", help="skip comparisons whose inputs are absent (dev only)")
    p.add_argument("--n-boot", type=int, default=10_000)
    args = p.parse_args(argv)
    if args.layout == "test" and args.allow_missing:
        raise SystemExit("--allow-missing is only for the dev layout")

    lay = layout(args.layout)
    out = Path(args.out) if args.out else RUNS / args.layout / "tables" if args.layout == "test" \
        else RUNS / "_tables_dev"
    out.mkdir(parents=True, exist_ok=True)
    rows = primary(lay, args.n_boot, args.allow_missing)
    adjust(rows)
    try:
        h6_res = h6(lay["e5"], min(args.n_boot, 2000))
    except Missing:
        if not args.allow_missing:
            raise
        h6_res = None
    family = sum(1 for r in rows if "p" in r)
    report = {"layout": args.layout, "split": lay["split"], "margin": MARGIN, "alpha": ALPHA, "family_size": family,
              "h1_method": H1_METHOD, "h2_method": H2_METHOD, "primary": rows, "h6": h6_res,
              "inputs": json.loads(json.dumps(lay, default=str)), "provenance": collect_provenance()}
    (out / "primary.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    md = to_markdown(rows, h6_res, family)
    (out / "primary.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
