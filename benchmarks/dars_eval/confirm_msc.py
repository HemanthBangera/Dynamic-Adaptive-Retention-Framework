"""
Confirmatory MSC study (``experiments/preregistration_addendum.md``).

Two commands:

select   On the MSC *dev* dialogues already used by the pre-registered study, choose the
         write-side DARS configurations (the same 286-vector weight grid and decay rates as E9,
         the same selection rule) and fit the metadata model. Writes ``selection.json``, whose
         values are copied into the addendum before it is frozen.

confirm  On the untouched MSC validation and test dialogues, compute the addendum's primary
         family once, with Holm correction, plus its secondary analyses.

    python -m benchmarks.dars_eval.confirm_msc select --run <dev run> --out <selection.json>
    python -m benchmarks.dars_eval.confirm_msc confirm --config <addendum config.json> \
        --runs <validation run> <test run> --importance <ratings.jsonl> --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from benchmarks.dars_eval.provenance import collect_provenance
from benchmarks.dars_eval.retention_signals import (
    dars_read_score, dars_write_score, generative_agents_score, memorybank_score, metadata_features,
    metadata_score, write_component_matrix, write_components,
)
from benchmarks.dars_eval.run_msc import DEFAULT_WEIGHTS, eviction
from benchmarks.dars_eval.stats import (
    auroc_cluster_bootstrap, auroc_cluster_bootstrap_paired, auroc_delong, auroc_delong_paired,
    cluster_bootstrap_mean, holm, paired_bootstrap_diff,
)

logger = logging.getLogger(__name__)
LABELS = ("lex_0.5", "lex_0.6", "lex_0.7", "embed_0.8")
KEEPS = (0.25, 0.5, 0.75)


def load_rows(run_dirs: Sequence[Path], split: Optional[str] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for d in run_dirs:
        for line in (Path(d) / "facts.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if split in (None, "all") or r["split"] == split:
                    rows.append(r)
    if not rows:
        raise ValueError(f"no facts in {run_dirs} for split {split!r}")
    if "mention_sessions" not in rows[0]:
        raise ValueError("these runs have no write history; use run_msc run --record-writes")
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
#  select (dev only)
# ═══════════════════════════════════════════════════════════════════════════════


def _harmful_fn(rows: Sequence[Dict[str, Any]], y: np.ndarray, keep: float = 0.5):
    tie = np.random.default_rng(0).random(len(rows))         # the same seeded tie-break as run_msc.eviction
    by_d: Dict[Any, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_d[r["dialogue"]].append(i)
    groups = [np.array(ix) for ix in by_d.values() if y[ix].any()]

    def harmful(S: np.ndarray) -> float:
        rates = []
        for ix in groups:
            n_keep = max(1, int(math.ceil(keep * len(ix))))
            kept = set(ix[np.lexsort((tie[ix], -S[ix]))[:n_keep]].tolist())
            need = ix[y[ix]]
            rates.append(sum(1 for i in need if i not in kept) / len(need))
        return float(np.mean(rates))

    return harmful


def select(rows: Sequence[Dict[str, Any]], label: str = "lex_0.5") -> Dict[str, Any]:
    from sklearn.linear_model import LogisticRegression

    from benchmarks.dars_eval.tuning import LAMBDAS, auc_rows, select as pick, simplex_grid

    y = np.array([r["labels"][label] for r in rows])
    grid = np.array(simplex_grid(0.1))
    harmful = _harmful_fn(rows, y)
    h4: Dict[Tuple, float] = {}
    h5: Dict[Tuple, float] = {}
    for lam in LAMBDAS:
        S = np.round(np.clip(grid @ write_component_matrix(rows, lam), 0.0, 1.0), 6)
        for w, a, s in zip(map(tuple, grid), auc_rows(S, y), S):
            h4[(lam, w)] = float(a)
            h5[(lam, w)] = harmful(s)
    best4, val4 = pick(h4)
    best5, val5 = pick(h5, higher_is_better=False)
    model = LogisticRegression().fit(metadata_features(rows), y)
    return {
        "label": label, "facts": len(rows), "dialogues": len({r["dialogue"] for r in rows}),
        "write_h4": {"lambda": best4[0], "weights": list(best4[1]), "dev_auroc": val4},
        "write_h5": {"lambda": best5[0], "weights": list(best5[1]), "dev_harmful_deletion": val5},
        "write_default_dev": {"auroc": h4[(0.005, tuple(DEFAULT_WEIGHTS))],
                              "harmful_deletion": h5[(0.005, tuple(DEFAULT_WEIGHTS))]},
        "write_single_component_dev_auroc": {c: h4[(best4[0], tuple(1.0 if j == i else 0.0 for j in range(4)))]
                                             for i, c in enumerate(("R", "F", "U", "P"))},
        "metadata_model": {"features": ["sessions_since_creation", "mentions"],
                           "estimator": "sklearn LogisticRegression() defaults (lbfgs, L2, C=1.0)",
                           "intercept": float(model.intercept_[0]), "coef": [float(c) for c in model.coef_[0]]},
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  confirm (untouched data, run once)
# ═══════════════════════════════════════════════════════════════════════════════


def all_scores(rows: Sequence[Dict[str, Any]], cfg: Dict[str, Any], importance: Dict[str, float]) -> Dict[str, np.ndarray]:
    return {
        "read_h4": dars_read_score(rows, cfg["read_h4"]["lambda"], cfg["read_h4"]["weights"]),
        "read_h5": dars_read_score(rows, cfg["read_h5"]["lambda"], cfg["read_h5"]["weights"]),
        "read_default": dars_read_score(rows, 0.005, DEFAULT_WEIGHTS),
        "write_h4": dars_write_score(rows, cfg["write_h4"]["lambda"], cfg["write_h4"]["weights"]),
        "write_h5": dars_write_score(rows, cfg["write_h5"]["lambda"], cfg["write_h5"]["weights"]),
        "write_default": dars_write_score(rows, 0.005, DEFAULT_WEIGHTS),
        "utility_read": np.array([r["components"]["U"] for r in rows]),
        "utility_write": np.array([write_components(r, 0.0)["U"] for r in rows]),
        "recency_lru": np.array([r["recency"] for r in rows]),
        "fifo": np.array([r["created_at"] for r in rows]),
        "frequency_lfu": np.array([r["frequency"] for r in rows], dtype=float),
        "mention_count": np.array([r["mentions"] for r in rows], dtype=float),
        "generative_agents": generative_agents_score(rows, importance),
        "memorybank": memorybank_score(rows),
        "metadata_model": metadata_score(rows, cfg["metadata_model"]["intercept"], cfg["metadata_model"]["coef"]),
    }


def compare_auroc(a: np.ndarray, b: np.ndarray, y: np.ndarray, clusters: Sequence, n_boot: int) -> Dict[str, Any]:
    out = auroc_cluster_bootstrap_paired(a, b, y, clusters, n_boot=n_boot)
    out["delong_p"] = auroc_delong_paired(a, b, y)["p_value"]
    return out


def compare_harm(rows, a, b, label, keep, n_boot) -> Dict[str, Any]:
    ea, eb = eviction(rows, a, label, keep), eviction(rows, b, label, keep)
    if ea["dialogues"] != eb["dialogues"]:
        raise RuntimeError("eviction comparison is not paired")
    out = paired_bootstrap_diff(ea["harmful_deletion_rate"], eb["harmful_deletion_rate"], ea["dialogues"], n_boot=n_boot)
    out["mean_a"] = float(np.mean(ea["harmful_deletion_rate"]))
    out["mean_b"] = float(np.mean(eb["harmful_deletion_rate"]))
    return out


# (id, endpoint, a, b, predicted sign of a - b)
FAMILY = (
    ("C1a", "auroc", "read_h4", "recency_lru", +1),
    ("C1b", "auroc", "read_h4", "frequency_lfu", +1),
    ("C2a", "harmful", "read_h5", "recency_lru", -1),
    ("C2b", "harmful", "read_h5", "fifo", -1),
    ("C3", "auroc", "write_h4", "read_h4", +1),
    ("C4", "harmful", "write_h5", "read_h5", -1),
    ("C5", "auroc", "utility_write", "utility_read", +1),
    ("C6a", "auroc", "write_h4", "generative_agents", +1),
    ("C6b", "auroc", "write_h4", "memorybank", +1),
    ("C6c", "harmful", "write_h5", "generative_agents", -1),
    ("C6d", "harmful", "write_h5", "memorybank", -1),
)


def family(rows, scores, label="lex_0.5", keep=0.5, n_boot_auc=5000, n_boot=10_000) -> List[Dict[str, Any]]:
    y = np.array([r["labels"][label] for r in rows])
    clusters = [r["dialogue"] for r in rows]
    results = []
    for cid, endpoint, a, b, sign in FAMILY:
        if endpoint == "auroc":
            res = compare_auroc(scores[a], scores[b], y, clusters, n_boot_auc)
        else:
            res = compare_harm(rows, scores[a], scores[b], label, keep, n_boot)
        results.append({"id": cid, "endpoint": endpoint, "a": a, "b": b, "predicted_sign": sign, **res})
    adjusted = holm([r["p_value"] for r in results])
    for r, p in zip(results, adjusted):
        r["holm_p"] = p
        direction = np.sign(r["diff"])
        if p < 0.05 and direction == r["predicted_sign"]:
            r["result"] = "confirmed"
        elif p < 0.05:
            r["result"] = "significant, opposite direction"
        else:
            r["result"] = "not confirmed"
    return results


def secondary(rows, scores, n_boot_auc=2000, n_boot=5000) -> Dict[str, Any]:
    clusters = [r["dialogue"] for r in rows]
    out: Dict[str, Any] = {}
    for label in LABELS:
        y = np.array([r["labels"][label] for r in rows])
        if y.all() or not y.any():
            continue
        entry: Dict[str, Any] = {"base_rate": float(y.mean()), "auroc": {}, "harmful_deletion": {}}
        for name, s in scores.items():
            entry["auroc"][name] = {"delong": auroc_delong(s, y),
                                    "clustered": auroc_cluster_bootstrap(s, y, clusters, n_boot=n_boot_auc)}
        for keep in KEEPS:
            entry["harmful_deletion"][str(keep)] = {}
            for name, s in scores.items():
                ev = eviction(rows, s, label, keep)
                entry["harmful_deletion"][str(keep)][name] = cluster_bootstrap_mean(
                    ev["harmful_deletion_rate"], ev["dialogues"], n_boot=n_boot).as_dict()
        out[label] = entry
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Confirmatory MSC addendum")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--run", required=True)
    s.add_argument("--split", default="dev")
    s.add_argument("--out", required=True)
    c = sub.add_parser("confirm")
    c.add_argument("--config", required=True)
    c.add_argument("--addendum", default="experiments/preregistration_addendum.md")
    c.add_argument("--runs", nargs="+", required=True)
    c.add_argument("--importance", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--primary", action="store_true", help="compute the primary family (horizon A runs only)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)

    if args.cmd == "select":
        rows = load_rows([Path(args.run)], args.split)
        report = select(rows)
        report["provenance"] = collect_provenance()
        Path(args.out).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "provenance"}, indent=1))
        return

    from benchmarks.dars_eval.importance import load_ratings

    addendum = Path(args.addendum)
    text = addendum.read_text(encoding="utf-8")
    if "Status: FROZEN" not in text:
        raise SystemExit("the addendum is not frozen; refusing to touch the confirmatory data")
    expected = Path(str(addendum) + ".sha256").read_text(encoding="utf-8").split()[0]
    if _sha256(addendum) != expected:
        raise SystemExit("the addendum does not match its recorded SHA-256; refusing to run")
    if _sha256(Path(args.config)) not in text:
        raise SystemExit("the config's SHA-256 is not the one recorded in the addendum; refusing to run")
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    rows = load_rows([Path(r) for r in args.runs])
    importance = load_ratings(Path(args.importance))
    missing = sorted({r["text"] for r in rows} - set(importance))
    if missing:
        raise SystemExit(f"{len(missing)} fact texts have no importance rating")
    scores = all_scores(rows, cfg, importance)
    label_sessions = sorted({r["label_session"] for r in rows})
    report: Dict[str, Any] = {
        "addendum_sha256": expected, "config": cfg, "runs": args.runs, "label_sessions": label_sessions,
        "facts": len(rows), "dialogues": len({r["dialogue"] for r in rows}),
        "by_split": {sp: len({r["dialogue"] for r in rows if r["split"] == sp}) for sp in sorted({r["split"] for r in rows})},
    }
    if args.primary:
        if label_sessions != [3]:
            raise SystemExit("the primary family is defined on horizon A (label session 3) only")
        report["primary_family"] = family(rows, scores, cfg.get("label", "lex_0.5"), cfg.get("keep", 0.5))
        report["primary_by_split"] = {
            sp: family([r for r in rows if r["split"] == sp],
                       {k: v[[i for i, r in enumerate(rows) if r["split"] == sp]] for k, v in scores.items()},
                       cfg.get("label", "lex_0.5"), cfg.get("keep", 0.5), n_boot_auc=2000, n_boot=5000)
            for sp in sorted({r["split"] for r in rows})}
    report["secondary"] = secondary(rows, scores)
    report["provenance"] = collect_provenance()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = "confirm_primary.json" if args.primary else f"confirm_secondary_s{'-'.join(map(str, label_sessions))}.json"
    (out / name).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    if args.primary:
        for r in report["primary_family"]:
            print(f"{r['id']:4s} {r['endpoint']:8s} {r['a']:>18s} - {r['b']:<18s} diff={r['diff']:+.4f} "
                  f"[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] p={r['p_value']:.4g} holm={r['holm_p']:.4g}  {r['result']}")
    print(f"wrote {out / name}")


if __name__ == "__main__":
    main()
