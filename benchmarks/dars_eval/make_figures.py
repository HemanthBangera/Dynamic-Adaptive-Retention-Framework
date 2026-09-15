"""
Figures 1–7 of the revised manuscript, drawn from the frozen test-phase artifacts
(``benchmark_runs/revision/test``, written by ``experiments/run_test_phase.sh``).

``--layout dev`` draws the same figures from the development artifacts, to validate
the plotting code before the test phase. Panels whose inputs are absent are skipped
and listed; any other error is raised.

Reader-accuracy panels use the same per-question mean over reader seeds as the H1
analysis in ``make_tables``. Output: one PDF and one 300-dpi PNG per figure.

Usage
-----
python -m benchmarks.dars_eval.make_figures --layout test
python -m benchmarks.dars_eval.make_figures --layout dev
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from benchmarks.dars_eval.make_tables import RUNS, Missing, by_question, read_json, read_jsonl  # noqa: E402
from benchmarks.dars_eval.stats import _groups, cluster_bootstrap_mean, cohen_kappa  # noqa: E402

# Okabe–Ito colour-blind-safe palette
COL = {
    "similarity": "#000000", "bm25": "#999999", "recency": "#E69F00", "random": "#BBBBBB",
    "dars_rrf_k15": "#0072B2", "dars_rrf_k50": "#0072B2", "dars_blend_a0.8": "#56B4E9",
    "dars_selected": "#0072B2", "dars_default": "#56B4E9", "recency_lru": "#E69F00", "fifo": "#D55E00",
    "frequency_lfu": "#CC79A7", "utility": "#009E73", "mention_count": "#F0E442", "no_memory": "#DDDDDD",
    "full_context": "#009E73",
}
LABEL = {
    "similarity": "Similarity", "bm25": "BM25", "recency": "Recency", "random": "Random",
    "dars_rrf_k15": "DARS (submitted, H1)", "dars_blend_a0.8": "DARS blend α=0.8", "dars_rrf_k50": "DARS RRF",
    "dars_selected": "DARS (selected)", "dars_default": "DARS (default)", "recency_lru": "LRU", "fifo": "FIFO",
    "frequency_lfu": "LFU", "utility": "Utility only", "mention_count": "Mention count",
    "no_memory": "No memory", "full_context": "Full context",
}
SOURCE_LABEL = {
    "ruler_qa1_197K": "RULER QA1", "ruler_qa2_421K": "RULER QA2", "longmemeval_s": "LongMemEval",
    "factconsolidation_sh_32k": "FactCons. single-hop", "factconsolidation_mh_32k": "FactCons. multi-hop",
    "eventqa_65536": "EventQA-65K", "eventqa_full": "EventQA-full",
}
SHORT_SOURCE = {"eventqa_65536": "EvQA\n65K", "eventqa_full": "EvQA\nfull", "ruler_qa1_197K": "QA1",
                "ruler_qa2_421K": "QA2", "longmemeval_s": "LME"}
LABELLED = ("ruler_qa1_197K", "ruler_qa2_421K", "longmemeval_s", "factconsolidation_sh_32k", "factconsolidation_mh_32k")
READER_SOURCES = ("eventqa_65536", "eventqa_full", "ruler_qa1_197K", "ruler_qa2_421K", "longmemeval_s")
BUDGETS = (1024, 2048, 5120, 8192, 16384)
FC = ("factconsolidation_sh_32k", "factconsolidation_mh_32k")
JUDGE_MODELS = ("gpt-4.1-nano-2025-04-14", "gpt-4o-mini-2024-07-18")

plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "legend.fontsize": 6,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
WIDE = 7.2          # inches, Scientific Reports double column


def layout(name: str) -> Dict[str, Any]:
    if name == "test":
        t = RUNS / "test"
        return {
            "split": "test", "out": t / "figures",
            "e1_recall": {s: t / "E1" / s for s in LABELLED},
            "e1_reader": {s: t / "E1" / s for s in READER_SOURCES},
            "e2_prec": {s: {"dars_l0.001": t / "E2" / f"{s}_b256_l0.001", "dars_l0.005": t / "E2" / f"{s}_b256_l0.005",
                            "base": t / "E2" / f"{s}_b256_l0.001"} for s in FC},
            "e2_dyn": t / "E2" / "lme_dynamics",
            "e2_noise": t / "E2" / "lme_feedback_noise",
            "e2_sources": t / "E2" / "lme_feedback_sources",
            "lme_evict": {"dars": [t / "E2" / "lme_budget50" / "dars"], "lru": [t / "E2" / "lme_budget50" / "lru"],
                          "fifo": [t / "E2" / "lme_budget50" / "fifo"], "lfu": [t / "E2" / "lme_budget50" / "lfu"],
                          "random": [t / "E2" / "lme_budget50" / f"random_s{s}" for s in range(5)]},
            "e4": t / "E4", "e5": t / "E5", "e6": t / "E6", "e8": t / "E8",
            "e9_eval": t / "E9" / "evaluate_test.json", "e9_analysis": t / "E9" / "analysis_test.json",
            "e10_eval": {sp: t / "E10" / f"evaluate_{sp}.json" for sp in ("test_in", "test_out")},
        }
    if name == "dev":
        d = RUNS
        return {
            "split": "dev", "out": d / "_figures_dev",
            "e1_recall": {s: d / "E1" / "dev" / f"{s}_dev" for s in LABELLED},
            "e1_reader": {s: d / "E1" / "dev_reader" / f"{s}_b5120" for s in READER_SOURCES},
            "e2_prec": {s: {"dars_l0.001": d / "E2" / "dev" / "lambda" / f"{s}_b256_l0.001",
                            "dars_l0.005": d / "E2" / "dev" / f"{s}_b256", "base": d / "E2" / "dev" / f"{s}_b256"}
                        for s in FC},
            "e2_dyn": d / "E2" / "dev" / "factconsolidation_sh_32k_b256",
            "e2_noise": d / "E2" / "dev" / "lme_feedback_noise",
            "e2_sources": d / "E2" / "dev" / "lme_feedback_sources",
            "lme_evict": {k: [d / "E2" / "dev" / "lme_budget50" / k] for k in ("dars", "lru", "fifo", "lfu", "random")},
            "e4": d / "E4" / "dev", "e5": d / "E5" / "dev", "e6": d / "E6" / "dev",
            "e8": d / "_smoke" / "E8_qa1",
            "e9_eval": d / "E9" / "dev_default" / "evaluate_dev.json",
            "e9_analysis": d / "E9" / "dev_default" / "analysis_dev.json",
            "e10_eval": {"dev": d / "E10" / "dev_default" / "evaluate_dev.json"},
        }
    raise ValueError(name)


# ── Helpers ──────────────────────────────────────────────────────────────────


def summary(run_dir: Path) -> Dict[str, Any]:
    return read_json(run_dir / "summary.json")


def bar(ax, x: float, est: Dict[str, float], color: str, width: float = 0.8, label: Optional[str] = None,
        mean_key: str = "mean") -> None:
    m = est[mean_key]
    ax.bar(x, m, width=width, color=color, edgecolor="black", linewidth=0.4, label=label)
    if est.get("ci_lo") is not None:
        ax.errorbar(x, m, yerr=[[m - est["ci_lo"]], [est["ci_hi"] - m]], fmt="none", ecolor="black", lw=0.6, capsize=1.5)


def legend_below(ax, ncol: int, offset: float = -0.18, fontsize: float = 5.5) -> None:
    ax.legend(frameon=False, fontsize=fontsize, loc="upper center", bbox_to_anchor=(0.5, offset), ncol=ncol)


def seed_mean_em(rec: Dict[str, Any]) -> Optional[float]:
    vals = [r["metrics"].get("substring_exact_match", 0.0) for r in rec.get("reader", [])]
    return float(np.mean(vals)) if vals else None


def reader_estimate(run_dir: Path, method: str, n_boot: int) -> Dict[str, float]:
    recs = by_question(run_dir, method)
    pairs = [(seed_mean_em(r), k[0]) for k, r in sorted(recs.items()) if seed_mean_em(r) is not None]
    if not pairs:
        raise Missing(f"{run_dir}: no reader outputs for {method}")
    return cluster_bootstrap_mean([v for v, _ in pairs], [c for _, c in pairs], n_boot=n_boot).as_dict()


def kappa_ci(pred: np.ndarray, ref: np.ndarray, clusters: np.ndarray, n_boot: int, seed: int = 0) -> Dict[str, float]:
    groups = _groups(clusters)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        picks = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[g][rng.integers(0, len(groups[g]), len(groups[g]))] for g in picks])
        boots.append(cohen_kappa(pred[idx], ref[idx]))
    return {"mean": cohen_kappa(pred, ref), "ci_lo": float(np.nanquantile(boots, 0.025)),
            "ci_hi": float(np.nanquantile(boots, 0.975))}


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── Figure 1: architecture and evaluation map ────────────────────────────────


def fig1(lay: Dict[str, Any], n_boot: int) -> List[str]:
    fig, ax = plt.subplots(figsize=(WIDE, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    w, h = 2.7, 1.5
    boxes = {
        "A": (0.1, 3.0, "Layer A · Cognitive Gateway\nquery reformulation,\nprompt construction", "E7"),
        "D": (3.65, 3.0, "Layer D · Memory Vault\n$S = w_rR + w_fF + w_uU + w_pP$\nfused retrieval", "E1–E4, E9, E10"),
        "C": (7.2, 3.0, "Layer C · Maintenance Manager\ntiers, compression,\nshadow indexing, eviction", "E3, E6, E8; H5"),
        "B": (3.65, 0.3, "Layer B · Learning Engine\nLLM judge → utility,\nfrequency, recency", "E5; E2 feedback"),
    }
    for x, y, text, exps in boxes.values():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc="#EAF2FB", ec="#0072B2", lw=0.8))
        ax.text(x + w / 2, y + 0.9, text, ha="center", va="center", fontsize=6)
        ax.text(x + w / 2, y + 0.18, f"tested in: {exps}", ha="center", va="center", fontsize=5.5, color="#D55E00")

    def arrow(p0, p1, both=False):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="<|-|>" if both else "-|>", mutation_scale=7, lw=0.8,
                                     color="#333333"))

    arrow((2.8, 3.75), (3.65, 3.75))
    ax.text(3.225, 3.9, "query", ha="center", va="bottom", fontsize=5.5)
    arrow((6.35, 3.75), (7.2, 3.75), both=True)
    ax.text(6.775, 3.9, "triage", ha="center", va="bottom", fontsize=5.5)
    arrow((5.6, 3.0), (5.6, 1.8))
    ax.text(5.7, 2.4, "memories\n+ answer", ha="left", va="center", fontsize=5.5)
    arrow((4.7, 1.8), (4.7, 3.0))
    ax.text(4.6, 2.4, "metadata\nupdates", ha="right", va="center", fontsize=5.5)
    ax.text(7.25, 1.45, "Pre-registered hypotheses\nH1 static retrieval (E1)\nH2 newest-fact precedence (E2)\n"
                        "H3 location ranking (E10)\nH4 retention prediction (E9)\nH5 harmful deletion (E2, E9, E10)\n"
                        "H6 judge reliability (E5)", fontsize=5.8, va="center")
    save(fig, lay["out"], "fig1_architecture")
    return []


# ── Figure 2: static retrieval at equal budgets (E1, H1) ────────────────────


def fig2(lay: Dict[str, Any], n_boot: int) -> List[str]:
    skipped = []
    fig, axes = plt.subplots(2, 3, figsize=(WIDE, 4.6))
    styles = {"bm25": "-", "dars_rrf_k15": "-", "dars_blend_a0.8": (0, (3, 1)), "recency": "-", "random": ":",
              "similarity": "-"}
    methods = ("bm25", "dars_rrf_k15", "dars_blend_a0.8", "recency", "random", "similarity")   # similarity drawn last
    for ax, src in zip(axes.flat[:5], LABELLED):
        try:
            s = summary(lay["e1_recall"][src])
        except Missing as exc:
            skipped.append(str(exc))
            ax.set_visible(False)
            continue
        for m in methods:
            if m not in s:
                continue
            pts = [(b, s[m].get(f"group_recall@{b}")) for b in BUDGETS]
            pts = [(b, e) for b, e in pts if e]
            if not pts:
                continue
            xs = [b for b, _ in pts]
            top = m == "similarity"
            ax.plot(xs, [e["mean"] for _, e in pts], marker="o", ms=2.5 if not top else 2, lw=1.4 if top else 1.0,
                    color=COL[m], label=LABEL[m], ls=styles[m], zorder=5 if top else 3)
            ax.fill_between(xs, [e["ci_lo"] for _, e in pts], [e["ci_hi"] for _, e in pts], color=COL[m], alpha=0.10, lw=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks(BUDGETS)
        ax.set_xticklabels(["1k", "2k", "5k", "8k", "16k"])
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(SOURCE_LABEL[src])
        ax.set_xlabel("Token budget B")
        ax.set_ylabel("Evidence recall")
    axes.flat[0].legend(loc="center right", frameon=False, fontsize=5.5)

    ax = axes.flat[5]
    readers = ("similarity", "bm25", "dars_rrf_k15", "no_memory")
    plotted = 0
    for i, src in enumerate(READER_SOURCES):
        for j, m in enumerate(readers):
            try:
                est = reader_estimate(lay["e1_reader"][src], m, n_boot)
            except Missing as exc:
                skipped.append(str(exc))
                continue
            bar(ax, i + (j - 1.5) * 0.2, est, COL[m], width=0.18, label=LABEL[m] if i == 0 or not plotted else None)
            plotted += 1
    ax.set_xticks(range(len(READER_SOURCES)))
    ax.set_xticklabels([SHORT_SOURCE[s] for s in READER_SOURCES], fontsize=5.5)
    ax.set_ylabel("Reader exact match (B = 5,120)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Reader accuracy")
    if plotted:
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), frameon=False, fontsize=5, loc="upper center",
                  bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout()
    save(fig, lay["out"], "fig2_static_retrieval")
    return skipped


# ── Figure 3: temporal precedence and ranking dynamics (E2, H2) ─────────────


def fig3(lay: Dict[str, Any], n_boot: int) -> List[str]:
    skipped = []
    split = lay["split"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDE, 2.9), gridspec_kw={"width_ratios": [1.2, 1]})
    series = [("similarity", "base", "similarity", "Similarity"), ("recency", "base", "recency", "Recency"),
              ("dars_rrf_k50", "dars_l0.005", "dars_default", "DARS RRF, λ = 0.005 (default)"),
              ("dars_rrf_k50", "dars_l0.001", "dars_selected", "DARS RRF, λ = 0.001 (selected)")]
    for i, src in enumerate(FC):
        for j, (method, run, color_key, label) in enumerate(series):
            try:
                est = summary(lay["e2_prec"][src][run])[f"{method}|none|{split}"]["precedence_rank"]
            except (Missing, KeyError) as exc:
                skipped.append(f"fig3 {src} {label}: {exc}")
                continue
            bar(a1, i + (j - 1.5) * 0.2, est, COL[color_key], width=0.18, label=label if i == 0 else None)
    a1.set_xticks(range(len(FC)))
    a1.set_xticklabels([SOURCE_LABEL[s] for s in FC])
    a1.set_ylabel("Rank precedence of the newest fact")
    a1.set_ylim(0, 1.05)
    legend_below(a1, ncol=2, offset=-0.12)
    a1.set_title("Newest version ranked above superseded versions")

    try:
        s = summary(lay["e2_dyn"])
        comps = ("R", "F", "U", "P")
        for j, fb in enumerate(("none", "oracle")):
            key = f"dars_rrf_k50|{fb}|{split}"
            vals = [s[key][f"dyn_neutral_changes_topk_{c}"] for c in comps]
            for i, v in enumerate(vals):
                bar(a2, i + (j - 0.5) * 0.35, v, "#0072B2" if fb == "none" else "#56B4E9", width=0.33,
                    label=("no feedback" if fb == "none" else "oracle feedback") if i == 0 else None)
        a2.set_xticks(range(4))
        a2.set_xticklabels(comps)
        a2.set_ylabel("Fraction of queries whose top-10 changes")
        a2.set_ylim(0, 1.05)
        legend_below(a2, ncol=2, offset=-0.12)
        a2.set_title("Influence of each component (set to its mean)")
    except (Missing, KeyError) as exc:
        skipped.append(f"fig3 dynamics: {exc}")
        a2.set_visible(False)
    fig.tight_layout()
    save(fig, lay["out"], "fig3_temporal")
    return skipped


# ── Figure 4: retention prediction and harmful deletion (H4, H5) ────────────


def fig4(lay: Dict[str, Any], n_boot: int) -> List[str]:
    skipped = []
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDE, 3.0), gridspec_kw={"width_ratios": [1, 1.5]})
    try:
        h4 = read_json(lay["e9_eval"])["h4"]
        entries = [("dars_selected", {"mean": h4["auroc_selected"]["auc"], "ci_lo": h4["auroc_selected"]["ci_lo"],
                                      "ci_hi": h4["auroc_selected"]["ci_hi"]}),
                   ("dars_default", {"mean": h4["auroc_default"]["auc"], "ci_lo": h4["auroc_default"]["ci_lo"],
                                     "ci_hi": h4["auroc_default"]["ci_hi"]})]
        for k in ("fifo", "utility", "mention_count", "recency_lru", "frequency_lfu"):
            c = h4["comparators"][k]
            entries.append((k, {"mean": c["auc"], "ci_lo": c["ci_lo"], "ci_hi": c["ci_hi"]}))
        for i, (k, est) in enumerate(entries):
            bar(a1, i, est, COL[k])
        a1.axhline(0.5, color="grey", lw=0.6, ls=":")
        a1.set_xticks(range(len(entries)))
        a1.set_xticklabels([LABEL[k] for k, _ in entries], rotation=40, ha="right")
        a1.set_ylabel("AUROC for later restatement")
        a1.set_ylim(0, 1)
        a1.set_title("MSC: which facts are needed later (H4)")
    except (Missing, KeyError) as exc:
        skipped.append(f"fig4 H4: {exc}")
        a1.set_visible(False)

    policies = ("dars_selected", "dars_default", "recency_lru", "fifo", "frequency_lfu")
    groups: List[tuple] = []
    try:
        h5 = read_json(lay["e9_eval"])["h5"]
        groups.append(("MSC", {"dars_selected": h5["selected"], **{k: h5["comparators"][k] for k in
                                                                    ("recency_lru", "fifo", "frequency_lfu")},
                               "dars_default": h5["comparators"]["default"]}))
    except (Missing, KeyError) as exc:
        skipped.append(f"fig4 H5 MSC: {exc}")
    for sp, path in lay["e10_eval"].items():
        try:
            hd = read_json(path)["h5"]["harmful_deletion"]
            groups.append((f"ALFWorld {sp.replace('test_', '')}", {"dars_selected": hd["selected"], "dars_default": hd["default"],
                                                              **{k: hd[k] for k in ("recency_lru", "fifo", "frequency_lfu")}}))
        except (Missing, KeyError) as exc:
            skipped.append(f"fig4 H5 ALFWorld {sp}: {exc}")
    try:
        split = lay["split"]
        key = f"dars_blend_a0.5|oracle|{split}"
        lme = {}
        for name, dirs in (("dars_selected", "dars"), ("recency_lru", "lru"), ("fifo", "fifo"), ("frequency_lfu", "lfu")):
            lme[name] = summary(lay["lme_evict"][dirs][0])[key]["harmful_deletion"]
        lme["dars_default"] = lme["dars_selected"]
        groups.append(("LongMemEval", lme))
    except (Missing, KeyError) as exc:
        skipped.append(f"fig4 H5 LongMemEval: {exc}")
    for i, (name, vals) in enumerate(groups):
        for j, p in enumerate(policies):
            bar(a2, i + (j - 2) * 0.16, vals[p], COL[p], width=0.15, label=LABEL[p] if i == 0 else None)
    a2.set_xticks(range(len(groups)))
    a2.set_xticklabels([g for g, _ in groups])
    a2.set_ylabel("Harmful-deletion rate (keep 50 %)")
    a2.set_ylim(0, 1.05)
    legend_below(a2, ncol=5, offset=-0.12)
    a2.set_title("Eviction under a memory budget (H5)")
    fig.tight_layout()
    save(fig, lay["out"], "fig4_retention")
    return skipped


# ── Figure 5: predictive relevance P (E4, E10) ──────────────────────────────


def fig5(lay: Dict[str, Any], n_boot: int) -> List[str]:
    skipped = []
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDE, 2.7), gridspec_kw={"width_ratios": [1.4, 1]})
    variants = ("none", "as_submitted", "task", "dynamic", "oracle")
    try:
        s = summary(lay["e4"])
        sources = [src for src in s]
        grid = np.full((len(variants), len(sources)), np.nan)
        for j, src in enumerate(sources):
            for i, v in enumerate(variants):
                d = s[src].get(f"{v}|dars_rrf_k50", {}).get("vs_similarity", {}).get("group_recall@1024")
                if d:
                    grid[i, j] = d["diff"]
        lim = max(0.05, float(np.nanmax(np.abs(grid))))
        im = a1.imshow(grid, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
        for i in range(len(variants)):
            for j in range(len(sources)):
                if not np.isnan(grid[i, j]):
                    a1.text(j, i, f"{grid[i, j]:+.2f}", ha="center", va="center", fontsize=5.5,
                            color="white" if abs(grid[i, j]) > 0.6 * lim else "black")
        a1.set_yticks(range(len(variants)))
        a1.set_yticklabels(["none", "as submitted", "domain-matched", "dynamic", "oracle"])
        a1.set_xticks(range(len(sources)))
        a1.set_xticklabels([SOURCE_LABEL.get(x.replace("*", ""), x) for x in sources], rotation=30, ha="right")
        a1.set_title("Retrieval: recall change vs similarity (RRF, B = 1k)")
        fig.colorbar(im, ax=a1, fraction=0.04, pad=0.02)
    except (Missing, KeyError) as exc:
        skipped.append(f"fig5 E4: {exc}")
        a1.set_visible(False)
    entries = []
    for sp, path in lay["e10_eval"].items():
        try:
            hd = read_json(path)["h5"]["harmful_deletion"]
            entries += [(f"P only\n({sp})", hd["selected"], "#0072B2"), (f"default\n({sp})", hd["default"], "#56B4E9"),
                        (f"LFU\n({sp})", hd["frequency_lfu"], COL["frequency_lfu"])]
        except (Missing, KeyError) as exc:
            skipped.append(f"fig5 ALFWorld {sp}: {exc}")
    for i, (label, est, color) in enumerate(entries):
        bar(a2, i, est, color)
    a2.set_xticks(range(len(entries)))
    a2.set_xticklabels([e[0] for e in entries], fontsize=5.5)
    a2.set_ylabel("Harmful-deletion rate (keep 50 %)")
    a2.set_title("Retention: ALFWorld eviction")
    fig.tight_layout()
    save(fig, lay["out"], "fig5_predictive_relevance")
    return skipped


# ── Figure 6: judge reliability and feedback noise (E5, E2) ─────────────────


def fig6(lay: Dict[str, Any], n_boot: int) -> List[str]:
    skipped = []
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDE, 2.7), gridspec_kw={"width_ratios": [1.4, 1]})
    judge_error = None
    try:
        from benchmarks.dars_eval.run_judge import lexical_turn_verdict

        items = read_jsonl(lay["e5"] / "items.jsonl")
        ref = np.array([it["reference"] for it in items])
        clusters = np.array([f"{it['source']}:{it['context']}" for it in items])
        entries = []
        for model in JUDGE_MODELS:
            for variant in ("layer_b|s0", "layer_b|s1", "paraphrase|s0"):
                key = f"{model}|{variant}"
                if key not in items[0].get("judge", {}):
                    continue
                v = np.array([it["judge"][key] for it in items])
                keep = v != "NEUTRAL"
                est = kappa_ci(v[keep] == "YES", ref[keep], clusters[keep], min(n_boot, 1000))
                short = "nano" if "nano" in model else "4o-mini"
                vlabel = {"layer_b|s0": "seed 0", "layer_b|s1": "seed 1",
                          "paraphrase|s0": "paraphrase"}.get(variant, variant)
                entries.append((f"{short}\n{vlabel}", est,
                                "#0072B2" if "nano" in model else "#56B4E9"))
                if key == f"{JUDGE_MODELS[0]}|layer_b|s0":
                    judge_error = float(np.mean((v[keep] == "YES") != ref[keep]))
        for tau in (0.3, 0.5, 0.7):
            pred = np.array([lexical_turn_verdict(it.get("parsed") or it["answer"], it["memories"], tau) for it in items])
            entries.append((f"lexical\nτ={tau}", kappa_ci(pred, ref, clusters, min(n_boot, 1000)), "#009E73"))
        for i, (label, est, color) in enumerate(entries):
            bar(a1, i, est, color)
        a1.axhline(0.60, color="#D55E00", lw=0.7, ls="--")
        a1.text(len(entries) - 0.5, 0.62, "H6 threshold 0.60", color="#D55E00", fontsize=5.5, ha="right")
        a1.set_xticks(range(len(entries)))
        a1.set_xticklabels([e[0] for e in entries], fontsize=5)
        a1.set_ylabel("Cohen's κ vs reference labels")
        a1.set_ylim(min(0, min(e[1]["ci_lo"] for e in entries) - 0.05), 1)
        a1.set_title("Layer B judge reliability (H6)")
    except (Missing, KeyError, IndexError) as exc:
        skipped.append(f"fig6 E5: {exc}")
        a1.set_visible(False)
    try:
        s = summary(lay["e2_noise"])
        split = lay["split"]
        xs, ys = [], []
        for eps, fb in ((0.0, "oracle"), (0.1, "oracle_noisy:0.1"), (0.2, "oracle_noisy:0.2"),
                        (0.3, "oracle_noisy:0.3"), (0.5, "oracle_noisy:0.5")):
            key = f"dars_rrf_k50|{fb}|{split}"
            if key in s:
                xs.append(eps)
                ys.append(s[key]["group_recall"])
        a2.errorbar(xs, [y["mean"] for y in ys], yerr=[[y["mean"] - y["ci_lo"] for y in ys], [y["ci_hi"] - y["mean"] for y in ys]],
                    marker="o", ms=3, lw=1, color="#0072B2", capsize=1.5, label="oracle with flipped verdicts")
        none = s.get(f"dars_rrf_k50|none|{split}", {}).get("group_recall")
        if none:
            a2.axhline(none["mean"], color="grey", ls=":", lw=0.8, label="no feedback")
        try:
            judge = summary(lay["e2_sources"])[f"dars_rrf_k50|judge|{split}"]["group_recall"]
            if judge_error is not None:
                a2.errorbar([judge_error], [judge["mean"]], yerr=[[judge["mean"] - judge["ci_lo"]], [judge["ci_hi"] - judge["mean"]]],
                            marker="s", ms=4, color="#D55E00", capsize=1.5, label="LLM judge (at its error rate)")
        except (Missing, KeyError):
            skipped.append("fig6: judge-feedback stream not available")
        a2.set_xlabel("Fraction of feedback verdicts that are wrong (ε)")
        a2.set_ylabel("Evidence recall (LongMemEval, B = 2,048)")
        legend_below(a2, ncol=1, offset=-0.22)
        a2.set_title("Effect of feedback errors")
    except (Missing, KeyError) as exc:
        skipped.append(f"fig6 noise: {exc}")
        a2.set_visible(False)
    fig.tight_layout()
    save(fig, lay["out"], "fig6_judge")
    return skipped


# ── Figure 7: compression and efficiency (E6, E8) ───────────────────────────


def fig7(lay: Dict[str, Any], n_boot: int) -> List[str]:
    skipped = []
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 2.8))
    comps = ("semantic", "llmlingua2", "llmlingua2@matched", "extractive", "extractive@matched")
    names = {"semantic": "Layer C", "llmlingua2": "LLML-2", "llmlingua2@matched": "LLML-2\nmatched",
             "extractive": "Extract.", "extractive@matched": "Extract.\nmatched"}
    colors = {"semantic": "#0072B2", "llmlingua2": "#009E73", "llmlingua2@matched": "#8FD0B8",
              "extractive": "#E69F00", "extractive@matched": "#F4CE8A"}
    try:
        s = summary(lay["e6"])
        present = [c for c in comps if c in s]
        ticks = [f"{names[c]}\n({s[c]['compression_ratio']['mean']:.2f})" for c in present]
        for i, c in enumerate(present):
            bar(axes[0], i - 0.2, s[c]["answer_retention"], colors[c], width=0.38)
            d = s[c]["reader_delta_vs_original"]
            bar(axes[0], i + 0.2, {"mean": d["diff"], "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"]}, "white", width=0.38)
        axes[0].axhline(0, color="black", lw=0.5)
        axes[0].set_xticks(range(len(present)))
        axes[0].set_xticklabels(ticks, fontsize=5)
        axes[0].set_ylim(-0.5, 1.05)
        axes[0].set_ylabel("Answer retained / Δ reader accuracy")
        axes[0].set_title("Fidelity (ratio in brackets)")
        for i, c in enumerate(present):
            bar(axes[1], i - 0.2, s[c]["recall@10_kept_vector"], "#999999", width=0.38,
                label="original vector kept" if i == 0 else None)
            bar(axes[1], i + 0.2, s[c]["recall@10_reembedded"], colors[c], width=0.38,
                label="summary re-embedded" if i == 0 else None)
        axes[1].set_xticks(range(len(present)))
        axes[1].set_xticklabels([names[c] for c in present], fontsize=5)
        axes[1].set_ylabel("Recall@10 of the evidence memory")
        axes[1].set_ylim(0, 1.05)
        legend_below(axes[1], ncol=1, offset=-0.2, fontsize=5)
        axes[1].set_title("Shadow indexing vs re-embedding")
    except (Missing, KeyError) as exc:
        skipped.append(f"fig7 E6: {exc}")
        axes[0].set_visible(False)
        axes[1].set_visible(False)
    try:
        s = summary(lay["e8"])
        rows = {}
        for key, v in s.items():
            src, method, comp, b = key.split("|")
            if method == "similarity":
                rows.setdefault(comp, {}).setdefault(int(b), []).append(v["answer_present"]["mean"])
        styles = {"none": ("#000000", "original text"), "extractive": ("#E69F00", "extractive"),
                  "llmlingua2": ("#009E73", "LLMLingua-2")}
        budgets = sorted({b for by_b in rows.values() for b in by_b})
        for comp, by_b in rows.items():
            bs = sorted(by_b)
            axes[2].plot(bs, [float(np.mean(by_b[b])) for b in bs], marker="o", ms=3, lw=1,
                         color=styles[comp][0], label=styles[comp][1])
        axes[2].set_xscale("log", base=2)
        axes[2].set_xticks(budgets)
        axes[2].set_xticklabels([f"{b // 1024}k" if b >= 1024 else str(b) for b in budgets])
        axes[2].minorticks_off()
        axes[2].set_xlabel("Token budget B")
        axes[2].set_ylabel("Gold answer present in context")
        axes[2].set_ylim(0, 1.05)
        legend_below(axes[2], ncol=1, offset=-0.25, fontsize=5)
        axes[2].set_title("Equal-budget efficiency (similarity)")
    except (Missing, KeyError) as exc:
        skipped.append(f"fig7 E8: {exc}")
        axes[2].set_visible(False)
    fig.tight_layout()
    save(fig, lay["out"], "fig7_compression_efficiency")
    return skipped


FIGURES: Dict[str, Callable[[Dict[str, Any], int], List[str]]] = {
    "fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5, "fig6": fig6, "fig7": fig7,
}


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Figures 1–7 of the revised manuscript")
    p.add_argument("--layout", choices=("test", "dev"), default="test")
    p.add_argument("--only", nargs="+", default=list(FIGURES), choices=list(FIGURES))
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args(argv)
    lay = layout(args.layout)
    report = {}
    for name in args.only:
        report[name] = FIGURES[name](lay, args.n_boot)
        print(f"{name}: drawn" + (f"; skipped {len(report[name])} panel input(s)" if report[name] else ""))
        for msg in report[name]:
            print(f"   - {msg}")
    lay["out"].mkdir(parents=True, exist_ok=True)
    (lay["out"] / "figures_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
