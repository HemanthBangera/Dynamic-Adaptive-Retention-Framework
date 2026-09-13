"""
Figures 2–7 of the manuscript after the fourth audit, drawn only from artifacts.

Fig. 1 (architecture) comes from ``make_figures`` and Table 1 from ``make_tables``. Panels whose inputs do not
exist yet are skipped and listed in ``figures_revised_report.json``; any other error is raised.

    python -m benchmarks.dars_eval.make_figures_revised
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from benchmarks.dars_eval.make_figures import WIDE, fig1, save  # noqa: E402

RUNS = Path("benchmark_runs/revision")
TEST = RUNS / "test"
AUD = TEST / "E13_audit"
ADD = RUNS / "addendum"
ADD2 = RUNS / "addendum2"
OUT = TEST / "figures_revised"

C_WRITE, C_READ, C_BASE, C_PRIOR, C_META, C_BAD = "#0072B2", "#56B4E9", "#999999", "#CC79A7", "#009E73", "#D55E00"


def _load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def _bars(ax, names: Sequence[str], means: Sequence[float], los: Sequence[float], his: Sequence[float],
          colors: Sequence[str], horizontal: bool = True) -> None:
    y = np.arange(len(names))
    err = np.array([[m - lo for m, lo in zip(means, los)], [hi - m for m, hi in zip(means, his)]])
    if horizontal:
        ax.barh(y, means, xerr=err, color=colors, height=0.7, error_kw={"lw": 0.6, "capsize": 1.5})
        ax.set_yticks(y, names)
        ax.invert_yaxis()
    else:
        ax.bar(y, means, yerr=err, color=colors, width=0.7, error_kw={"lw": 0.6, "capsize": 1.5})
        ax.set_xticks(y, names, rotation=40, ha="right")


# ── Fig. 2: write- vs read-anchored retention signals ──────────────────────────


def fig_confirmatory() -> List[str]:
    """(a) AUROC and (b) harmful deletion on untouched MSC dialogues; (c) the same ordering on the original test."""
    missing: List[str] = []
    prim = _load(ADD / "confirm" / "confirm_primary.json")
    order = ["write_h4", "write_default", "metadata_model", "read_h4", "utility_read", "fifo", "recency_lru",
             "generative_agents", "memorybank", "frequency_lfu"]
    label = {"write_h4": "DARS, write-anchored (dev-selected)", "write_default": "DARS, write-anchored (default weights)",
             "metadata_model": "Two-feature metadata model", "read_h4": "DARS, read-anchored (pre-registered)",
             "utility_read": "Utility credited on retrieval", "fifo": "FIFO (creation order)", "recency_lru": "LRU",
             "generative_agents": "Generative Agents-style", "memorybank": "MemoryBank-style", "frequency_lfu": "LFU"}
    colors = {"write_h4": C_WRITE, "write_default": C_WRITE, "metadata_model": C_META, "read_h4": C_READ,
              "utility_read": C_READ, "fifo": C_BASE, "recency_lru": C_BASE, "generative_agents": C_PRIOR,
              "memorybank": C_PRIOR, "frequency_lfu": C_BASE}
    sec = prim["secondary"]["lex_0.5"]
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 2.9), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    au = sec["auroc"]
    _bars(axes[0], [label[k] for k in order], [au[k]["clustered"]["auc"] for k in order],
          [au[k]["clustered"]["ci_lo"] for k in order], [au[k]["clustered"]["ci_hi"] for k in order],
          [colors[k] for k in order])
    axes[0].axvline(0.5, color="k", lw=0.5, ls=":")
    axes[0].set_xlabel("AUROC, restated in next session")
    axes[0].set_title("a  Untouched MSC dialogues (n = 1,001)", loc="left")
    hd = sec["harmful_deletion"]["0.5"]
    order_h = ["write_h4", "metadata_model", "read_h4", "fifo", "utility_read", "recency_lru", "generative_agents",
               "memorybank", "frequency_lfu"]
    _bars(axes[1], [label[k].split(" (")[0] for k in order_h], [hd[k]["mean"] for k in order_h],
          [hd[k]["ci_lo"] for k in order_h], [hd[k]["ci_hi"] for k in order_h], [colors[k] for k in order_h])
    axes[1].set_xlabel("Harmful deletion at 50 % budget (lower is better)")
    axes[1].set_title("b  Eviction", loc="left")
    # (c) write − read AUROC by label threshold, three datasets
    labels = ["lex_0.5", "lex_0.6", "lex_0.7", "embed_0.8"]
    series = {"Untouched, next session": [prim["secondary"][l]["auroc"]["write_h4"]["delong"]["auc"]
                                          - prim["secondary"][l]["auroc"]["read_h4"]["delong"]["auc"] for l in labels]}
    try:
        s4 = _load(ADD / "confirm" / "confirm_secondary_s4.json")["secondary"]
        series["Untouched, 2 sessions ahead"] = [s4[l]["auroc"]["write_h4"]["delong"]["auc"]
                                                - s4[l]["auroc"]["read_h4"]["delong"]["auc"] for l in labels]
    except FileNotFoundError as e:
        missing.append(str(e))
    try:
        mw = _load(AUD / "msc_writes.json")["labels"]
        series["Original test (exploratory)"] = [mw[l]["auroc"]["write_h4"] - mw[l]["auroc"]["read_h4"] for l in labels]
    except FileNotFoundError as e:
        missing.append(str(e))
    x = np.arange(len(labels))
    for i, (name, vals) in enumerate(series.items()):
        axes[2].bar(x + (i - 1) * 0.26, vals, width=0.26, label=name, color=[C_WRITE, C_READ, C_BASE][i % 3])
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].set_xticks(x, ["F1≥.5", "F1≥.6", "F1≥.7", "emb≥.8"], rotation=45, ha="right")
    axes[2].set_ylabel("AUROC, write − read anchored")
    axes[2].set_xlabel("Restatement label")
    axes[2].set_title("c  Robust to label and horizon", loc="left")
    axes[2].legend(frameon=False, fontsize=5, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1)
    fig.tight_layout()
    save(fig, OUT, "fig4_write_vs_read")
    return missing


# ── Fig. 3: mechanisms ─────────────────────────────────────────────────────────


def fig_mechanisms() -> List[str]:
    per = _load(AUD / "persistence.json")
    unit = _load(AUD / "lme_rank_oracle_unit" / "summary.json")
    age = _load(AUD / "lme_age.json")
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 2.4))
    names = ["MSC (next session)", "RULER QA1", "FactCons. mh", "FactCons. sh", "LongMemEval"]
    vals = [per["msc"]["p_needed_given_restated_in_s2"],
            per["mab"]["ruler_qa1_197K"]["share_of_evidence_units_needed_by_more_than_one_question"],
            per["mab"]["factconsolidation_mh_32k"]["share_of_evidence_units_needed_by_more_than_one_question"],
            per["mab"]["factconsolidation_sh_32k"]["share_of_evidence_units_needed_by_more_than_one_question"],
            per["mab"]["longmemeval_s*"]["share_of_evidence_units_needed_by_more_than_one_question"]]
    axes[0].barh(np.arange(len(names)), vals, color=[C_WRITE, C_BASE, C_BASE, C_BASE, C_BAD], height=0.7)
    for i, v in enumerate(vals):
        axes[0].text(v + 0.02, i, f"{v:.3f}" if v < 0.01 else f"{v:.2f}", va="center", fontsize=5.5)
    axes[0].set_xlim(0, 1)
    axes[0].set_yticks(np.arange(len(names)), names)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Share of needed memories needed again")
    axes[0].set_title("a  Do needs recur?", loc="left")
    methods = [("dars_rrf_k50", "Rank fusion"), ("dars_blend_a0.5", "Blend")]
    fb = [("none", "No feedback", C_BASE), ("oracle_unit", "Oracle, per memory", C_WRITE), ("oracle", "Oracle, all shown", C_BAD)]
    x = np.arange(len(methods))
    for i, (key, lab, col) in enumerate(fb):
        m = [unit[f"{meth}|{key}|test"]["mrr"] for meth, _ in methods]
        axes[1].bar(x + (i - 1) * 0.27, [v["mean"] for v in m], width=0.27, color=col, label=lab,
                    yerr=[[v["mean"] - v["ci_lo"] for v in m], [v["ci_hi"] - v["mean"] for v in m]],
                    error_kw={"lw": 0.6, "capsize": 1.5})
    axes[1].set_xticks(x, [n for _, n in methods])
    axes[1].set_ylabel("Evidence MRR (LongMemEval)")
    axes[1].set_title("b  Credit assignment", loc="left")
    axes[1].legend(frameon=False, fontsize=5)
    k = "share_of_evidence_in_newest_half"
    axes[2].bar([0], [age[k]["mean"]], yerr=[[age[k]["mean"] - age[k]["ci_lo"]], [age[k]["ci_hi"] - age[k]["mean"]]],
                color=C_BAD, width=0.5, error_kw={"lw": 0.6, "capsize": 2})
    axes[2].axhline(0.5, color="k", lw=0.5, ls=":")
    axes[2].set_xticks([0], ["LongMemEval evidence"])
    axes[2].set_xlim(-1, 1)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Share in newest half of store")
    axes[2].set_title("c  Why FIFO wins there", loc="left")
    fig.tight_layout()
    save(fig, OUT, "fig5_mechanisms")
    return []


# ── Fig. 4: conflicting facts and display order ───────────────────────────────


def fig_display_order() -> List[str]:
    missing: List[str] = []
    base = AUD / "fc_serial" / "factconsolidation_sh_32k"
    order = AUD / "fc_order" / "factconsolidation_sh_32k"
    conds = [("MAB prompt,\ntop memory first", base / "prefix_with__prompt_mab"),
             ("No serial rule", base / "prefix_with__prompt_no_serial_rule"),
             ("No rule,\nno serial prefix", base / "prefix_without__prompt_no_serial_rule"),
             ("MAB prompt,\ntop memory last", order / "order_best_last")]
    meths = [("bm25", "BM25", C_BASE), ("similarity", "Similarity", "#000000"), ("dars_rrf_k50", "DARS", C_READ),
             ("bm25_dars_wrrf_k50_b0.5", "DARS over BM25", C_WRITE)]
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.5), gridspec_kw={"width_ratios": [2.2, 1]})
    x = np.arange(len(conds))
    for i, (key, lab, col) in enumerate(meths):
        vals, lo, hi = [], [], []
        for _, d in conds:
            s = _load(d / "summary.json")[f"{key}|none|test"]["reader_substring_exact_match"]
            vals.append(s["mean"]); lo.append(s["mean"] - s["ci_lo"]); hi.append(s["ci_hi"] - s["mean"])
        axes[0].bar(x + (i - 1.5) * 0.2, vals, width=0.2, color=col, label=lab, yerr=[lo, hi], error_kw={"lw": 0.5, "capsize": 1})
    axes[0].set_xticks(x, [c for c, _ in conds])
    axes[0].set_ylabel("Reader accuracy (FactConsolidation single-hop)")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(frameon=False, ncol=4, fontsize=5, loc="upper center")
    axes[0].set_title("a  Ranking converts to answers only when the top memory is shown last", loc="left")
    try:
        conf = _load(ADD2 / "confirm_display_order.json")["family"]
        names = [f"{r['id']}\n{r['source'].split('_')[-1]}" for r in conf]
        _bars(axes[1], names, [r["diff"] for r in conf], [r["ci_lo"] for r in conf], [r["ci_hi"] for r in conf],
              [C_WRITE if r["result"] == "confirmed" else C_BASE for r in conf], horizontal=False)
        axes[1].axhline(0, color="k", lw=0.5)
        axes[1].set_ylabel("Accuracy difference")
        axes[1].set_title("b  Pre-registered confirmation", loc="left")
    except FileNotFoundError as e:
        missing.append(str(e))
        axes[1].axis("off")
    fig.tight_layout()
    save(fig, OUT, "fig3_display_order")
    return missing


# ── Fig. 5: feedback reliability ───────────────────────────────────────────────


def fig_feedback() -> List[str]:
    law = _load(TEST / "E12" / "feedback_law.json")
    e5 = _load(TEST / "E5" / "report.json")
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 2.4))
    cols = ["#000000", "#0072B2", "#56B4E9", "#BBBBBB"]
    for lv, col in zip(law["levels"], cols):
        eps = [p["eps"] for p in lv["curve"]]
        axes[0].plot(eps, [p["harmful_deletion"] for p in lv["curve"]], color=col, lw=1,
                     label=f"{lv['mean_verdicts_per_memory_with_feedback']:.1f} verdicts/memory")
        axes[1].plot(eps, [p["auroc"] for p in lv["curve"]], color=col, lw=1)
    axes[0].axhline(law["baselines"]["fifo_harmful_deletion"], color=C_BAD, lw=0.8, ls="--", label="FIFO")
    axes[1].axhline(law["baselines"]["recency_auroc"], color=C_BAD, lw=0.8, ls="--", label="Recency")
    jc = law["judge_calibrated"]
    axes[0].scatter([0.37], [jc["harmful_deletion"]], color=C_BAD, s=12, zorder=5, label="Judge's error profile")
    axes[1].scatter([0.37], [jc["auroc"]], color=C_BAD, s=12, zorder=5)
    axes[0].set_xlabel("Verdict error rate ε"); axes[0].set_ylabel("Harmful deletion (MSC)")
    axes[1].set_xlabel("Verdict error rate ε"); axes[1].set_ylabel("AUROC (MSC)")
    axes[0].legend(frameon=False, fontsize=4.5)
    axes[0].set_title("a  Eviction is fragile", loc="left")
    axes[1].set_title("b  Ranking is tolerant", loc="left")
    raters = [("gpt-4.1-nano-2025-04-14|layer_b|s0", "Judge (nano)"), ("gpt-4.1-nano-2025-04-14|paraphrase|s0", "Nano, paraphrase"),
              ("gpt-4o-mini-2024-07-18|layer_b|s0", "Judge (4o-mini)"), ("lexical_tau0.5", "Lexical rule")]
    ks = [e5["raters"][k]["kappa"] for k, _ in raters]
    axes[2].barh(np.arange(len(raters)), ks, color=[C_READ, C_READ, C_BAD, C_BASE], height=0.7)
    axes[2].axvline(0.60, color="k", lw=0.6, ls="--")
    axes[2].set_yticks(np.arange(len(raters)), [n for _, n in raters])
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Cohen's κ vs ground truth (bar: 0.60)")
    axes[2].set_title("c  LLM judges (H6)", loc="left")
    fig.tight_layout()
    save(fig, OUT, "fig6_feedback_reliability")
    return []


# ── Fig. 6: static retrieval and the embedder ─────────────────────────────────


def fig_static() -> List[str]:
    h1 = _load(AUD / "h1.json")["sources"]
    emb = _load(AUD / "embedder_robustness.json")["sources"]
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.4))
    srcs = [("ruler_qa1_197K", "RULER QA1"), ("ruler_qa2_421K", "RULER QA2"), ("longmemeval_s*", "LongMemEval")]
    budgets = ["1024", "2048", "5120"]
    x = np.arange(len(budgets))
    for i, (s, lab) in enumerate(srcs):
        d = h1[s]["budgets"]["dars_rrf_k50"]
        vals = [d[b]["recall_diff"]["diff"] for b in budgets]
        err = [[d[b]["recall_diff"]["diff"] - d[b]["recall_diff"]["ci_lo"] for b in budgets],
               [d[b]["recall_diff"]["ci_hi"] - d[b]["recall_diff"]["diff"] for b in budgets]]
        axes[0].bar(x + (i - 1) * 0.27, vals, width=0.27, yerr=err, label=lab, color=[C_BASE, C_READ, C_BAD][i],
                    error_kw={"lw": 0.5, "capsize": 1})
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_xticks(x, [f"B = {b}" for b in budgets])
    axes[0].set_ylabel("Evidence recall, DARS − similarity")
    axes[0].set_title("a  Where DARS (fetch 50) changes the shown set, recall falls", loc="left")
    axes[0].legend(frameon=False, fontsize=5)
    rows = [("similarity_minilm", "MiniLM similarity", C_READ), ("bm25", "BM25", C_BASE), ("similarity_bge", "bge-small similarity", C_WRITE)]
    x = np.arange(len(srcs))
    for i, (k, lab, col) in enumerate(rows):
        m = [emb[s]["means"][f"{k}|recall@5120"] for s, _ in srcs]
        axes[1].bar(x + (i - 1) * 0.27, [v["mean"] for v in m], width=0.27, color=col, label=lab,
                    yerr=[[v["mean"] - v["ci_lo"] for v in m], [v["ci_hi"] - v["mean"] for v in m]], error_kw={"lw": 0.5, "capsize": 1})
    axes[1].set_xticks(x, [l for _, l in srcs])
    axes[1].set_ylabel("Evidence recall at B = 5,120")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(frameon=False, fontsize=5)
    axes[1].set_title("b  The static winner depends on the embedder", loc="left")
    fig.tight_layout()
    save(fig, OUT, "fig2_static_retrieval")
    return []


# ── Fig. 7: gateway and maintenance layers ────────────────────────────────────


def fig_layers() -> List[str]:
    missing: List[str] = []
    e7 = _load(TEST / "E7" / "summary.json")
    lme = _load(AUD / "lme_layera" / "summary.json")
    e6 = _load(TEST / "E6" / "summary.json")
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 2.4))
    keys = [("reformulation (mab, best_first)", "Reformulation"), ("xml vs mab (raw, best_first)", "XML prompt"),
            ("best_last vs best_first (raw, mab)", "Best-last order"), ("gateway as implemented vs direct (best_first)", "Gateway as built")]
    x = np.arange(len(keys))
    for i, (d, lab, col) in enumerate([(e7, "EventQA-65K + RULER QA1", C_READ), (lme, "LongMemEval", C_WRITE)]):
        c = d["contrasts"]
        vals = [c[k]["diff"] for k, _ in keys]
        err = [[c[k]["diff"] - c[k]["ci_lo"] for k, _ in keys], [c[k]["ci_hi"] - c[k]["diff"] for k, _ in keys]]
        axes[0].bar(x + (i - 0.5) * 0.38, vals, width=0.38, color=col, label=lab, yerr=err, error_kw={"lw": 0.5, "capsize": 1})
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_xticks(x, [n for _, n in keys], rotation=30, ha="right")
    axes[0].set_ylabel("Reader accuracy difference")
    axes[0].set_title("a  Layer A", loc="left")
    axes[0].legend(frameon=False, fontsize=5)
    comps = [("semantic", "Semantic"), ("llmlingua2", "LLMLingua-2"), ("extractive", "Extractive")]
    d = [e6[k]["reader_delta_vs_original"] for k, _ in comps]
    axes[1].bar(np.arange(3), [v["diff"] for v in d], color=[C_READ, C_BASE, C_BAD],
                yerr=[[v["diff"] - v["ci_lo"] for v in d], [v["ci_hi"] - v["diff"] for v in d]], error_kw={"lw": 0.5, "capsize": 1.5})
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_xticks(np.arange(3), [n for _, n in comps])
    axes[1].set_ylabel("Accuracy change, compressed evidence")
    axes[1].set_title("b  Layer C compression", loc="left")
    try:
        w = _load(AUD / "e6_whole_context" / "summary.json")
        conds = [("recall@10_kept_vector", "Kept vector (shadow index)", C_WRITE),
                 ("recall@10_reembedded", "Re-embedded, evidence only", C_BASE),
                 ("recall@10_reembedded_whole_context", "Re-embedded, whole store", C_READ)]
        x, width = np.arange(len(comps)), 0.26
        for i, (k, lab, col) in enumerate(conds):
            vals = [w[c][k] for c, _ in comps]
            axes[2].bar(x + (i - 1) * width, [v["mean"] for v in vals], width, color=col, label=lab,
                        yerr=[[v["mean"] - v["ci_lo"] for v in vals], [v["ci_hi"] - v["mean"] for v in vals]],
                        error_kw={"lw": 0.5, "capsize": 1.2})
        axes[2].set_xticks(x, [n for _, n in comps])
        axes[2].set_ylim(0, 1.38)
        axes[2].set_yticks(np.arange(0, 1.01, 0.2))
        axes[2].legend(frameon=False, fontsize=5, loc="upper left", ncol=1)
        axes[2].set_ylabel("Recall@10 of compressed evidence")
        axes[2].set_title("c  Shadow indexing", loc="left")
    except (FileNotFoundError, KeyError) as e:
        missing.append(f"fig7c: {e}")
        axes[2].axis("off")
    fig.tight_layout()
    save(fig, OUT, "fig7_layers")
    return missing


# Numbered in order of first citation in the manuscript.
FIGURES = {"fig1": lambda: fig1({"out": OUT}, 0), "fig2": fig_static, "fig3": fig_display_order,
           "fig4": fig_confirmatory, "fig5": fig_mechanisms, "fig6": fig_feedback, "fig7": fig_layers}


def main(argv: Optional[Sequence[str]] = None) -> None:
    argparse.ArgumentParser(description="Revised figures").parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, fn in FIGURES.items():
        try:
            report[name] = fn()
            print(f"{name}: drawn" + (f" (missing: {report[name]})" if report[name] else ""))
        except FileNotFoundError as e:
            report[name] = [f"skipped: {e}"]
            print(f"{name}: skipped, missing {e}")
    (OUT / "figures_revised_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
