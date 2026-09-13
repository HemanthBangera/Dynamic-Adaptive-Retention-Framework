"""
Tables for the post-freeze analyses: the two confirmatory addenda and the fourth audit's exploratory analyses.

Every number is read from an artifact; nothing is typed by hand. Output:
``benchmark_runs/revision/_tables_test/exploratory.md`` (for the SI) and ``exploratory.json``.

    python -m benchmarks.dars_eval.make_exploratory_tables
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

RUNS = Path("benchmark_runs/revision")
T = RUNS / "test"
A = T / "E13_audit"
OUT = RUNS / "_tables_test"


def _j(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ci(d: Dict[str, Any], key: str = "mean", fmt: str = "{:.3f}") -> str:
    v = d.get(key, d.get("diff", d.get("auc")))
    return f"{fmt.format(v)} [{fmt.format(d['ci_lo'])}, {fmt.format(d['ci_hi'])}]"


def esc(x: Any) -> str:
    """A table cell: run keys such as "similarity|random" would split the cell, so their pipes become dots."""
    return str(x).replace("|", " · ")


def p(x: float) -> str:
    return "< 1e-04" if x < 1e-4 else f"{x:.3g}"


def section_confirm(lines: List[str], data: Dict[str, Any]) -> None:
    prim = _j(RUNS / "addendum" / "confirm" / "confirm_primary.json")
    lines += ["## Confirmatory addendum 1 — write- vs read-anchored retention signals (MSC validation + test)", "",
              f"Addendum SHA-256 `{prim['addendum_sha256']}`; {prim['facts']} facts, {prim['dialogues']} dialogues {prim['by_split']}.", "",
              "| ID | a − b | Endpoint | Effect [95% CI] | p | Holm p | Result |", "|---|---|---|---|---|---|---|"]
    for r in prim["primary_family"]:
        lines.append(f"| {r['id']} | {r['a']} − {r['b']} | {r['endpoint']} | {r['diff']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] "
                     f"| {p(r['p_value'])} | {p(r['holm_p'])} | {r['result']} |")
    lines += ["", "Per split:", ""]
    for sp, fam in prim["primary_by_split"].items():
        lines.append(f"- {sp}: " + "; ".join(f"{r['id']} {r['diff']:+.3f} ({r['result']})" for r in fam))
    for name, path in (("Horizon A (next session)", RUNS / "addendum" / "confirm" / "confirm_primary.json"),
                       ("Horizon B (two sessions ahead)", RUNS / "addendum" / "confirm" / "confirm_secondary_s4.json")):
        if not path.exists():
            continue
        sec = _j(path)["secondary"]
        lines += ["", f"{name}: AUROC (DeLong) and harmful deletion at keep 50 % by label", "",
                  "| Score | " + " | ".join(f"AUROC {l}" for l in sec) + " | " + " | ".join(f"harm {l}" for l in sec) + " |",
                  "|---|" + "---|" * (2 * len(sec))]
        for score in sec["lex_0.5"]["auroc"]:
            lines.append(f"| {score} | " + " | ".join(f"{sec[l]['auroc'][score]['delong']['auc']:.3f}" for l in sec) + " | "
                         + " | ".join(f"{sec[l]['harmful_deletion']['0.5'][score]['mean']:.3f}" for l in sec) + " |")
    data["confirm_primary"] = prim["primary_family"]


def section_addendum2(lines: List[str], data: Dict[str, Any]) -> None:
    path = RUNS / "addendum2" / "confirm_display_order.json"
    if not path.exists():
        return
    d = _j(path)
    lines += ["", "## Confirmatory addendum 2 — display order under conflicting memories (FactConsolidation 64k, 262k)", "",
              f"Addendum SHA-256 `{d['addendum_sha256']}`.", "",
              "| ID | Source | a | b | mean a | mean b | Effect [95% CI] | Holm p | Result |", "|---|---|---|---|---|---|---|---|---|"]
    for r in d["family"]:
        lines.append(f"| {r['id']} | {r['source']} | {' · '.join(r['a'])} | {' · '.join(r['b'])} | {r['mean_a']:.3f} | {r['mean_b']:.3f} "
                     f"| {r['diff']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | {p(r['holm_p'])} | {r['result']} |")
    data["addendum2"] = d["family"]


def section_audit(lines: List[str], data: Dict[str, Any]) -> None:
    lines += ["", "## Fourth-audit exploratory analyses (test artifacts; announced in the deviations log before running)", ""]
    h1 = _j(A / "h1.json")["sources"]
    lines += ["### S-H1. DARS − similarity in the static session (share of questions whose shown set is identical; recall and MRR differences)", "",
              "| Source | Method | Budget | identical set | by construction | recall diff [95% CI] | MRR diff |", "|---|---|---|---|---|---|---|"]
    for src, e in h1.items():
        for m, bs in e["budgets"].items():
            for b, v in bs.items():
                lines.append(f"| {src} | {m} | {b} | {v['share_identical_shown_set']:.2f} | {v['by_construction']} | "
                             f"{v['recall_diff']['diff']:+.3f} [{v['recall_diff']['ci_lo']:+.3f}, {v['recall_diff']['ci_hi']:+.3f}] | {v['mrr_diff']['diff']:+.3f} |")
    ar = _j(A / "alfworld_rank.json")["splits"]
    lines += ["", "### S-H3. ALFWorld location MRR by ranker and tie-break", "", "| Split | Ranker · tie-break | MRR [95% CI] | vs similarity |", "|---|---|---|---|"]
    for sp, e in ar.items():
        for k, v in e["mrr"].items():
            d = e["paired_vs_similarity"].get(k)
            lines.append(f"| {sp} | {esc(k)} | {ci(v)} | " + (f"{d['diff']:+.3f} (p {p(d['p_value'])})" if d else "—") + " |")
        c = e["paired_count_prior_vs_selected"]
        lines.append(f"| {sp} | count prior − selected | — | {c['diff']:+.3f} [{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}] |")
    msc = _j(A / "msc.json")["labels"]
    lines += ["", "### S-H4. MSC (original test): AUROC and harmful deletion at keep 50 % by label", "",
              "| Score | " + " | ".join(msc) + " | " + " | ".join(f"harm {l}" for l in msc) + " |", "|---|" + "---|" * (2 * len(msc))]
    for score in msc["lex_0.5"]["auroc"]:
        lines.append(f"| {score} | " + " | ".join(f"{msc[l]['auroc'][score]['delong']['auc']:.3f}" for l in msc) + " | "
                     + " | ".join(f"{msc[l]['harmful_deletion']['0.5'][score]['mean']:.3f}" for l in msc) + " |")
    ev = _j(A / "alfworld_evict.json")["splits"]
    lines += ["", "### S-H5a. ALFWorld eviction baselines (harmful deletion)", "", "| Split | Keep | " + " | ".join(ev["test_in"]["harmful_deletion"]["0.5"]) + " |",
              "|---|---|" + "---|" * len(ev["test_in"]["harmful_deletion"]["0.5"])]
    for sp, e in ev.items():
        for keep, pols in e["harmful_deletion"].items():
            lines.append(f"| {sp} | {keep} | " + " | ".join(f"{v['mean']:.3f}" for v in pols.values()) + " |")
    if (A / "alfworld_prior_art.json").exists():
        pa = _j(A / "alfworld_prior_art.json")["splits"]
        lines += ["", "ALFWorld prior-art eviction (re-run snapshot with texts):", "", "| Split | Keep | " + " | ".join(pa["test_in"]["0.5"]) + " |",
                  "|---|---|" + "---|" * len(pa["test_in"]["0.5"])]
        for sp, e in pa.items():
            for keep, pols in e.items():
                lines.append(f"| {sp} | {keep} | " + " | ".join(f"{v['mean']:.3f}" for v in pols.values()) + " |")
    lines += ["", "### S-H5b. LongMemEval eviction: registered set-level oracle, per-memory oracle, prior art, reader accuracy", "",
              "| Keep | Policy | harmful (set-level oracle) | harmful (per-memory oracle) | reader EM |", "|---|---|---|---|---|"]
    for keep, reg in (("0.25", "lme_budget25"), ("0.5", "lme_budget50"), ("0.75", "lme_budget75")):
        for pol in ("dars", "lru", "fifo", "lfu", "random", "generative_agents", "memorybank"):
            def mean(path: Path, key: str) -> str:
                if not path.exists():
                    return "—"
                s = _j(path)
                v = next(iter(s.values()))
                return f"{v[key]['mean']:.3f}" if key in v else "—"
            reg_path = T / "E2" / reg / (pol if pol != "random" else "random_s0") / "summary.json"
            if pol in ("generative_agents", "memorybank"):
                reg_path = A / "lme_evict_prior_art" / f"keep{keep}" / pol / "summary.json"
            lines.append(f"| {keep} | {pol} | {mean(reg_path, 'harmful_deletion')} | "
                         f"{mean(A / 'lme_evict_oracle_unit' / f'keep{keep}' / pol / 'summary.json', 'harmful_deletion')} | "
                         f"{mean(A / 'lme_evict_reader' / f'keep{keep}' / pol / 'summary.json', 'reader_substring_exact_match')} |")
    per = _j(A / "persistence.json")
    age = _j(A / "lme_age.json")
    unit = _j(A / "lme_rank_oracle_unit" / "summary.json")
    lines += ["", "### S-M. Mechanisms", "", f"- Need persistence: {json.dumps(per['mab'])}; MSC {json.dumps(per['msc'])}; ALFWorld {json.dumps(per['alfworld'])}",
              f"- LongMemEval evidence in newest half: {ci(age['share_of_evidence_in_newest_half'])}; mean age percentile {ci(age['mean_age_percentile_of_evidence (0 oldest, 1 newest)'])}",
              "- LongMemEval ranking MRR by feedback credit: " + "; ".join(f"{k}: {ci(v['mrr'])}" for k, v in sorted(unit.items()))]
    law = _j(T / "E12" / "feedback_law.json")
    lines += ["", "### S-F. Feedback reliability (MSC)", "", "| verdicts kept q | mean verdicts/memory | ε* AUROC vs recency | ε* eviction vs FIFO |", "|---|---|---|---|"]
    for lv in law["levels"]:
        lines.append(f"| {lv['q']} | {lv['mean_verdicts_per_memory_with_feedback']:.2f} | {lv['auroc_crosses_recency_at']} | {lv['harmful_crosses_fifo_at']} |")
    jc = law["judge_calibrated"]
    lines.append(f"\nJudge-calibrated asymmetric noise (fnr {jc['fnr']:.3f}, fpr {jc['fpr']:.3f}): AUROC {jc['auroc']:.3f}, harmful deletion {jc['harmful_deletion']:.3f}. "
                 f"Constant c of the (1−2ε)²·n law varied with CV {law['law_auroc_c_recency']['c_cv']:.2f} (AUROC) and {law['law_harmful_c_fifo']['c_cv']:.2f} (eviction): not supported.")
    tr = _j(A / "transfer.json")
    lines += ["", "### S-P1. Cross-domain transfer", "", "| Config | MSC AUROC | MSC harm | ALF in MRR | ALF in harm | ALF out MRR | ALF out harm |", "|---|---|---|---|---|---|---|"]
    for name in tr["msc"]:
        m, ai, ao = tr["msc"][name], tr["alfworld"]["test_in"][name], tr["alfworld"]["test_out"][name]
        lines.append(f"| {name} | {m['auroc']:.3f} | {m['harmful_deletion']['mean']:.3f} | {ai['location_mrr_score_only']['mean']:.3f} | "
                     f"{ai['harmful_deletion']['mean']:.3f} | {ao['location_mrr_score_only']['mean']:.3f} | {ao['harmful_deletion']['mean']:.3f} |")
    rob = _j(A / "robustness.json")
    lines += ["", "### S-P2. Robustness (Dirichlet, λ)", "", f"```\n{json.dumps(rob['msc'], indent=1)}\n{json.dumps(rob['alfworld'], indent=1)}\n```"]
    drift = _j(A / "dev_test_drift.json")["rows"]
    lines += ["", "### S-P4. Development-to-test drift", "", "| Selection | dev | test | test − dev |", "|---|---|---|---|"]
    for r in drift:
        lines.append(f"| {r['selection']} | {r['dev']:.3f} | {r['test']:.3f} | {r['test_minus_dev']:+.3f} |")
    emb = _j(A / "embedder_robustness.json")["sources"]
    lines += ["", "### S-L4. Stronger embedder (bge-small-en-v1.5) on identical memory units", "", "| Source | Metric | value [95% CI] |", "|---|---|---|"]
    for src, e in emb.items():
        for k, v in sorted(e["means"].items()):
            lines.append(f"| {src} | {esc(k)} | {ci(v)} |")
    mab = _j(A / "mab_published_comparison.json")
    lines += ["", "### S-P7. Harness against MemoryAgentBench's published gpt-4o-mini numbers", "", f"Published (Table 3): {json.dumps(mab['published']['rows'])}",
              f"Ours (test split, reader substring EM, mean of seeds): {json.dumps(mab['ours_test_split_reader_substring_em_mean_of_seeds'])}",
              "Differences in setting: " + "; ".join(mab["differences_in_setting"])]


def main() -> None:
    lines: List[str] = ["# Post-freeze analyses (generated by make_exploratory_tables.py)", ""]
    data: Dict[str, Any] = {}
    section_confirm(lines, data)
    section_addendum2(lines, data)
    section_audit(lines, data)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exploratory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "exploratory.json").write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    print(f"wrote {OUT / 'exploratory.md'} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
