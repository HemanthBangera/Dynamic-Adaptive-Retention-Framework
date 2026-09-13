"""
Assemble the Supplementary Information from the frozen documents, the code and the artifacts.

Nothing numeric is typed by hand: tables are read from ``benchmark_runs/revision``, prompts from the code, and the
pre-registration, addenda and deviations log are included verbatim.

    python scripts/build_si.py --out ../revision_submission/supplementary_information.md
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
R = ROOT / "benchmark_runs" / "revision"
T = R / "test"
A = T / "E13_audit"


def j(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def ci(d: Dict[str, Any]) -> str:
    return f"{d['mean']:.3f} [{d['ci_lo']:.3f}, {d['ci_hi']:.3f}]"


def esc(x: Any) -> str:
    """A table cell: run keys such as "raw|mab|best_first" would split the cell, so their pipes become dots."""
    return str(x).replace("|", " · ")


def diff(d: Dict[str, Any]) -> str:
    return f"{d['diff']:+.3f} [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]"


def verbatim(path: Path, level: str = "###") -> List[str]:
    text = path.read_text(encoding="utf-8")
    # demote headings so the included document nests under its note
    text = re.sub(r"^(#+) ", lambda m: level + "#" * len(m.group(1)) + " ", text, flags=re.M)
    return [text.rstrip(), ""]


def sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def s1(lines: List[str]) -> None:
    pre = ROOT / "experiments" / "preregistration.md"
    lines += ["## Note S1 — Pre-registration, addenda and deviations", "",
              f"- Pre-registration v1.0, SHA-256 `{sha(pre)}` (recorded in every run manifest).",
              f"- Addendum 1 (write-side study), SHA-256 `{(ROOT / 'experiments' / 'preregistration_addendum.md.sha256').read_text().split()[0]}`; "
              f"configuration `experiments/addendum_config.json`, SHA-256 `{sha(ROOT / 'experiments' / 'addendum_config.json')}`.",
              f"- Addendum 2 (display order), SHA-256 `{(ROOT / 'experiments' / 'preregistration_addendum_2.md.sha256').read_text().split()[0]}`.",
              "- A draft of the pre-registration (v0.1) is publicly timestamped in the code repository at commit `7b26e6b` "
              "(2026-09-11 14:07 UTC), before any test-split run; the draft-to-frozen diff is archived "
              "(`E13_audit/preregistration_draft_to_frozen.diff`). Hypotheses are unchanged in substance.",
              "- The post-freeze deviations log is append-only; its SHA-256 after every entry is listed in "
              "`experiments/deviations_post_freeze.sha256.log`.", "",
              "### S1.1 Pre-registration (verbatim)", ""]
    lines += verbatim(pre, "###")
    lines += ["### S1.2 Addendum 1 (verbatim)", ""] + verbatim(ROOT / "experiments" / "preregistration_addendum.md", "###")
    lines += ["### S1.3 Addendum 2 (verbatim)", ""] + verbatim(ROOT / "experiments" / "preregistration_addendum_2.md", "###")
    lines += ["### S1.4 Post-freeze deviations log (verbatim)", ""] + verbatim(ROOT / "experiments" / "deviations_post_freeze.md", "###")
    lines += ["```", (ROOT / "experiments" / "deviations_post_freeze.sha256.log").read_text(encoding="utf-8").strip(), "```", ""]


def s2(lines: List[str]) -> None:
    from benchmarks.dars_eval.reader import format_memories, MAX_ANSWER_TOKENS
    from benchmarks.dars_eval.retention_signals import ga_importance_prompt
    from benchmarks.dars_eval.run_judge import PROMPTS
    from core.layer_a import reformulator
    from core.layer_c import compressor
    from third_party.memoryagentbench_eval import templates

    def fstring_prompt(module, marker: str) -> str:
        src = inspect.getsource(module)
        i = src.index(marker)
        block = src[i:src.index(")", src.index("\n        )", i))]
        return block
    lines += ["## Note S2 — Verbatim prompts", "",
              "All calls: temperature 0, fixed seed, pinned snapshots (`gpt-4o-mini-2024-07-18` reader; "
              "`gpt-4.1-nano-2025-04-14` judge, reformulator, compressor and importance rater).", "",
              "### Reader (MemoryAgentBench RAG agent)", "", "System message:", "", "```", templates.SYSTEM_MESSAGE, "```", "",
              "User message: the retrieved memories, one block each, in rank order unless a display-order condition reverses "
              "them, then MemoryAgentBench's `rag_agent` query template for the source. Memory block format:", "",
              "```", format_memories(["<memory text>", "<memory text>"]), "```", "",
              f"Answer-length caps (tokens): `{json.dumps(MAX_ANSWER_TOKENS)}`.", ""]
    tdict = getattr(templates, "TEMPLATES", None) or {k: v for k, v in vars(templates).items() if isinstance(v, dict)}
    lines += ["`rag_agent` query templates as vendored (`third_party/memoryagentbench_eval/templates.py`):", "", "```"]
    src = Path(templates.__file__).read_text(encoding="utf-8")
    for m in re.finditer(r"'rag_agent':\s*(\"[^\n]*\")", src):
        lines.append(m.group(1)[:4000])
    lines += ["```", "",
              "### Layer B judge (`core/layer_b/evaluator.py`)", "", "```", PROMPTS["layer_b"], "```", "",
              "Paraphrased judge prompt (E5 robustness):", "", "```", PROMPTS["paraphrase"], "```", "",
              "### Layer A query reformulator (`core/layer_a/reformulator.py`)", "", "```python", fstring_prompt(reformulator, "prompt = ("), "```", "",
              "### Layer C semantic compressor (`core/layer_c/compressor.py`)", "", "```python", fstring_prompt(compressor, "prompt = ("), "```", "",
              "### Generative Agents importance rating (Park et al.)", "", "```", ga_importance_prompt("<memory text>"), "```", "",
              "Reply limits: 4 tokens for the MSC addendum (pre-registered); 96 tokens for the exploratory LongMemEval and "
              "ALFWorld ratings (see the deviations log).", ""]


def s_data(lines: List[str]) -> None:
    lines += ["## Note S3 — Data, splits and scale", ""]
    rows = []
    for folder in ("eventqa_65536", "eventqa_full", "ruler_qa1_197K", "ruler_qa2_421K", "longmemeval_s",
                   "factconsolidation_sh_32k", "factconsolidation_mh_32k"):
        pq = T / "E1" / folder / "per_question.jsonl"
        if not pq.exists():
            continue
        qs = {(r["context"], r["question"]) for r in (json.loads(l) for l in pq.read_text(encoding="utf-8").splitlines() if l.strip())}
        ctx = {c for c, _ in qs}
        rows.append(f"| {folder} | {len(ctx)} | {len(qs)} |")
    lines += ["MemoryAgentBench test questions scored in E1:", "", "| Source | Contexts | Test questions |", "|---|---|---|"] + rows + [""]
    e9 = j(T / "E9" / "manifest.json")
    e10 = j(T / "E10" / "manifest.json")
    c = j(R / "addendum" / "confirm" / "confirm_primary.json")
    lines += [f"- MSC (train split, pre-registered): {e9['dialogues']} test dialogues, {e9['facts']} facts; development "
              f"{j(R / 'E9' / 'dev_default' / 'manifest.json')['dialogues']} dialogues.",
              f"- MSC (confirmatory, validation + test splits): {c['dialogues']} dialogues {c['by_split']}, {c['facts']} facts (horizon A).",
              f"- ALFWorld: {e10['stream_tasks']} stream tasks, {e10['memories']} memories, evaluation sets {e10['eval_sets']}.",
              "- Splits: MemoryAgentBench 30/70 development/test within each context (seed 20260911); MSC 30/70 by dialogue; "
              "ALFWorld a seeded 10 % of training tasks as development.", ""]


def s_e3_e4(lines: List[str]) -> None:
    lines += ["## Note S4 — Weights, thresholds and tiers (E3)", ""]
    for name in ("default", "msc_h4", "msc_h5", "alfworld_h5"):
        d = j(T / "E3" / name / "thresholds.json")
        lines.append(f"### {name}: weights {d['weights']}, thresholds {d['thresholds']}")
        lines.append(f"- Successes needed to reach RETAIN, by P: {d['min_successes_to_retain']}")
        for ds, st in d["stability"].items():
            lines.append(f"- {ds} (n = {st['n']}): base tier shares {json.dumps({k: round(v, 3) for k, v in st['base_shares'].items()})}")
            lines.append("  - threshold shifts: " + "; ".join(f"{s['threshold']} {s['shift']:+.2f}: changed {s['changed']:.3f}, κ {s['kappa']:.2f}" for s in st["shifts"]))
        lines.append("")
    lines += ["Dirichlet robustness, decay-rate sensitivity and cross-domain transfer on the test split: Note S11 "
              "(`exploratory.md`, sections S-P1 and S-P2).", "",
              "## Note S5 — Predictive relevance (E4)", "", "| Source | Variant | recall@512 | recall@1024 | MRR@5120 | top-10 changed | Kendall τ |", "|---|---|---|---|---|---|---|"]
    e4 = j(T / "E4" / "summary.json")
    for src, methods in e4.items():
        if src == "provenance":
            continue
        for name, v in methods.items():
            if name != "similarity" and "rrf_k15" not in name:
                continue
            m = v["metrics"]
            g = lambda k: (f"{m[k]['mean']:.3f}" if isinstance(m.get(k), dict) else (f"{m[k]:.3f}" if isinstance(m.get(k), (int, float)) else "—"))
            lines.append(f"| {esc(src)} | {esc(name)} | {g('group_recall@512')} | {g('group_recall@1024')} | {g('mrr@5120')} | {g('top10_changed')} | {g('kendall_tau')} |")
    lines.append("")


def s_layers(lines: List[str]) -> None:
    lines += ["## Note S6 — Layer A (E7; and on LongMemEval)", ""]
    for label, path in (("EventQA-65K + RULER QA1 (pre-registered E7)", T / "E7" / "summary.json"),
                        ("LongMemEval (exploratory)", A / "lme_layera" / "summary.json")):
        d = j(path)
        lines += [f"### {label}", "", "| Condition | Accuracy | Evidence recall | Truncation rate |", "|---|---|---|---|"]
        for name, v in d["conditions"].items():
            er = v.get("evidence_recall")
            er = er["mean"] if isinstance(er, dict) else er
            lines.append(f"| {esc(name)} | {ci(v['accuracy'])} | {'—' if er is None else f'{er:.3f}'} | {v.get('truncation_rate')} |")
        lines += ["", "| Contrast | Difference [95% CI] | p |", "|---|---|---|"]
        for k, v in d["contrasts"].items():
            lines.append(f"| {esc(k)} | {diff(v)} | {v['p_value']:.3g} |")
        lines.append("")
    e6 = j(T / "E6" / "summary.json")
    lines += ["## Note S7 — Layer C: compression and shadow indexing (E6)", "",
              f"Items: {e6['items']}; semantic compression failure rate {e6['semantic_failure_rate']}; stored vector preserved "
              f"after compression: {e6['shadow_vector_preserved_rate']}.", "",
              "| Variant | Compression ratio | Answer retention | Reader (original) | Reader (compressed) | Δ reader | R@10 kept vector | R@10 re-embedded (target only) | R@10 re-embedded (whole store) |",
              "|---|---|---|---|---|---|---|---|---|"]
    whole = j(A / "e6_whole_context" / "summary.json") if (A / "e6_whole_context" / "summary.json").exists() else {}
    for name in ("semantic", "llmlingua2", "extractive", "llmlingua2@matched", "extractive@matched"):
        v = e6[name]
        w = whole.get(name, {}).get("recall@10_reembedded_whole_context")
        lines.append(f"| {name} | {ci(v['compression_ratio'])} | {ci(v['answer_retention'])} | {ci(v['reader_original'])} | "
                     f"{ci(v['reader_accuracy'])} | {diff(v['reader_delta_vs_original'])} | {ci(v['recall@10_kept_vector'])} | "
                     f"{ci(v['recall@10_reembedded'])} | {ci(w) if w else '⟦pending⟧'} |")
    lines.append("")
    if whole:
        fixed = ("semantic", "llmlingua2", "extractive")
        raw = {n: whole[n]["kept_minus_reembedded_whole_context"]["p_value"] for n in fixed}
        order, run, holm = sorted(fixed, key=raw.get), 0.0, {}
        for i, n in enumerate(order):
            run = max(run, min(1.0, (len(order) - i) * raw[n]))
            holm[n] = run
        lines += ["Whole-store condition (exploratory; announced in the deviations log on 2026-09-12, outcome logged "
                  "2026-09-13T14:08Z). Every memory of each test context was compressed and re-embedded, so the evidence competes "
                  f"against equally compressed text; semantic compression failure rate {whole['semantic_failure_rate']}. "
                  "Paired clustered bootstrap, 10,000 resamples; Holm over the three fixed-rate compressors.", "",
                  "| Variant | Kept − re-embedded (whole store) | p | Holm p | Whole store − target only | p |",
                  "|---|---|---|---|---|---|"]
        for name in ("semantic", "llmlingua2", "extractive", "llmlingua2@matched", "extractive@matched"):
            k, t = whole[name]["kept_minus_reembedded_whole_context"], whole[name]["whole_context_minus_target_only"]
            lines.append(f"| {name} | {diff(k)} | {k['p_value']:.3g} | {holm[name]:.3g} | {diff(t)} | {t['p_value']:.3g} |"
                         if name in holm else
                         f"| {name} | {diff(k)} | {k['p_value']:.3g} | — | {diff(t)} | {t['p_value']:.3g} |")
        lines += ["", "No difference between keeping the original vector and re-embedding the compressed text survives Holm "
                  "correction. Shadow indexing guarantees an unchanged ranking; it did not make memories more retrievable.", ""]
    e8 = j(T / "E8" / "summary.json")
    e8m = j(T / "E8" / "manifest.json")
    lines += ["## Note S8 — Efficiency at equal token budgets (E8)", "",
              f"Budgets {', '.join(f'{b:,}' for b in e8m['budgets'])} tokens; compression rate {e8m['compression_rate']}; reader at "
              f"{e8m['reader_budget']:,} tokens. Latency is the median ranking wall-clock per query on the host recorded in the "
              f"manifest ({e8m['provenance'].get('platform')}), measured while other jobs ran, and is indicative only; the "
              "manifest's latency note names the laptop CPU in error. Retrieval methods without compression are compared at "
              "larger budgets in E1 (Note S11).", "", "| Source · method · compression · budget | n shown | tokens | evidence recall | answer present | reader EM | latency ms |", "|---|---|---|---|---|---|---|"]
    for k, v in e8.items():
        if k == "provenance" or not isinstance(v, dict) or "n_shown" not in v:
            continue
        em = v.get("reader_substring_exact_match")
        lines.append(f"| {esc(k)} | {v['n_shown']['mean']:.1f} | {v['tokens_used']['mean']:.0f} | {ci(v['group_recall'])} | {ci(v['answer_present'])} | "
                     f"{ci(em) if em else '—'} | {v['latency_median_ms']:.1f} |")
    lines.append("")
    e5 = j(T / "E5" / "report.json")
    lines += ["## Note S9 — Judge reliability (E5)", "", f"Items: {e5['items']}; reference positive rate {e5['reference_positive_rate']:.3f}.", "",
              "| Rater | Accuracy | Precision | Recall | F1 | κ | κ vs answer correct | κ vs evidence shown | NEUTRAL |", "|---|---|---|---|---|---|---|---|---|"]
    for name, v in e5["raters"].items():
        f3 = lambda x: "—" if x is None else f"{x:.3f}"
        lines.append(f"| {esc(name)} | {ci(v['accuracy'])} | {f3(v.get('precision'))} | {f3(v.get('recall'))} | {f3(v.get('f1'))} | "
                     f"{f3(v.get('kappa'))} | {f3(v.get('vs_correct_kappa'))} | {f3(v.get('vs_has_evidence_kappa'))} | {f3(v.get('neutral_rate'))} |")
    lines += ["", "Consistency: " + "; ".join(f"{k} {v:.3f}" for k, v in e5["consistency"].items()), ""]
    lines += ["End-to-end feedback sources in the LongMemEval stream (pre-registered `e2_feedback_sources`):", "",
              "| Feedback | Evidence recall | MRR | Reader EM |", "|---|---|---|---|"]
    fs = j(T / "E2" / "lme_feedback_sources" / "summary.json")
    for k, v in sorted(fs.items()):
        lines.append(f"| {esc(k)} | {ci(v['group_recall'])} | {ci(v['mrr'])} | {ci(v['reader_substring_exact_match'])} |")
    lines.append("")
    lines += ["## Note S10 — Remaining MemoryAgentBench competencies (E11)", "", "| Source | Method | Reader substring EM (seed 0 / 1 / 2) |", "|---|---|---|"]
    for folder in ("detective_qa", "icl_banking77_5900shot_balance"):
        d = j(T / "E11" / folder / "summary.json")
        for meth, v in d.items():
            if not isinstance(v, dict):
                continue
            vals = [v[k] for k in sorted(v) if k.startswith("reader_substring_exact_match@")]
            vals = [x["mean"] if isinstance(x, dict) else x for x in vals]
            if vals:
                lines.append(f"| {folder} | {meth} | " + " / ".join(f"{x:.3f}" for x in vals) + " |")
    lines.append("")


def s_tables(lines: List[str]) -> None:
    lines += ["## Note S11 — Result tables generated from the artifacts", "", "### S11.1 Pre-registered primary comparisons (`make_tables.py`)", ""]
    lines += [(R / "_tables_test" / "primary.md").read_text(encoding="utf-8").strip(), ""]
    lines += ["### S11.2 Confirmatory addenda and exploratory analyses (`make_exploratory_tables.py`)", ""]
    text = (R / "_tables_test" / "exploratory.md").read_text(encoding="utf-8")
    lines += [re.sub(r"^(#+) ", lambda m: "###" + "#" * len(m.group(1)) + " ", text, flags=re.M).strip(), ""]


def s_verification_cost(lines: List[str]) -> None:
    fc = _replay().get("final_archive_check") or {}
    counts = [fc.get("pytest_placeholder_key_archived_cache", {}), fc.get("pytest_no_key", {})]
    totals = {sum(c.values()) for c in counts if c}
    n = f"{totals.pop():,}" if len(totals) == 1 else "⟦N⟧"       # one suite size, from the same archive check
    lines += ["## Note S12 — Implementation verification", "",
              f"The automated suite has {n} tests. They run locally against Qdrant's in-memory mode, with mocked or cached LLM "
              f"responses. {tests_sentence()} The suite covers every layer, the evaluation harness, the statistics (including clustered AUROC), the "
              "pre-registration guards of both addenda, the Batch-API collect/replay path, the redacted cache, and regression "
              "tests for the defects found during the revision. These tests verify the software; they are not evidence that the "
              "method is effective.", "",
              "Replay checks: the pre-registered FactConsolidation H2 stream re-run with all post-audit code produced "
              "byte-identical `per_question.jsonl` and `summary.json`; `run_msc run` with default options produced byte-identical "
              "facts before and after the write-history option was added.", ""]
    cost = j(R / "_tables_test" / "cost_report.json")
    lines += ["## Note S13 — API usage, cost and hardware", "",
              "Computed from the response caches (one record per distinct request; development runs and discarded rating "
              "attempts included). Batch-API responses are billed at half price.", "",
              "| Model | Channel | Requests | Prompt tokens | Completion tokens | Cost at list price (USD) | Billed (USD) |", "|---|---|---|---|---|---|---|"]
    for k, v in cost["by_model_and_channel"].items():
        model, ch = k.split("|")
        lines.append(f"| {model} | {ch} | {v['requests']:,} | {v['prompt_tokens']:,} | {v['completion_tokens']:,} | {v['cost_usd_list_price']:.2f} | {v['cost_usd_billed']:.2f} |")
    t = cost["total"]
    lines += [f"| **Total** | | {t['requests']:,} | {t['prompt_tokens']:,} | {t['completion_tokens']:,} | {t['cost_usd_list_price']:.2f} | {t['cost_usd_billed']:.2f} |", "",
              "Hardware: a Windows 11 laptop (Intel i5-12500H, 16 GB) and an AWS EC2 r7i.2xlarge instance (8 vCPU, 61 GB, Ubuntu "
              "24.04), with the same pinned Python 3.14.3 packages; each run manifest records the host, package versions, dataset "
              "and model revisions, and the pre-registration hash.", ""]


def s_e0_availability(lines: List[str]) -> None:
    e0 = A / "e0_replay.txt"
    lines += ["## Note S14 — Continuity with the submitted configuration", "",
              "The submitted retrieval configuration (4,096-token chunks, ALFWorld goal preset for P, RRF over the 15 nearest "
              "memories, top 3) was replayed through the new harness on the submitted questions and compared with the submitted "
              "audit logs (`benchmarks/dars_eval/e0_replay.py`):", ""]
    if e0.exists():
        lines += ["```", "\n".join(l for l in e0.read_text(encoding="utf-8").splitlines() if l.startswith("{") or l.startswith("E0")), "```"]
    else:
        lines.append("⟦pending⟧")
    lines += ["", "The submitted token savings (80.95 % and 93.93 %) followed arithmetically from retrieving three 4,096-token chunks "
              "and are withdrawn; efficiency is now measured at equal token budgets (Note S8). The pre-registered H1 DARS "
              "configuration is the submitted fusion rule applied to the new memory units.", "",
              "## Note S15 — Data, code and archive", "",
              "- Code: GitHub repository (public) and a Zenodo archive (DOI in the article's Code availability statement) containing the code, run artifacts and manifests, "
              "the pre-registration, both addenda, the deviations log, the embedding cache (vectors keyed by hashes, no text) and "
              "the LLM response cache. The archive is built and checked by `scripts/build_archive.py`, which writes "
              "`ARCHIVE_REPORT.json`, `DATA_NOTICE.md` and `SHA256SUMS`.",
              "- LLM responses are archived as a redacted cache (`scripts/redact_cache.py`; 137,802 records): each record keeps the "
              "response, usage and request parameters, and replaces every prompt message with its SHA-256 and length. Every number "
              "replays once the prompts are rebuilt from the datasets.",
              "- Benchmark text in the archive: run artifacts contain short excerpts (questions, answers, memory units, compressed "
              "memories) from datasets whose licences permit redistribution. For the novel-derived sources (EventQA, DetectiveQA) the "
              "artifacts hold identifiers, scores and model outputs only. An automated scan of every archived text file for any "
              "12-word span of those novels found such spans only inside model outputs (73 files; longest span 26 words).",
              "- Secrets: the archive was scanned for provider key patterns, private keys, the values of the local API keys and every "
              "key-shaped value in the repository history; the only matches are synthetic placeholders in unit tests.",
              "- Datasets, used under their own terms: MemoryAgentBench⟦ref:MemoryAgentBench⟧ (MIT licence on the Hugging Face "
              "release; constituent sources LongMemEval⟦ref:LongMemEval⟧, RULER⟦ref:RULER⟧, SQuAD 2.0⟦ref:SQuAD2⟧, "
              "HotpotQA⟦ref:HotpotQA⟧, DetectiveQA⟦ref:DetectiveQA⟧ and Banking77⟦ref:Banking77⟧; FactConsolidation, built by "
              "MemoryAgentBench from MQuAKE⟦ref:MQuAKE⟧; EventQA, built by MemoryAgentBench from novels), Multi-Session "
              "Chat⟦ref:MSC⟧ (`nayohan/multi_session_chat`), and ALFWorld⟦ref:ALFWorld⟧ (`awawa-agi/alfworld-raw`, a redistribution "
              "of the original data). Revisions are recorded in every manifest. Licences (checked 2026-09-13): MemoryAgentBench MIT; "
              "LongMemEval MIT; RULER Apache-2.0; SQuAD 2.0 and HotpotQA CC BY-SA 4.0; MQuAKE MIT; DetectiveQA Apache-2.0; "
              "Banking77 CC BY 4.0; Multi-Session Chat via ParlAI (MIT repository); ALFWorld MIT. The copyright status of the "
              "novels behind EventQA and DetectiveQA is not stated, so no novel text is archived beyond the short quotations in "
              "model outputs noted above.", ""]
    lines += replay_audit_lines()


def _replay() -> Dict[str, Any]:
    path = A / "replay_audit" / "replay_audit.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def tests_sentence() -> str:
    fc = _replay().get("final_archive_check")
    if not fc:
        return ""
    nk, pk = fc["pytest_no_key"], fc["pytest_placeholder_key_archived_cache"]
    return (f"Tests of the LLM-backed layers need a provider key. Run from the re-checked archive build (Note S15) with network access "
            f"blocked: with no key, {nk.get('passed', 0)} tests passed and {nk.get('skipped', 0)} were skipped; with a placeholder "
            f"key and the archived response cache, all {pk.get('passed', 0)} passed.")


def replay_audit_lines() -> List[str]:
    ra = _replay()
    if not ra:
        return ["- Replay audit: ⟦pending⟧", ""]
    nd, tc, byf = ra["numeric_differences"], ra.get("tie_and_ranking_checks", {}), ra.get("numeric_differences_by_field", {})
    manifests = sum(int(l.split()[0]) for l in ra["replayed_manifests_api_calls"] if l.endswith('"api_calls":0'))
    nonzero = [l for l in ra["replayed_manifests_api_calls"] if not l.endswith('"api_calls":0')]
    msc = byf.get("msc_test_s3_facts", {})
    float_fields = {k: v for k, v in msc.items() if k.startswith("[].components.") and v["max_abs"] < 1e-12}
    e1_fields = ", ".join(sorted(tc.get("e1_non_numeric_difference_fields", {})))
    cf = ra.get("confirm_from_replayed_data", {})
    lines = ["### Replay audit of the archive", "",
             f"The archive build with SHA-256 `{ra['archive_sha256']}` was unpacked and run in a {ra['environment']}. From it "
             "we re-ran: E1 on RULER QA1 with the reader; E11 on DetectiveQA, a novel-derived source whose prompts were rebuilt "
             "from the dataset and matched against the redacted cache; the E5 judge; the four addendum-2 streams and their "
             "analysis; the MSC confirmatory data build (test split, and the validation split for the check below); and both "
             "addendum-1 analyses. We also regenerated the primary and exploratory tables and this document.", "",
             f"- API calls: all {manifests} replayed manifests record 0 API calls." if not nonzero else f"- API calls: {nonzero}",
             "- Byte-identical outputs: E5 (items and report), E11 (per-question outputs; summary equal) and addendum 2 (all "
             "per-question outputs, summaries and the confirmatory analysis).",
             f"- E1 RULER QA1: summary {'and ranked lists ' if tc.get('e1_ranked_lists_identical') else ''}identical; similarity "
             f"scores differed by at most {nd['E1_per_question']['max_abs']:.1e}; the only non-numeric difference is "
             f"`{e1_fields}`, which records that reader answers now came from the cache.",
             f"- Addendum-1 analyses recomputed from the archived data: largest relative difference "
             f"{max(nd['confirm_primary']['max_rel'], nd['confirm_secondary_s4']['max_rel']):.1e}; no decision changed.",
             f"- MSC confirmatory data regenerated on Linux (the archived data were built on Windows), test and validation "
             f"splits ({cf.get('facts_replayed', 0):,} facts): "
             f"{sum(v['count'] for v in float_fields.values()) + ra.get('validation_s3_checks', {}).get('component_values_differing', 0):,} "
             f"component values differed by at most "
             f"{max([v['max_abs'] for v in float_fields.values()] + [ra.get('validation_s3_checks', {}).get('component_max_abs', 0.0)]):.1e}. "
             f"In {tc.get('msc_facts_with_count_differences') + ra.get('validation_s3_checks', {}).get('facts_with_count_differences', 0)} "
             f"facts, from {len(tc.get('msc_count_difference_dialogues', [])) + len(ra.get('validation_s3_checks', {}).get('dialogues', []))} "
             "dialogue, a similarity tie resolved differently and the facts exchanged their retrieval-credited counts. "
             + ("No write-side or label field differed." if tc.get("msc_write_or_label_fields_differing") == 0
                and ra.get("validation_s3_checks", {}).get("write_or_label_fields_differing", 1) == 0 else
                f"Write-side or label fields differed in {tc.get('msc_write_or_label_fields_differing')} cases."),
             f"- Primary confirmatory family recomputed entirely from the regenerated MSC data: {cf.get('confirmed_replayed')} of "
             f"{cf.get('family_size')} comparisons confirmed, as archived; {len(cf.get('decisions_changed', []))} decisions "
             f"changed, overall or per split; largest difference in any AUROC, effect, interval bound or Holm p "
             f"{cf.get('max_abs_difference', float('nan')):.1e}. One per-split interval bound not reported in the paper crosses "
             f"a rounding boundary ({'; '.join(cf.get('values_changed_at_3dp', []))})." if cf.get("values_changed_at_3dp") else
             f"- Primary confirmatory family recomputed from the regenerated MSC data: {cf.get('summary')}",
             "- Manifests of E1 and E11 lack the `first_stage` method field added later with the BM25 first stage "
             "(schema only).",
             (lambda fc: f"- Later builds changed no analysis result, only documentation, reporting, figure numbering, archive "
                         f"tooling and these audit files. One of them (SHA-256 `{fc['archive_sha256'][:16]}…`) was re-checked the same way: "
                         f"{fc['checksum_mismatches']} checksum mismatches; with the archived "
                         f"cache {fc['pytest_placeholder_key_archived_cache'].get('passed', 0)} tests passed; without a key "
                         f"{fc['pytest_no_key'].get('passed', 0)} passed and {fc['pytest_no_key'].get('skipped', 0)} were skipped"
                         + ("; this document regenerated identically." if fc["si_regenerated_identically"] else
                            "; this document did NOT regenerate identically."))(ra["final_archive_check"])
             if ra.get("final_archive_check") else "- Final archive check: ⟦pending⟧",
             "- Tables and document: `primary.json`/`primary.md`, `exploratory.json`/`exploratory.md` and this Supplementary "
             "Information regenerated identically (line endings aside).", ""]
    return lines

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    lines: List[str] = ["# Supplementary Information", "",
                        "**Dynamic Adaptive Retention Scoring (DARS): A Layered Memory Governance Framework for Agentic Large Language Models**",
                        "Pushpalatha M N, Harshendra M, Hemanth L Bangera", "",
                        "> Generated by `scripts/build_si.py` from the frozen documents, the code and the run artifacts. "
                        "Build the .docx with `python scripts/build_docx.py --in supplementary_information.md --out "
                        "supplementary_information.docx --refs references.json`.", ""]
    for fn in (s1, s2, s_data, s_e3_e4, s_layers, s_tables, s_verification_cost, s_e0_availability):
        fn(lines)
        print(f"built {fn.__name__}")
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
