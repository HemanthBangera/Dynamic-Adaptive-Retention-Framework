"""
E5 — Layer B judge reliability against deterministic reference labels (H6).

Inputs are E1 reader runs (``per_question.jsonl`` with ``reader`` outputs).  For
each (question, method) the memories shown to the reader are rebuilt from the
ranked list at the reader budget, and three kinds of verdicts are compared with
a reference label derived from the benchmarks' own gold annotations:

reference   success = the reader's answer is correct (MemoryAgentBench substring
            exact match) AND the shown memories contain gold evidence (RULER gold
            paragraph, LongMemEval has_answer turn, FactConsolidation newest fact).
            Per memory: the memory is part of the gold evidence.
judges      Layer B ``SuccessEvaluator`` prompt on gpt-4.1-nano (the system's judge)
            and gpt-4o-mini; each run twice (seeds 0, 1) and with one paraphrased
            prompt, to measure accuracy, agreement with the reference (Cohen's κ),
            test–retest and inter-judge agreement, and the NEUTRAL rate.
lexical     the deterministic feedback mode: a memory is credited when the token F1
            between the reader's answer and the memory is at least τ; the turn-level
            verdict is "any memory credited".  No LLM.

Usage
-----
python -m benchmarks.dars_eval.run_judge --inputs benchmark_runs/revision/E1/dev/ruler_qa1_197K_dev \
    --out benchmark_runs/revision/E5/dev
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tiktoken

from benchmarks.dars_eval.datasets import load_contexts
from benchmarks.dars_eval.memory_units import WindowGuard
from benchmarks.dars_eval.provenance import collect_provenance
from benchmarks.dars_eval.retrieval_eval import cut_to_budget
from benchmarks.dars_eval.stats import cluster_bootstrap_mean, cohen_kappa
from core.layer_b.evaluator import PROMPT_TEMPLATE, parse_verdict
from third_party.memoryagentbench_eval.eval_other_utils import f1_score

logger = logging.getLogger(__name__)
ENC = tiktoken.encoding_for_model("gpt-4o-mini")

# The Layer B prompt, imported from the system itself so that E5 validates exactly
# what Layer B sends, plus one paraphrase for robustness.  Memories are joined with
# newlines, as in ``LearningEngine.process_feedback_loop``; verdicts are parsed by
# the system's own ``parse_verdict``.
PROMPTS = {
    "layer_b": PROMPT_TEMPLATE,
    "paraphrase": (
        "Below are a question, an assistant's answer, and the memory snippets the assistant was given.\n"
        "Question: {query}\n"
        "Answer: {response}\n"
        "Memory snippets:\n{memories}\n"
        "Was the answer correct AND supported by the memory snippets? "
        "Reply with exactly one word: YES or NO."
    ),
}


def lexical_turn_verdict(answer: str, memories: Sequence[str], tau: float) -> bool:
    return any(f1_score(answer or "", m)[0] >= tau for m in memories)


def build_items(run_dir: Path, guard: WindowGuard) -> List[Dict[str, Any]]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    source, budget = manifest["source"], manifest["reader_budget"]
    contexts = {c.index: c for c in load_contexts(source, guard)}
    items = []
    for line in (run_dir / "per_question.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if not rec.get("reader"):
            continue
        ctx = contexts[rec["context"]]
        q = rec["question"]
        if not ctx.evidence[q]:
            continue                                  # unlabelled (e.g. LongMemEval abstention)
        unit_tokens = [len(ENC.encode(u.text)) for u in ctx.units]
        shown = cut_to_budget(rec["ranked"], unit_tokens, budget)
        evidence_units = {u for g in ctx.evidence[q] for u in g}
        for r in rec["reader"]:
            correct = r["metrics"].get("substring_exact_match", 0.0) >= 1.0
            has_evidence = any(u in evidence_units for u in shown)
            items.append({
                "source": source, "context": ctx.index, "question": q, "method": rec["method"],
                "seed": r["seed"], "query": ctx.formatted_queries[q], "answer": r["output"],
                "parsed": r.get("parsed_output") or r["output"],
                "memories": [ctx.units[u].text for u in shown],
                "memory_is_evidence": [u in evidence_units for u in shown],
                "reference": bool(correct and has_evidence),
                "correct": bool(correct), "has_evidence": bool(has_evidence),
            })
    return items


async def judge_items(items: List[Dict[str, Any]], models: Sequence[str], seeds: Sequence[int],
                      concurrency: int) -> Dict[str, Any]:
    from core.llm_transport import OpenAITransport

    transports = {m: OpenAITransport(m, max_tokens=5, max_concurrency=concurrency) for m in models}

    async def one(item, model, prompt_name, seed):
        prompt = PROMPTS[prompt_name].format(
            query=item["query"], response=item["answer"], memories="\n".join(item["memories"]))
        out = await transports[model].complete(prompt, seed=seed)
        return parse_verdict(out["text"])

    jobs = []
    keys = []
    for i, item in enumerate(items):
        for model in models:
            for prompt_name in PROMPTS:
                for seed in (seeds if prompt_name == "layer_b" else seeds[:1]):
                    jobs.append(one(item, model, prompt_name, seed))
                    keys.append((i, model, prompt_name, seed))
    verdicts = await asyncio.gather(*jobs)
    for (i, model, prompt_name, seed), v in zip(keys, verdicts):
        items[i].setdefault("judge", {})[f"{model}|{prompt_name}|s{seed}"] = v
    return {m: t.ledger.as_dict() for m, t in transports.items()}


def agreement_report(items: List[Dict[str, Any]], taus: Sequence[float], n_boot: int) -> Dict[str, Any]:
    ref = np.array([it["reference"] for it in items])
    clusters = [f"{it['source']}:{it['context']}" for it in items]
    report: Dict[str, Any] = {"items": len(items), "reference_positive_rate": float(ref.mean()), "raters": {}}

    def summarise(pred: np.ndarray, neutral: Optional[np.ndarray] = None) -> Dict[str, Any]:
        mask = np.ones(len(pred), dtype=bool) if neutral is None else ~neutral
        p, r = pred[mask], ref[mask]
        tp = int(np.sum(p & r)); fp = int(np.sum(p & ~r)); fn = int(np.sum(~p & r))
        acc = cluster_bootstrap_mean((p == r).astype(float), [c for c, m in zip(clusters, mask) if m], n_boot=n_boot)
        return {
            "n": int(mask.sum()),
            "neutral_rate": float(1 - mask.mean()),
            "accuracy": acc.as_dict(),
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
            "kappa": cohen_kappa(p, r) if len(set(p.tolist())) > 1 or len(set(r.tolist())) > 1 else None,
        }

    if items and "judge" in items[0]:
        for key in items[0]["judge"]:
            v = np.array([it["judge"][key] for it in items])
            report["raters"][key] = summarise(v == "YES", neutral=(v == "NEUTRAL"))
        models = sorted({k.split("|")[0] for k in items[0]["judge"]})
        consistency = {}
        for m in models:
            a = np.array([it["judge"][f"{m}|layer_b|s0"] == "YES" for it in items])
            if f"{m}|layer_b|s1" in items[0]["judge"]:
                b = np.array([it["judge"][f"{m}|layer_b|s1"] == "YES" for it in items])
                consistency[f"{m}: test-retest kappa"] = cohen_kappa(a, b)
            c = np.array([it["judge"][f"{m}|paraphrase|s0"] == "YES" for it in items])
            consistency[f"{m}: prompt-paraphrase kappa"] = cohen_kappa(a, c)
        if len(models) > 1:
            a = np.array([it["judge"][f"{models[0]}|layer_b|s0"] == "YES" for it in items])
            b = np.array([it["judge"][f"{models[1]}|layer_b|s0"] == "YES" for it in items])
            consistency["inter-judge kappa"] = cohen_kappa(a, b)
        report["consistency"] = consistency
    for tau in taus:
        pred = np.array([lexical_turn_verdict(it["parsed"], it["memories"], tau) for it in items])
        report["raters"][f"lexical_tau{tau}"] = summarise(pred)
        # per-memory attribution against per-memory evidence labels
        mem_pred, mem_ref = [], []
        for it in items:
            for m, is_ev in zip(it["memories"], it["memory_is_evidence"]):
                mem_pred.append(f1_score(it["parsed"] or "", m)[0] >= tau)
                mem_ref.append(is_ev)
        mp, mr = np.array(mem_pred), np.array(mem_ref)
        report["raters"][f"lexical_tau{tau}"]["per_memory_kappa"] = cohen_kappa(mp, mr) if mr.any() else None
    return report


async def run(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    guard = WindowGuard()
    items: List[Dict[str, Any]] = []
    for d in args.inputs:
        items.extend(build_items(Path(d), guard))
    logger.info("E5: %d judged items", len(items))
    usage = {}
    if not args.no_llm:
        from config.settings import DARSConfig
        usage = await judge_items(items, [DARSConfig.OPENAI_AUX_MODEL, DARSConfig.OPENAI_READER_MODEL],
                                  seeds=[0, 1], concurrency=args.concurrency)
    report = agreement_report(items, args.taus, args.n_boot)
    with (out / "items.jsonl").open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it) + "\n")
    manifest = {"experiment": "E5_judge", "inputs": args.inputs, "items": len(items), "prompts": PROMPTS,
                "lexical_taus": args.taus, "llm_usage": usage, "provenance": collect_provenance()}
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "raters"}, indent=1))
    for name, r in report["raters"].items():
        print(f"{name:45s} n={r['n']:4d} acc={r['accuracy']['mean']:.3f} kappa={r['kappa']} "
              f"neutral={r['neutral_rate']:.3f}")


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="E5 judge reliability")
    p.add_argument("--inputs", nargs="+", required=True, help="E1 run directories with reader outputs")
    p.add_argument("--out", required=True)
    p.add_argument("--taus", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    p.add_argument("--no-llm", action="store_true", help="only the deterministic (lexical) rater")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for noisy in ("httpx", "core.layer_d.storage", "benchmarks.memory_agent_bench.loader"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
