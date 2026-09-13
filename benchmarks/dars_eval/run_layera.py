"""
E7 — Layer A (Cognitive Gateway): query reformulation, prompt format and memory order.

On a reader subset (EventQA-65K and RULER QA1 dev questions, static single-session
store, B = 5120 tokens), every combination of

query    raw       the question text (as in E1)
         reform    expanded by the real ``QueryReformulator`` (gpt-4.1-nano); the raw
                   question is still what the reader answers
format   mab       MemoryAgentBench blocks ``Memory i:\\n<text>`` before the query
         xml       the real ``PromptConstructor`` XML (system_context, memory_stream
                   with system_weight / last_accessed, current_user_query) with its
                   character cap lifted, so it carries exactly the memories the mab
                   format carries: a pure format contrast
order    best_first   highest-ranked memory first (current implementation)
         best_last    highest-ranked memory last, i.e. next to the query (the
                      ordering the submitted manuscript described)

plus the gateway exactly as implemented (``xml_capped``: PromptConstructor's default
20,000-character cap, best-first, raw and reformulated query).  At B = 5120 that cap
drops the lowest-ranked memories, so each row records how many memories reached the
prompt and evidence recall counts only those.

Every condition is answered by the gpt-4o-mini reader and scored with MemoryAgentBench
metrics; evidence recall is reported where labels exist (RULER).  Retrieval uses the
similarity ranking unless ``--method`` names another E1 method.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import tiktoken

from benchmarks.dars_eval.datasets import load_contexts, split_name_for
from benchmarks.dars_eval.memory_units import MemoryUnit, WindowGuard
from benchmarks.dars_eval.provenance import collect_provenance
from benchmarks.dars_eval.rankers import MemoryIndex, rank
from benchmarks.dars_eval.reader import MABReader, build_prompt
from benchmarks.dars_eval.retrieval_eval import cut_to_budget, evidence_metrics
from benchmarks.dars_eval.run_static import STATIC_TIME, default_methods
from benchmarks.dars_eval.splits import question_splits
from benchmarks.dars_eval.stats import cluster_bootstrap_mean, paired_bootstrap_diff
from core.layer_a.prompt_constructor import PromptConstructor
from core.layer_d.schema import MemoryPayload, MemoryPoint
from third_party.memoryagentbench_eval import post_process

logger = logging.getLogger(__name__)
ENC = tiktoken.encoding_for_model("gpt-4o-mini")
XML_SYSTEM = ("You are the assistant. Use the XML memory stream to answer the current user query.")
FACTORIAL = list(itertools.product(("raw", "reform"), ("mab", "xml"), ("best_first", "best_last")))
GATEWAY_AS_IMPLEMENTED = [("raw", "xml_capped", "best_first"), ("reform", "xml_capped", "best_first")]
CONDITIONS = FACTORIAL + GATEWAY_AS_IMPLEMENTED


def xml_prompt(query: str, texts: List[str], max_chars: Optional[int]) -> str:
    points = [
        MemoryPoint(point_id=f"m{i}", vector=[],
                    payload=MemoryPayload(text_content=t, recency=STATIC_TIME), dars_score=None)
        for i, t in enumerate(texts)
    ]
    return PromptConstructor.build(query=query, memories=points, max_chars=max_chars)


def prompt_units(shown: List[int], order: str, n_in_prompt: int) -> List[int]:
    """Units that reached the prompt, in rank order.

    The cap cuts the display order, so under best_last it drops the highest-ranked
    memories rather than the lowest."""
    ordered = shown if order == "best_first" else shown[::-1]
    kept = set(ordered[:n_in_prompt])
    return [u for u in shown if u in kept]


async def run(args: argparse.Namespace) -> None:
    from config.settings import DARSConfig
    from core.layer_a.reformulator import QueryReformulator
    from core.llm_transport import OpenAITransport

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    guard = WindowGuard()
    method = {m.name: m for m in default_methods()}[args.method]
    reader_t = OpenAITransport(DARSConfig.OPENAI_READER_MODEL, max_concurrency=args.concurrency)
    aux = OpenAITransport(DARSConfig.OPENAI_AUX_MODEL, max_tokens=200, max_concurrency=args.concurrency)
    reformulator = QueryReformulator(transport=aux)

    rows: List[Dict[str, Any]] = []
    for source in args.sources:
        reader = MABReader(reader_t, source, split_name_for(source))
        for ctx in load_contexts(source, guard):
            splits = question_splits(source, ctx.index, len(ctx.questions))
            qs = [q for q in range(len(ctx.questions)) if splits[q] == args.split or args.split == "all"]
            if args.max_per_context:
                qs = qs[: args.max_per_context]
            units = [MemoryUnit(u.text, None, dict(u.meta)) for u in ctx.units]
            index = MemoryIndex(units, f"e7_{source}_{ctx.index}", default_time=STATIC_TIME)
            unit_tokens = [len(ENC.encode(u.text)) for u in units]

            async def one(q: int) -> List[Dict[str, Any]]:
                expanded = await reformulator.reformulate_query(ctx.queries[q])
                results = []
                for query_kind, fmt, order in CONDITIONS:
                    qtext = ctx.queries[q] if query_kind == "raw" else expanded
                    ranked, _ = rank(index, method, qtext, limit=200, current_time=STATIC_TIME)
                    shown = cut_to_budget(ranked, unit_tokens, args.budget)
                    texts = [units[u].text for u in shown]
                    if order == "best_last":
                        texts = texts[::-1]
                    if fmt == "mab":
                        answer = await reader.answer(texts, ctx.formatted_queries[q], ctx.answers[q])
                        metrics, output = answer["metrics"], answer["output"]
                        n_in_prompt = len(texts)
                    else:
                        cap = PromptConstructor.DEFAULT_MAX_PROMPT_CHARS if fmt == "xml_capped" else None
                        prompt = xml_prompt(ctx.formatted_queries[q], texts, max_chars=cap)
                        n_in_prompt = prompt.count("<memory id=")   # memory text is XML-escaped
                        res = await reader_t.complete(prompt, system=XML_SYSTEM, seed=0,
                                                      max_tokens=reader.max_answer_tokens)
                        m, _extra = post_process({"output": res["text"]}, ctx.answers[q],
                                                 {"sub_dataset": source, "dataset": split_name_for(source)})
                        metrics = {k: float(v) for k, v in m.items() if isinstance(v, (bool, int, float))}
                        output = res["text"]
                    included = prompt_units(shown, order, n_in_prompt)
                    ev = evidence_metrics(included, ctx.evidence[q]) if ctx.evidence[q] else None
                    results.append({
                        "source": source, "context": ctx.index, "question": q,
                        "query": query_kind, "format": fmt, "order": order,
                        "memories_shown": len(shown), "memories_in_prompt": n_in_prompt,
                        "reformulated": expanded if query_kind == "reform" else None,
                        "reformulation_fell_back": expanded == ctx.queries[q],
                        "evidence": ev, "metrics": metrics, "output": output,
                    })
                return results

            for batch in await asyncio.gather(*(one(q) for q in qs)):
                rows.extend(batch)
            logger.info("E7: %s context %d done", source, ctx.index)

    with (out / "per_question.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    summary = summarise(rows, args.n_boot)
    manifest = {"experiment": "E7_layer_a", "sources": args.sources, "split": args.split,
                "method": method.as_dict(), "budget": args.budget, "conditions": CONDITIONS,
                "gateway_max_chars": PromptConstructor.DEFAULT_MAX_PROMPT_CHARS,
                "xml_system": XML_SYSTEM,
                "llm_usage": {reader_t.model: reader_t.ledger.as_dict(), aux.model: aux.ledger.as_dict()},
                "provenance": collect_provenance()}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    for key, v in summary["conditions"].items():
        print(f"{key:30s} acc={v['accuracy']['mean']:.3f} [{v['accuracy']['ci_lo']:.3f},{v['accuracy']['ci_hi']:.3f}]"
              + (f" recall={v['evidence_recall']['mean']:.3f}" if v.get("evidence_recall") else ""))
    for key, d in summary["contrasts"].items():
        print(f"{key:40s} diff={d['diff']:+.3f} [{d['ci_lo']:+.3f},{d['ci_hi']:+.3f}] p={d['p_value']:.3g}")


def summarise(rows: List[Dict[str, Any]], n_boot: int) -> Dict[str, Any]:
    by: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in rows:
        by.setdefault((r["query"], r["format"], r["order"]), []).append(r)
    for rs in by.values():
        rs.sort(key=lambda r: (r["source"], r["context"], r["question"]))
    out: Dict[str, Any] = {"conditions": {}, "contrasts": {}}
    for key, rs in by.items():
        acc = [r["metrics"].get("substring_exact_match", 0.0) for r in rs]
        cl = [f"{r['source']}:{r['context']}" for r in rs]
        entry = {"n": len(rs), "accuracy": cluster_bootstrap_mean(acc, cl, n_boot=n_boot).as_dict()}
        ev = [r["evidence"]["group_recall"] for r in rs if r["evidence"]]
        if ev:
            entry["evidence_recall"] = cluster_bootstrap_mean(ev, n_boot=n_boot).as_dict()
        entry["reformulation_fallback_rate"] = float(np.mean([r["reformulation_fell_back"] for r in rs]))
        entry["memories_in_prompt"] = float(np.mean([r["memories_in_prompt"] for r in rs]))
        entry["truncation_rate"] = float(np.mean([r["memories_in_prompt"] < r["memories_shown"] for r in rs]))
        out["conditions"]["|".join(key)] = entry

    def contrast(a: tuple, b: tuple, name: str) -> None:
        ra, rb = by[a], by[b]
        acc_a = [r["metrics"].get("substring_exact_match", 0.0) for r in ra]
        acc_b = [r["metrics"].get("substring_exact_match", 0.0) for r in rb]
        cl = [f"{r['source']}:{r['context']}" for r in ra]
        out["contrasts"][name] = paired_bootstrap_diff(acc_a, acc_b, cl, n_boot=n_boot)

    contrast(("reform", "mab", "best_first"), ("raw", "mab", "best_first"), "reformulation (mab, best_first)")
    contrast(("raw", "xml", "best_first"), ("raw", "mab", "best_first"), "xml vs mab (raw, best_first)")
    contrast(("raw", "mab", "best_last"), ("raw", "mab", "best_first"), "best_last vs best_first (raw, mab)")
    contrast(("raw", "xml_capped", "best_first"), ("raw", "xml", "best_first"), "gateway cap vs uncapped (raw, xml, best_first)")
    contrast(("reform", "xml_capped", "best_first"), ("raw", "mab", "best_first"),
             "gateway as implemented vs direct (best_first)")
    return out


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="E7 Layer A ablation")
    p.add_argument("--sources", nargs="+", default=["eventqa_65536", "ruler_qa1_197K"])
    p.add_argument("--out", required=True)
    p.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    p.add_argument("--method", default="similarity")
    p.add_argument("--budget", type=int, default=5120)
    p.add_argument("--max-per-context", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
    for noisy in ("httpx", "core.layer_d.storage", "core.layer_a.reformulator", "benchmarks.memory_agent_bench.loader"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
