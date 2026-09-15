"""
E6 — Layer C: does compression preserve answer-relevant information, and does
shadow indexing keep compressed memories retrievable?

Items: gold-evidence memories of dev questions (RULER QA1/QA2: the gold-paragraph
unit containing the answer; LongMemEval: has_answer turn units).

(a) Compression fidelity.  Each evidence memory is compressed by
    * ``semantic``            — the real Layer C path: ``SemanticCompressor.compress_memory``
                                on a local vault point (gpt-4.1-nano), text read back from
                                the patched payload.  A failed compression is recorded
                                (``semantic_ok``) and excluded from the semantic metrics —
                                never replaced by the original text;
    * ``llmlingua2``          — LLMLingua-2 token classification (CPU), rate 0.5;
    * ``extractive``          — deterministic: keep the sentences most similar to the memory
                                centroid, in original order, within half the tokens;
    * ``<name>@matched``      — LLMLingua-2 / extractive at the compression ratio the
                                semantic compressor achieved on the same memory, so the
                                baselines are compared at an equal token budget per memory.
    Metrics: compression ratio (tokens after / before), answer retention (normalised
    gold answer still present), and reader accuracy when the reader sees only the
    question's evidence memories, original vs compressed (MemoryAgentBench scoring).

(b) Shadow indexing.
    * Verification: after ``compress_memory`` the vector stored in the vault is read
      back and compared with the original embedding (``shadow_vector_preserved``).
    * Consequence for retrieval: with the original vector kept, the memory's
      similarity rank is unchanged by construction; re-embedding the compressed text
      instead changes it.  Reported as Recall@10 with the kept vector (= uncompressed)
      vs with the re-embedded text.  Exact cosine ranking (as local Qdrant).
    Whether the retrieved compressed text still answers the question is measured
    by (a).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import tiktoken

from benchmarks.dars_eval.datasets import load_contexts, split_name_for
from benchmarks.dars_eval.labels import normalize_answer
from benchmarks.dars_eval.memory_units import WindowGuard
from benchmarks.dars_eval.provenance import collect_provenance
from benchmarks.dars_eval.splits import question_splits
from benchmarks.dars_eval.stats import cluster_bootstrap_mean, paired_bootstrap_diff

logger = logging.getLogger(__name__)
ENC = tiktoken.encoding_for_model("gpt-4o-mini")
COMPRESSORS = ("semantic", "llmlingua2", "extractive")
FIXED_RATE = 0.5


def answer_retained(text: str, answers: Sequence[str]) -> bool:
    t = normalize_answer(text)
    return any(normalize_answer(a) and normalize_answer(a) in t for a in answers)


def variant_names(compressors: Sequence[str]) -> List[str]:
    """Compressor variants evaluated: the named ones plus matched-ratio baselines."""
    names = list(compressors)
    if "semantic" in compressors:
        names += [f"{c}@matched" for c in ("llmlingua2", "extractive") if c in compressors]
    return names


def matched_rate(original_tokens: int, compressed_tokens: int) -> float:
    """Compression ratio achieved by the semantic compressor, clipped to [0.05, 0.95]."""
    return float(min(max(compressed_tokens / max(original_tokens, 1), 0.05), 0.95))


def extractive_compress(text: str, embedder, ratio: float = FIXED_RATE) -> str:
    from nltk import sent_tokenize

    sents = [s for s in sent_tokenize(text) if s.strip()]
    if len(sents) <= 1:
        words = text.split()
        return " ".join(words[: max(1, int(len(words) * ratio))])
    vecs = np.asarray(embedder.encode_batch(sents))
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    centroid = vecs.mean(axis=0)
    order = np.argsort(-(vecs @ centroid), kind="stable")
    budget = max(1, int(len(ENC.encode(text)) * ratio))
    keep, used = set(), 0
    for i in order:
        n = len(ENC.encode(sents[i]))
        if keep and used + n > budget:
            continue
        keep.add(int(i))
        used += n
    return " ".join(sents[i] for i in sorted(keep))


class LinguaCompressor:
    MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"

    def __init__(self):
        from llmlingua import PromptCompressor

        self.pc = PromptCompressor(model_name=self.MODEL, use_llmlingua2=True, device_map="cpu")

    def __call__(self, text: str, rate: float = FIXED_RATE) -> str:
        return self.pc.compress_prompt(text, rate=rate, force_tokens=["\n", ".", "?", "!"])["compressed_prompt"]


def stored_vector(vault, point_id: str) -> np.ndarray:
    records = vault.client.retrieve(collection_name=vault.collection_name, ids=[point_id], with_vectors=True)
    vec = records[0].vector
    if isinstance(vec, dict):
        vec = next(iter(vec.values()))
    return np.asarray(vec, dtype=np.float32)


def collect_items(source: str, guard: WindowGuard, split: str) -> List[Dict[str, Any]]:
    items = []
    for ctx in load_contexts(source, guard):
        splits = question_splits(source, ctx.index, len(ctx.questions))
        for q in range(len(ctx.questions)):
            if splits[q] != split and split != "all":
                continue
            if not ctx.evidence[q]:
                continue
            answers = ctx.answers[q] if isinstance(ctx.answers[q], list) else [ctx.answers[q]]
            units = sorted({u for g in ctx.evidence[q] for u in g})
            items.append({"source": source, "context": ctx.index, "question": q, "units": units,
                          "answers": answers, "formatted_query": ctx.formatted_queries[q],
                          "retrieval_query": ctx.queries[q]})
    return items


async def run(args: argparse.Namespace) -> None:
    from benchmarks.dars_eval.reader import MABReader
    from config.settings import DARSConfig
    from core.layer_c.compressor import SemanticCompressor
    from core.layer_d.storage import MemoryVault
    from core.llm_transport import OpenAITransport

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    guard = WindowGuard()
    emb = guard.embedder
    aux = OpenAITransport(DARSConfig.OPENAI_AUX_MODEL, max_tokens=300, max_concurrency=args.concurrency)
    reader_t = OpenAITransport(DARSConfig.OPENAI_READER_MODEL, max_concurrency=args.concurrency)
    lingua = LinguaCompressor() if "llmlingua2" in args.compressors else None
    names = variant_names(args.compressors)

    records: List[Dict[str, Any]] = []
    for source in args.sources:
        contexts = {c.index: c for c in load_contexts(source, guard)}
        items = collect_items(source, guard, args.split)
        if args.max_items:
            items = items[: args.max_items]
        reader = MABReader(reader_t, source, split_name_for(source))
        vectors = {ci: np.asarray(emb.encode_batch([u.text for u in c.units], batch_size=64), dtype=np.float32)
                   for ci, c in contexts.items() if any(it["context"] == ci for it in items)}
        vault = MemoryVault(collection_name=f"layerc_{source}", location=":memory:")
        vault.initialize_collection(recreate=True)
        compressor = SemanticCompressor(vault=vault, transport=aux)

        compressed: Dict[tuple, Dict[str, str]] = {}
        unit_meta: Dict[tuple, Dict[str, Any]] = {}
        for it in items:
            ctx = contexts[it["context"]]
            for u in it["units"]:
                key = (it["context"], u)
                if key in compressed:
                    continue
                text = ctx.units[u].text
                original_vec = vectors[it["context"]][u]
                comp: Dict[str, str] = {}
                meta: Dict[str, Any] = {"original_tokens": len(ENC.encode(text))}
                rate = FIXED_RATE
                if "semantic" in args.compressors:
                    pid = vault.store_memory(text, predictive_value=0.0, vector_override=original_vec.tolist())
                    ok = bool(await compressor.compress_memory(pid, text))
                    meta["semantic_ok"] = ok
                    if ok:
                        comp["semantic"] = vault.get_memory(pid).payload.text_content
                        meta["shadow_vector_preserved"] = bool(
                            np.allclose(stored_vector(vault, pid), original_vec, atol=1e-6))
                        rate = matched_rate(meta["original_tokens"], len(ENC.encode(comp["semantic"])))
                    meta["matched_rate"] = rate
                if lingua is not None:
                    comp["llmlingua2"] = lingua(text, rate=FIXED_RATE)
                    if "llmlingua2@matched" in names:
                        comp["llmlingua2@matched"] = lingua(text, rate=rate)
                if "extractive" in args.compressors:
                    comp["extractive"] = extractive_compress(text, emb, FIXED_RATE)
                    if "extractive@matched" in names:
                        comp["extractive@matched"] = extractive_compress(text, emb, rate)
                compressed[key] = comp
                unit_meta[key] = meta

        for it in items:
            ctx = contexts[it["context"]]
            keys = [(it["context"], u) for u in it["units"]]
            V = vectors[it["context"]]
            Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
            qv = np.asarray(emb.encode(it["retrieval_query"]))
            qv = qv / np.linalg.norm(qv)
            base_sims = Vn @ qv
            rec: Dict[str, Any] = {k: it[k] for k in ("source", "context", "question", "units", "answers")}
            originals = [ctx.units[u].text for u in it["units"]]
            rec["original_tokens"] = int(sum(len(ENC.encode(t)) for t in originals))
            rec["original_answer_retained"] = answer_retained(" ".join(originals), it["answers"])
            if "semantic" in args.compressors:
                rec["semantic_ok"] = all(unit_meta[k]["semantic_ok"] for k in keys)
                rec["shadow_vector_preserved"] = (all(unit_meta[k]["shadow_vector_preserved"] for k in keys)
                                                  if rec["semantic_ok"] else None)
                rec["matched_rates"] = [unit_meta[k]["matched_rate"] for k in keys]
            base = await reader.answer(originals, it["formatted_query"], ctx.answers[it["question"]])
            rec["reader_original"] = base["metrics"]
            best_rank = min(int((base_sims > base_sims[u]).sum()) + 1 for u in it["units"])
            rec["rank_uncompressed"] = best_rank
            for name in names:
                if name == "semantic" and not rec["semantic_ok"]:
                    rec[name] = None
                    continue
                texts = [compressed[k][name] for k in keys]
                res = await reader.answer(texts, it["formatted_query"], ctx.answers[it["question"]])
                re_vecs = np.asarray(emb.encode_batch(texts))
                re_vecs = re_vecs / np.linalg.norm(re_vecs, axis=1, keepdims=True)
                ranks_re = []
                for j, u in enumerate(it["units"]):
                    sims = base_sims.copy()
                    sims[u] = float(re_vecs[j] @ qv)
                    ranks_re.append(int((sims > sims[u]).sum()) + 1)
                rec[name] = {
                    "tokens": int(sum(len(ENC.encode(t)) for t in texts)),
                    "answer_retained": answer_retained(" ".join(texts), it["answers"]),
                    "reader": res["metrics"],
                    "rank_kept_vector": best_rank,
                    "rank_reembedded": min(ranks_re),
                    "texts": texts,
                }
            records.append(rec)
        if args.whole_context:
            await whole_context_ranks(source, items, records[len(records) - len(items):], contexts, vectors,
                                      compressed, unit_meta, names, args, emb, lingua, aux)
        logger.info("E6: %s done (%d items)", source, len(items))

    with (out / "items.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    summary = summarise(records, names, args.n_boot)
    manifest = {"experiment": "E6_layerc_compression", "sources": args.sources, "split": args.split,
                "compressors": args.compressors, "variants": names, "fixed_rate": FIXED_RATE,
                "lingua_model": LinguaCompressor.MODEL,
                "llm_usage": {aux.model: aux.ledger.as_dict(), reader_t.model: reader_t.ledger.as_dict()},
                "provenance": collect_provenance()}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=1))


async def whole_context_ranks(source: str, items: List[Dict[str, Any]], records: List[Dict[str, Any]],
                              contexts: Dict[int, Any], vectors: Dict[int, np.ndarray],
                              compressed: Dict[tuple, Dict[str, str]], unit_meta: Dict[tuple, Dict[str, Any]],
                              names: Sequence[str], args: argparse.Namespace, emb, lingua, aux,
                              chunk: int = 200) -> None:
    """Re-embed *every* memory of each context after compressing it, then rank the evidence.

    The target-only comparison re-embeds just the evidence memory while every competitor keeps
    its full-length vector, which favours re-embedding. Here each compressor is applied to all
    memories of the context, as a store that compresses its memories would do, so the evidence
    competes against equally compressed text. A memory whose semantic compression fails keeps
    its original text, as Layer C would leave it; the failure rate is recorded.
    """
    from core.layer_c.compressor import SemanticCompressor
    from core.layer_d.storage import MemoryVault

    for ci in sorted({it["context"] for it in items}):
        ctx = contexts[ci]
        V = vectors[ci]
        n = len(ctx.units)
        todo = [u for u in range(n) if (ci, u) not in compressed]
        failures = 0
        if "semantic" in args.compressors and todo:
            wvault = MemoryVault(collection_name=f"layerc_all_{source}_{ci}", location=":memory:")
            wvault.initialize_collection(recreate=True)
            wcomp = SemanticCompressor(vault=wvault, transport=aux)
            for lo in range(0, len(todo), chunk):
                part = todo[lo:lo + chunk]
                pids = [wvault.store_memory(ctx.units[u].text, predictive_value=0.0, vector_override=V[u].tolist())
                        for u in part]
                oks = await asyncio.gather(*(wcomp.compress_memory(pid, ctx.units[u].text) for pid, u in zip(pids, part)))
                for pid, u, ok in zip(pids, part, oks):
                    text = ctx.units[u].text
                    meta = {"original_tokens": len(ENC.encode(text)), "semantic_ok": bool(ok), "matched_rate": FIXED_RATE}
                    comp: Dict[str, str] = {}
                    if ok:
                        comp["semantic"] = wvault.get_memory(pid).payload.text_content
                        meta["matched_rate"] = matched_rate(meta["original_tokens"], len(ENC.encode(comp["semantic"])))
                    else:
                        failures += 1
                    compressed[(ci, u)] = comp
                    unit_meta[(ci, u)] = meta
        for u in todo:
            text = ctx.units[u].text
            comp = compressed.setdefault((ci, u), {})
            rate = unit_meta.setdefault((ci, u), {"matched_rate": FIXED_RATE}).get("matched_rate", FIXED_RATE)
            if lingua is not None:
                comp["llmlingua2"] = lingua(text, rate=FIXED_RATE)
                if "llmlingua2@matched" in names:
                    comp["llmlingua2@matched"] = lingua(text, rate=rate)
            if "extractive" in args.compressors:
                comp["extractive"] = extractive_compress(text, emb, FIXED_RATE)
                if "extractive@matched" in names:
                    comp["extractive@matched"] = extractive_compress(text, emb, rate)
        reembedded = {}
        for name in names:
            texts = [compressed[(ci, u)].get(name, ctx.units[u].text) for u in range(n)]
            M = np.asarray(emb.encode_batch(texts, batch_size=64), dtype=np.float32)
            reembedded[name] = M / np.linalg.norm(M, axis=1, keepdims=True)
        for it, rec in zip(items, records):
            if it["context"] != ci:
                continue
            qv = np.asarray(emb.encode(it["retrieval_query"]))
            qv = qv / np.linalg.norm(qv)
            rec["whole_context_semantic_failure_rate"] = failures / max(len(todo), 1)
            for name in names:
                if rec.get(name) is None:
                    continue
                sims = reembedded[name] @ qv
                rec[name]["rank_reembedded_all"] = min(int((sims > sims[u]).sum()) + 1 for u in it["units"])
        logger.info("E6 whole-context: %s context %d, %d memories (%d semantic failures)", source, ci, n, failures)


def summarise(records: List[Dict[str, Any]], names: Sequence[str], n_boot: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {"items": len(records)}
    if records and "semantic_ok" in records[0]:
        ok = [r["semantic_ok"] for r in records]
        out["semantic_failure_rate"] = float(1 - np.mean(ok))
        preserved = [r["shadow_vector_preserved"] for r in records if r["semantic_ok"]]
        out["shadow_vector_preserved_rate"] = float(np.mean(preserved)) if preserved else None
    for name in names:
        rs = [r for r in records if r.get(name) is not None]
        if not rs:
            continue
        clusters = [f"{r['source']}:{r['context']}" for r in rs]
        base_acc = [r["reader_original"].get("substring_exact_match", 0.0) for r in rs]
        acc = [r[name]["reader"].get("substring_exact_match", 0.0) for r in rs]
        ratio = [r[name]["tokens"] / max(r["original_tokens"], 1) for r in rs]
        kept = [float(r[name]["answer_retained"]) for r in rs if r["original_answer_retained"]]
        kept_cl = [c for c, r in zip(clusters, rs) if r["original_answer_retained"]]
        hit_kept = [float(r[name]["rank_kept_vector"] <= 10) for r in rs]
        hit_re = [float(r[name]["rank_reembedded"] <= 10) for r in rs]
        out[name] = {
            "n": len(rs),
            "compression_ratio": cluster_bootstrap_mean(ratio, clusters, n_boot=n_boot).as_dict(),
            "answer_retention": cluster_bootstrap_mean(kept, kept_cl, n_boot=n_boot).as_dict() if kept else None,
            "reader_original": cluster_bootstrap_mean(base_acc, clusters, n_boot=n_boot).as_dict(),
            "reader_accuracy": cluster_bootstrap_mean(acc, clusters, n_boot=n_boot).as_dict(),
            "reader_delta_vs_original": paired_bootstrap_diff(acc, base_acc, clusters, n_boot=n_boot),
            "recall@10_kept_vector": cluster_bootstrap_mean(hit_kept, clusters, n_boot=n_boot).as_dict(),
            "recall@10_reembedded": cluster_bootstrap_mean(hit_re, clusters, n_boot=n_boot).as_dict(),
            "kept_minus_reembedded": paired_bootstrap_diff(hit_kept, hit_re, clusters, n_boot=n_boot),
        }
        if all("rank_reembedded_all" in r[name] for r in rs):
            hit_all = [float(r[name]["rank_reembedded_all"] <= 10) for r in rs]
            out[name]["recall@10_reembedded_whole_context"] = cluster_bootstrap_mean(hit_all, clusters, n_boot=n_boot).as_dict()
            out[name]["kept_minus_reembedded_whole_context"] = paired_bootstrap_diff(hit_kept, hit_all, clusters, n_boot=n_boot)
            out[name]["whole_context_minus_target_only"] = paired_bootstrap_diff(hit_all, hit_re, clusters, n_boot=n_boot)
    return out


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="E6 Layer C compression fidelity and shadow indexing")
    p.add_argument("--sources", nargs="+", default=["ruler_qa1_197K", "ruler_qa2_421K", "longmemeval_s*"])
    p.add_argument("--out", required=True)
    p.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    p.add_argument("--compressors", nargs="+", default=list(COMPRESSORS))
    p.add_argument("--max-items", type=int, default=0)
    p.add_argument("--whole-context", action="store_true",
                   help="also compress and re-embed every memory of each context (fair shadow-indexing test)")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
    for noisy in ("httpx", "core.layer_d.storage", "core.layer_c.compressor", "benchmarks.memory_agent_bench.loader"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
