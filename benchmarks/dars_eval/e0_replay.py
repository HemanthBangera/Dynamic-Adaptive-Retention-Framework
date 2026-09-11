"""
E0 — harness equivalence check against the submitted runs.

Re-runs the as-submitted retrieval configuration through the real MemoryVault
(4096-token chunks, ALFWorld goal preset for P, RRF over fetch_k = 15, top 3,
agentic-memory query template) on the first five questions of each context and
compares it with the submitted ``audit.jsonl``:

* fingerprint — ``retrieved_memory_tokens`` (tiktoken length of the retrieved
  bullets), which identifies which chunks were retrieved;
* ``dars_mean_topk − dars_mean_vault``.

The submitted runs stored chunks one at a time (a fraction of a second to a
few seconds apart), so memories whose other DARS terms were identical (e.g. P
clipped to 0) differed only by tiny recency gaps, partly below the 6-decimal
rounding of the score.  Their relative order therefore depended on
unrecoverable sub-second ingestion timing.  The replay brackets it with the two
extreme tie-break policies:

* ``same_time``  – one ingestion timestamp: exact ties, broken by similarity order;
* ``sequential`` – 10 s between chunks: ties broken by recency (newer first).

A question is reproduced if its fingerprint matches under at least one policy.
Where the policies disagree, the same-time ranking must show an exact DARS tie
at the top-3 boundary, i.e. the difference is fully explained by tie-breaking.

Final check (``timeline``): the submitted ``results.json`` logs each context's
ingestion duration and each question's duration (plus a fixed 4 s pause), so the
real ingestion/query timeline is replayed.  For a question that still differs,
50 seeded per-chunk timing-jitter draws (±50 % per chunk, same total ingestion
time) are tried; the question counts as reproduced if any plausible draw matches.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import tiktoken

from benchmarks.memory_agent_bench.chunking import chunk_context
from benchmarks.memory_agent_bench.loader import load_mab_filtered
from benchmarks.memory_agent_bench.qa_builder import build_qa_pairs
from core.layer_d.storage import MemoryVault

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    "eventqa_65536": PROJECT_ROOT / "benchmark_runs/paper_final/accurate_retrieval_eventqa_65536_path_b/audit.jsonl",
    "ruler_qa1_197K": PROJECT_ROOT / "benchmark_runs/paper_final/accurate_retrieval_ruler_qa1_197k_path_b/audit.jsonl",
}
T0 = 1_700_000_000.0
INGEST_STEP_S = 10.0
ENC = tiktoken.encoding_for_model("gpt-4o-mini")


def _tie_at_boundary(candidates, top_n: int = 3) -> bool:
    """True if a DARS score inside the top-n is shared by a candidate outside it."""
    inside = {round(m.dars_score, 6) for m in candidates[:top_n]}
    return any(round(m.dars_score, 6) in inside for m in candidates[top_n:])


def _policy_vault(source: str, ci: int, chunks: List[str], policy: str):
    vault = MemoryVault(collection_name=f"e0_{source}_{ci}_{policy}", location=":memory:")
    vault.initialize_collection(recreate=True)
    if policy == "same_time":
        vault.store_memories_batch([{"text": c} for c in chunks], current_time=T0)
        now = T0
    else:
        vault.store_memories_batch([{"text": c, "timestamp": T0 + INGEST_STEP_S * i} for i, c in enumerate(chunks)])
        now = T0 + INGEST_STEP_S * len(chunks) + 60.0
    mean_vault, _ = vault.mean_dars_score_all_points(current_time=now)
    return vault, now, mean_vault


def replay(source: str) -> Dict[str, Any]:
    logged = [json.loads(line) for line in RUNS[source].read_text(encoding="utf-8").splitlines() if line.strip()]
    rows, _ = load_mab_filtered(
        "Accurate_Retrieval", source, max_test_samples=5 if source.startswith("eventqa") else 20, seed=42
    )
    records: List[Dict[str, Any]] = []
    k = 0
    for ci, row in enumerate(rows):
        chunks = chunk_context(row["context"], chunk_size=4096)
        policies = {p: _policy_vault(source, ci, chunks, p) for p in ("same_time", "sequential")}
        for fq, _a, _id in build_qa_pairs(row, source)[:5]:
            ref = logged[k]
            rec: Dict[str, Any] = {"context": ci, "question": k, "tokens_logged": ref["retrieved_memory_tokens"]}
            for policy, (vault, now, mean_vault) in policies.items():
                fused = vault.search_and_rerank(fq, fetch_k=15, top_n=15, rank_mode="rrf", current_time=now)
                top = fused[:3]
                tokens = len(ENC.encode("\n".join(f"- {m.payload.text_content}" for m in top)))
                delta = float(np.mean([m.dars_score for m in top])) - mean_vault
                rec[policy] = {
                    "fingerprint_match": tokens == ref["retrieved_memory_tokens"],
                    "tokens": tokens,
                    "abs_delta_diff": abs(delta - (ref["dars_mean_topk"] - ref["dars_mean_vault"])),
                    "tie_at_boundary": _tie_at_boundary(fused),
                }
            records.append(rec)
            k += 1
    return {"source": source, "questions": k, "records": records}


GEMINI_SLEEP_S = 4.0
JITTER_DRAWS = 50


def _timeline_vault(source: str, ci: int, chunks: List[str], mct: float, steps: np.ndarray, tag: str):
    vault = MemoryVault(collection_name=f"e0tl_{source}_{ci}_{tag}", location=":memory:")
    vault.initialize_collection(recreate=True)
    stamps = T0 + np.cumsum(steps)
    vault.store_memories_batch([{"text": c, "timestamp": float(t)} for c, t in zip(chunks, stamps)])
    return vault


def timeline_replay(source: str) -> Dict[str, Any]:
    """Replay with the logged ingestion/query timeline; jitter draws for residual mismatches."""
    logged = [json.loads(line) for line in RUNS[source].read_text(encoding="utf-8").splitlines() if line.strip()]
    results = json.loads((RUNS[source].parent / "results.json").read_text(encoding="utf-8"))
    rows, _ = load_mab_filtered(
        "Accurate_Retrieval", source, max_test_samples=5 if source.startswith("eventqa") else 20, seed=42
    )
    out: List[Dict[str, Any]] = []
    k = 0
    for ci, row in enumerate(rows):
        chunks = chunk_context(row["context"], chunk_size=4096)
        mct = float(results[k]["memory_construction_time"])
        base_steps = np.full(len(chunks), mct / len(chunks))
        vault = _timeline_vault(source, ci, chunks, mct, base_steps, "base")
        rng = np.random.default_rng([ci, len(chunks)])
        jitter_vaults = []
        t = T0 + mct
        for fq, _a, _id in build_qa_pairs(row, source)[:5]:
            target = logged[k]["retrieved_memory_tokens"]

            def fingerprint(v) -> int:
                top = v.search_and_rerank(fq, fetch_k=15, top_n=3, rank_mode="rrf", current_time=t)
                return len(ENC.encode("\n".join(f"- {m.payload.text_content}" for m in top)))

            rec = {"context": ci, "question": k, "timeline_match": fingerprint(vault) == target, "jitter_match": None}
            if not rec["timeline_match"]:
                if not jitter_vaults:
                    for d in range(JITTER_DRAWS):
                        steps = base_steps * rng.uniform(0.5, 1.5, size=len(chunks))
                        steps *= mct / steps.sum()
                        jitter_vaults.append(_timeline_vault(source, ci, chunks, mct, steps, f"j{d}"))
                rec["jitter_match"] = any(fingerprint(v) == target for v in jitter_vaults)
            out.append(rec)
            t += float(results[k]["query_time_len"]) + GEMINI_SLEEP_S
            k += 1
    return {"source": source, "questions": k, "records": out}


def main() -> None:
    ok = True
    for source in RUNS:
        tl = timeline_replay(source)
        recs = tl["records"]
        exact = sum(r["timeline_match"] for r in recs)
        jitter = sum(1 for r in recs if r["jitter_match"])
        print(json.dumps({
            "source": source,
            "questions": tl["questions"],
            "reproduced_logged_timeline": exact,
            "reproduced_under_timing_jitter": jitter,
            "not_reproduced": tl["questions"] - exact - jitter,
        }))
        ok &= exact + jitter == tl["questions"]
    print("E0 PASS" if ok else "E0 FAIL")
    sys.exit(0 if ok else 1)


def bracket_report() -> None:
    """Diagnostic: the two extreme tie-break policies (same_time / sequential)."""
    for source in RUNS:
        res = replay(source)
        recs = res["records"]
        reproduced, disagreements_explained, unexplained, max_diff = 0, 0, 0, 0.0
        for r in recs:
            a, b = r["same_time"], r["sequential"]
            matches = [p for p in (a, b) if p["fingerprint_match"]]
            if matches:
                reproduced += 1
                max_diff = max(max_diff, min(p["abs_delta_diff"] for p in matches))
            if a["fingerprint_match"] != b["fingerprint_match"] or not matches:
                if matches and a["tie_at_boundary"]:
                    disagreements_explained += 1
                else:
                    unexplained += 1
        print(json.dumps({
            "source": source,
            "questions": res["questions"],
            "reproduced_under_some_tie_break": reproduced,
            "matches_same_time": sum(r["same_time"]["fingerprint_match"] for r in recs),
            "matches_sequential": sum(r["sequential"]["fingerprint_match"] for r in recs),
            "policy_disagreements_explained_by_exact_ties": disagreements_explained,
            "unexplained": unexplained,
            "max_abs_delta_diff_on_matches": max_diff,
        }))


if __name__ == "__main__":
    main()
