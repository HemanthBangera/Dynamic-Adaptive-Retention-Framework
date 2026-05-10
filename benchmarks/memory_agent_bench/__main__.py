from __future__ import annotations

"""CLI: python -m benchmarks.memory_agent_bench.

Examples:
  python -m benchmarks.memory_agent_bench list-sources --split Accurate_Retrieval
  python -m benchmarks.memory_agent_bench run --split Accurate_Retrieval --source ruler_qa_xxx \\
      --max-samples 1 --chunk-size 4096 --path b --output-dir ./benchmark_runs/pilot/example
"""

import argparse
import asyncio
import json
import logging
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Optional, Set

from config.settings import DARSConfig
from core.gemini_transport import GovernedGeminiTransport, collect_gemini_keys_from_env_and_file
from core.layer_a.gateway import CognitiveGateway
from core.layer_a.reformulator import QueryReformulator
from core.layer_a.reranker import DARSReranker
from core.layer_d.storage import MemoryVault

from benchmarks.memory_agent_bench.loader import (
    SUPPORTED_SPLITS,
    list_sources_for_split,
    load_mab_filtered,
)
from benchmarks.memory_agent_bench.manifest import build_manifest, write_manifest
from benchmarks.memory_agent_bench.reader import GeminiBenchmarkReader
from benchmarks.memory_agent_bench.runner import (
    merge_episode_metrics,
    run_single_sample,
    summarize_metrics,
    write_summary_md,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _cmd_run_presets(args: argparse.Namespace) -> None:
    """Run several pilot jobs via subprocess (fresh Python per job)."""
    project_root = Path(__file__).resolve().parents[2]
    entries = json.loads(Path(args.preset_file).read_text(encoding="utf-8"))
    exe = sys.executable
    for item in entries:
        if not isinstance(item, dict) or "split" not in item:
            continue
        cmd = [
            exe,
            "-m",
            "benchmarks.memory_agent_bench",
            "run",
            "--split",
            item["split"],
            "--source",
            item["source"],
            "--max-samples",
            str(item.get("max_samples", 1)),
            "--chunk-size",
            str(item.get("chunk_size", args.chunk_size)),
            "--path",
            item.get("path", "b"),
            "--output-dir",
            item["output_dir"],
            "--hf-revision",
            args.hf_revision,
        ]
        if item.get("baseline"):
            cmd.extend(["--baseline", item["baseline"]])
        if item.get("min_context_tokens") is not None:
            cmd.extend(["--min-context-tokens", str(item["min_context_tokens"])])
        if item.get("max_context_tokens") is not None:
            cmd.extend(["--max-context-tokens", str(item["max_context_tokens"])])
        if item.get("gemini_keys_file"):
            cmd.extend(["--gemini-keys-file", item["gemini_keys_file"]])
        if item.get("gemini_min_interval") is not None:
            cmd.extend(["--gemini-min-interval", str(item["gemini_min_interval"])])
        if item.get("run_label"):
            cmd.extend(["--run-label", item["run_label"]])
        if item.get("keep_collection"):
            cmd.append("--keep-collection")
        if item.get("no_vault_recreate"):
            cmd.append("--no-vault-recreate")
        if item.get("audit_jsonl"):
            cmd.extend(["--audit-jsonl", item["audit_jsonl"]])
        if args.resume:
            cmd.append("--resume")
        logger.info("Running: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=project_root)


def _cmd_list_sources(args: argparse.Namespace) -> None:
    for s in list_sources_for_split(args.split, revision=args.hf_revision):
        print(s)


def _load_done_keys(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    done: Set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            done.add(obj.get("context_key", ""))
        except json.JSONDecodeError:
            continue
    return done


def _context_key(row: Dict[str, Any]) -> str:
    ctx = row.get("context") or ""
    return str(hash(ctx))


async def _cmd_run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    min_tok = args.min_context_tokens if getattr(args, "min_context_tokens", None) is not None else None
    max_tok = args.max_context_tokens if getattr(args, "max_context_tokens", None) is not None else None

    rows, load_stats = load_mab_filtered(
        args.split,
        args.source,
        max_test_samples=args.max_samples,
        seed=args.seed,
        revision=args.hf_revision,
        min_context_tokens=min_tok,
        max_context_tokens=max_tok,
        tiktoken_model=args.tiktoken_model,
    )

    audit_rel: Optional[str] = None
    audit_path: Optional[Path] = None
    if getattr(args, "audit_jsonl", None):
        audit_path = Path(args.audit_jsonl)
        if not audit_path.is_absolute():
            audit_path = out_dir / audit_path
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            audit_rel = str(audit_path.resolve().relative_to(out_dir.resolve()))
        except ValueError:
            audit_rel = str(audit_path.resolve())

    manifest = build_manifest(
        split=args.split,
        source=args.source,
        chunk_size=args.chunk_size,
        max_test_samples=args.max_samples,
        seed=args.seed,
        hf_revision=args.hf_revision,
        path_mode=args.path,
        fetch_k=args.fetch_k,
        top_n=args.top_n,
        alpha=args.alpha,
        baseline=args.baseline,
        tiktoken_model=args.tiktoken_model,
        max_qa=(None if args.max_qa == 0 else args.max_qa),
        gemini_sleep_s=args.gemini_sleep,
        gemini_max_retries=args.gemini_retries,
        upstream_mab_commit=args.upstream_pin,
        min_context_tokens=min_tok,
        max_context_tokens=max_tok,
        load_stats=load_stats,
        gemini_min_interval_s=args.gemini_min_interval,
        gemini_keys_file=args.gemini_keys_file,
        vault_recreate=not args.no_vault_recreate,
        keep_collection=args.keep_collection,
        run_label=args.run_label,
        audit_jsonl=audit_rel,
    )
    write_manifest(out_dir / "run_manifest.json", manifest)

    dataset_config: Dict[str, Any] = {
        "dataset": args.split,
        "sub_dataset": args.source,
        "debug": False,
    }

    keys_file = args.gemini_keys_file if getattr(args, "gemini_keys_file", None) else None
    keys = collect_gemini_keys_from_env_and_file(keys_file)
    transport: Optional[GovernedGeminiTransport] = None
    if keys:
        transport = GovernedGeminiTransport(
            keys=keys,
            model=DARSConfig.GEMINI_MODEL,
            timeout=DARSConfig.GEMINI_TIMEOUT,
            min_interval_s=float(args.gemini_min_interval),
            max_retries=int(args.gemini_retries),
        )
        logger.info(
            "Gemini governed transport: %d key(s), min_interval=%.2fs model=%s",
            len(keys),
            float(args.gemini_min_interval),
            DARSConfig.GEMINI_MODEL,
        )

    reader = GeminiBenchmarkReader(max_retries=args.gemini_retries, transport=transport)
    done = _load_done_keys(out_dir / "per_sample.jsonl") if args.resume else set()

    def vault_factory(cname: str) -> MemoryVault:
        return MemoryVault(collection_name=cname)

    def gateway_factory(vault: MemoryVault) -> CognitiveGateway:
        return CognitiveGateway(
            reformulator=QueryReformulator(transport=transport),
            reranker=DARSReranker(vault=vault),
            alpha=args.alpha,
            fetch_k=args.fetch_k,
            top_n=args.top_n,
        )

    acc: DefaultDict[str, list] = defaultdict(list)
    jsonl_path = out_dir / "per_sample.jsonl"
    all_detail_rows: list = []

    try:
        for row in rows:
            ck = _context_key(row)
            if ck in done:
                logger.info("resume_skip context_key=%s", ck)
                continue
            ep = await run_single_sample(
                row=row,
                split_name=args.split,
                source_filter=args.source,
                chunk_size=args.chunk_size,
                tiktoken_model=args.tiktoken_model,
                path_mode=args.path,
                fetch_k=args.fetch_k,
                top_n=args.top_n,
                alpha=args.alpha,
                baseline=args.baseline,
                max_qa=(None if args.max_qa == 0 else args.max_qa),
                gemini_sleep_s=args.gemini_sleep,
                dataset_config=dataset_config,
                reader=reader,
                gateway_factory=gateway_factory,
                vault_factory=vault_factory,
                vault_recreate=not args.no_vault_recreate,
                keep_collection=args.keep_collection,
                run_label=args.run_label,
                audit_jsonl_path=audit_path,
            )
            merge_episode_metrics(acc, ep)
            all_detail_rows.extend(ep.rows)
            rec = {
                "context_key": ck,
                "metrics_snapshot": {k: list(v)[-5:] for k, v in ep.metrics.items()},
                "n_qa": len(ep.rows),
            }
            with jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")

        summary = summarize_metrics(acc)
        (out_dir / "results.json").write_text(
            json.dumps(all_detail_rows, indent=2, default=str), encoding="utf-8"
        )
        (out_dir / "metrics_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        write_summary_md(
            out_dir / "summary.md",
            title=f"MemoryAgentBench pilot {args.split} / {args.source}",
            summary=summary,
            manifest=manifest,
        )
        logger.info("Wrote results under %s", out_dir.resolve())
    finally:
        if transport is not None:
            await transport.aclose()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="MemoryAgentBench × DARS driver")
    sub = p.add_subparsers(dest="cmd", required=True)

    ls = sub.add_parser("list-sources", help="List metadata.source values in a split")
    ls.add_argument("--split", required=True, choices=sorted(SUPPORTED_SPLITS))
    ls.add_argument("--hf-revision", default=DARSConfig.MAB_HF_REVISION)
    ls.set_defaults(func=_cmd_list_sources)

    run = sub.add_parser("run", help="Run pilot evaluation")
    run.add_argument("--split", required=True)
    run.add_argument("--source", required=True, help="metadata.source filter (see list-sources)")
    run.add_argument("--max-samples", type=int, default=2)
    run.add_argument("--chunk-size", type=int, default=4096)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--hf-revision", default=DARSConfig.MAB_HF_REVISION)
    run.add_argument("--tiktoken-model", default=DARSConfig.MAB_TIKTOKEN_MODEL)
    run.add_argument("--min-context-tokens", type=int, default=None, help="Drop rows with fewer context tokens")
    run.add_argument("--max-context-tokens", type=int, default=None, help="Drop rows with more context tokens")
    run.add_argument("--path", choices=("a", "b"), default="b", help="a=full gateway XML reader; b=rerank bullets (default)")
    run.add_argument("--fetch-k", type=int, default=DARSConfig.DEFAULT_FETCH_K)
    run.add_argument("--top-n", type=int, default=DARSConfig.DEFAULT_TOP_N)
    run.add_argument("--alpha", type=float, default=DARSConfig.RERANK_ALPHA)
    run.add_argument("--baseline", choices=("normal", "empty"), default="normal")
    run.add_argument(
        "--max-qa",
        type=int,
        default=5,
        help="Cap QA pairs per context (ruler rows can have dozens; avoids Gemini 429 during pilots). Use 0 for no cap.",
    )
    run.add_argument(
        "--gemini-sleep",
        type=float,
        default=4.0,
        help="Seconds to sleep after each QA episode step (reader pacing).",
    )
    run.add_argument(
        "--gemini-retries",
        type=int,
        default=4,
        help="Max transport-level retries per Gemini call (429/server; backoff capped in transport).",
    )
    run.add_argument(
        "--gemini-min-interval",
        type=float,
        default=4.0,
        help="Minimum seconds between Gemini generateContent calls (free-tier RPM). Use 0 to disable.",
    )
    run.add_argument(
        "--gemini-keys-file",
        type=str,
        default=None,
        help="Path to file with one Gemini API key per line (# comments ok). Merged with GEMINI_API_KEYS / GEMINI_API_KEY.",
    )
    run.add_argument("--run-label", type=str, default=None, help="Optional label segment in Qdrant collection name")
    run.add_argument(
        "--no-vault-recreate",
        action="store_true",
        help="If set, do not delete/recreate Qdrant collection at episode start (initialize only if missing).",
    )
    run.add_argument(
        "--keep-collection",
        action="store_true",
        help="Do not delete Qdrant collection after each context episode (debug / inspection).",
    )
    run.add_argument(
        "--audit-jsonl",
        type=str,
        default=None,
        help="Append per-QA audit records (money metrics) to this path (relative to output-dir if not absolute).",
    )
    run.add_argument(
        "--output-dir",
        default="./benchmark_runs/pilot",
        help="Directory for run_manifest.json, results.json, summary.md (default under benchmark_runs/).",
    )
    run.add_argument("--resume", action="store_true")
    run.add_argument("--upstream-pin", default="main", help="Record upstream MAB commit or branch label in manifest")
    run.set_defaults(func=lambda a: asyncio.run(_cmd_run(a)))

    rp = sub.add_parser(
        "run-presets",
        help="Run a JSON list of {split, source, max_samples, output_dir, ...} jobs sequentially",
    )
    rp.add_argument("--preset-file", required=True, help="JSON file (see pilot_presets.template.json)")
    rp.add_argument("--hf-revision", default=DARSConfig.MAB_HF_REVISION)
    rp.add_argument("--chunk-size", type=int, default=4096)
    rp.add_argument("--resume", action="store_true")
    rp.set_defaults(func=_cmd_run_presets)

    args = p.parse_args(list(argv) if argv is not None else None)
    args.func(args)


if __name__ == "__main__":
    main()
