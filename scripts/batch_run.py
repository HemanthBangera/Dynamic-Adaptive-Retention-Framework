"""Send collected LLM requests through the OpenAI Batch API and cache the responses.

The synchronous endpoint is capped at 10,000 gpt-4o-mini requests/day on Tier 1, which would
stretch the remaining test phase over days. The Batch API draws on a separate quota - it
accepted work while the daily cap was exhausted - and a 341-request batch came back in about
three minutes, so the reader and judge calls go through it instead.

Nothing about the experiment changes. The request bodies are byte-identical to what the
synchronous path would have sent (same model snapshot, temperature, seed and token limit),
and each response is written into the same content-addressed cache, so replaying a stage
afterwards produces exactly these answers with no further API calls.

The account allows 2,000,000 *enqueued* tokens at a time, so this submits chunks, waits for
them to finish, ingests the answers, and refills the pipe until everything is done.

Workflow per stage:

1. collect - run the stage with DARS_LLM_COLLECT set and a scratch --out. No API calls are
   made for the collected model; that pass's own outputs are meaningless and discarded.
2. batch   - python scripts/batch_run.py run --in <collected.jsonl> --state <state.json>
3. replay  - run the stage again normally: every call is now a cache hit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.llm_transport import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    estimate_tokens,
    load_openai_api_key,
    price_for,
)

MAX_REQUESTS_PER_BATCH = 2000
MAX_TOKENS_PER_BATCH = 900_000        # two chunks fit inside the enqueued allowance at once
MAX_TOKENS_IN_FLIGHT = 1_900_000      # account limit is 2,000,000 enqueued tokens
TERMINAL = ("completed", "failed", "expired", "cancelled")


def cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / key[:2] / f"{key}.json"


def load_requests(path: Path, cache_dir: Path) -> List[Dict[str, Any]]:
    """Collected lines that are not already cached, de-duplicated by cache key."""
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        key = item["custom_id"]
        if key in seen or cache_path(cache_dir, key).is_file():
            continue
        seen.add(key)
        out.append(item)
    return out


def chunks(items: List[Dict[str, Any]], max_requests: int,
           max_tokens: int) -> Iterable[List[Dict[str, Any]]]:
    """Split into batches small enough for the request-count and enqueued-token limits."""
    batch: List[Dict[str, Any]] = []
    tokens = 0
    for item in items:
        t = estimate_tokens(item["body"])
        if batch and (len(batch) >= max_requests or tokens + t > max_tokens):
            yield batch
            batch, tokens = [], 0
        batch.append(item)
        tokens += t
    if batch:
        yield batch


def write_cache_entry(cache_dir: Path, key: str, body: Dict[str, Any],
                      response: Dict[str, Any]) -> bool:
    """Store one batch response in the transport's cache format; False if unusable."""
    choices = response.get("choices") or []
    if not choices:
        return False
    usage = response.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    price = price_for(body.get("model", ""))
    cost = ((prompt_tokens * price[0] + completion_tokens * price[1]) / 1_000_000) if price else 0.0
    record = {
        "request": body,
        "text": (choices[0].get("message", {}).get("content") or "").strip(),
        "finish_reason": choices[0].get("finish_reason"),
        "model": response.get("model"),
        "system_fingerprint": response.get("system_fingerprint"),
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        "cost_usd": cost,
        "created": time.time(),
        "via": "batch",
    }
    path = cache_path(cache_dir, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".batch.tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return True


def ingest(client, batch, chunk: List[Dict[str, Any]], cache_dir: Path) -> Dict[str, int]:
    """Write a finished batch's answers into the cache. Returns counts."""
    stats = {"written": 0, "unusable": 0}
    if not getattr(batch, "output_file_id", None):
        return stats
    bodies = {i["custom_id"]: i["body"] for i in chunk}
    for line in client.files.content(batch.output_file_id).text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        response = row.get("response") or {}
        body = bodies.get(row.get("custom_id"))
        if body and response.get("status_code") == 200 and \
                write_cache_entry(cache_dir, row["custom_id"], body, response.get("body") or {}):
            stats["written"] += 1
        else:
            stats["unusable"] += 1
    return stats


def run_batches(items: List[Dict[str, Any]], cache_dir: Path, client,
                max_requests: int = MAX_REQUESTS_PER_BATCH,
                max_tokens_per_batch: int = MAX_TOKENS_PER_BATCH,
                max_tokens_in_flight: int = MAX_TOKENS_IN_FLIGHT,
                interval: float = 20.0, timeout: float = 24 * 3600,
                sleep: Callable[[float], None] = time.sleep,
                log: Callable[[str], None] = print) -> Dict[str, int]:
    """Push every request through the Batch API, keeping inside the enqueued-token limit.

    Chunks are submitted only while the in-flight total leaves room; a chunk rejected for
    the token limit goes back on the queue and is retried once capacity frees, so a
    too-eager submission never loses work.
    """
    queue = list(chunks(items, max_requests, max_tokens_per_batch))
    totals = {"written": 0, "unusable": 0, "batches": 0, "requests": len(items)}
    in_flight: Dict[str, Dict[str, Any]] = {}
    deadline = time.time() + timeout

    def tokens_of(chunk: List[Dict[str, Any]]) -> int:
        return sum(estimate_tokens(c["body"]) for c in chunk)

    while (queue or in_flight) and time.time() < deadline:
        used = sum(e["tokens"] for e in in_flight.values())
        while queue:
            chunk = queue[0]
            t = tokens_of(chunk)
            if in_flight and used + t > max_tokens_in_flight:
                break
            queue.pop(0)
            try:
                batch = _create_batch(client, chunk)
            except Exception as exc:                      # transient or over-limit: retry later
                log(f"  submit deferred ({type(exc).__name__}): {str(exc)[:120]}")
                queue.insert(0, chunk)
                break
            in_flight[batch.id] = {"chunk": chunk, "tokens": t}
            used += t
            totals["batches"] += 1
            log(f"  submitted {batch.id}: {len(chunk)} requests, ~{t:,} tokens "
                f"(in flight ~{used:,})")

        for bid in list(in_flight):
            batch = client.batches.retrieve(bid)
            if batch.status not in TERMINAL:
                continue
            entry = in_flight.pop(bid)
            if batch.status == "completed":
                stats = ingest(client, batch, entry["chunk"], cache_dir)
                totals["written"] += stats["written"]
                totals["unusable"] += stats["unusable"]
                log(f"  {bid}: completed, cached {stats['written']}")
            else:
                # A rejected batch (usually the enqueued-token limit) is re-queued whole.
                log(f"  {bid}: {batch.status} - requeued ({getattr(batch, 'errors', None)})")
                queue.append(entry["chunk"])
                totals["batches"] -= 1
        if queue or in_flight:
            sleep(interval)
    totals["pending"] = len(queue) + len(in_flight)
    return totals


def _create_batch(client, chunk: List[Dict[str, Any]]):
    import io

    payload = ("\n".join(json.dumps(c, ensure_ascii=False) for c in chunk) + "\n").encode("utf-8")
    uploaded = client.files.create(file=io.BytesIO(payload), purpose="batch")
    return client.batches.create(input_file_id=uploaded.id, endpoint="/v1/chat/completions",
                                 completion_window="24h")


def recover(inputs: List[Path], cache_dir: Path, client, limit: int = 100,
            log: Callable[[str], None] = print) -> Dict[str, int]:
    """Ingest completed batches that no live scheduler is tracking.

    A dropped connection can kill the submitting process while its batches are still
    running; their answers are paid for and sitting on the server, so we match them back
    to the collected requests by custom_id and cache them rather than paying twice.
    """
    bodies: Dict[str, Dict[str, Any]] = {}
    for path in inputs:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                bodies[item["custom_id"]] = item["body"]
    totals = {"written": 0, "skipped": 0, "batches": 0}
    for batch in client.batches.list(limit=limit).data:
        if batch.status != "completed" or not getattr(batch, "output_file_id", None):
            continue
        rows = client.files.content(batch.output_file_id).text.splitlines()
        wrote = 0
        for line in rows:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("custom_id")
            body = bodies.get(key)
            response = row.get("response") or {}
            if not body or cache_path(cache_dir, key).is_file():
                totals["skipped"] += 1
                continue
            if response.get("status_code") == 200 and                     write_cache_entry(cache_dir, key, body, response.get("body") or {}):
                wrote += 1
        if wrote:
            totals["batches"] += 1
            totals["written"] += wrote
            log(f"  recovered {wrote} response(s) from {batch.id}")
    return totals


def _client():
    from openai import OpenAI

    return OpenAI(api_key=load_openai_api_key())


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Batch-API runner for collected LLM requests")
    p.add_argument("command", choices=("run", "recover"))
    p.add_argument("--in", dest="input", action="append", required=True,
                   help="collected JSONL; repeat for recover across several stages")
    p.add_argument("--state", required=False)
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--max-requests", type=int, default=MAX_REQUESTS_PER_BATCH)
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS_PER_BATCH)
    p.add_argument("--in-flight-tokens", type=int, default=MAX_TOKENS_IN_FLIGHT)
    p.add_argument("--interval", type=float, default=20.0)
    p.add_argument("--timeout", type=float, default=6 * 3600)
    args = p.parse_args(argv)

    cache_dir = Path(args.cache_dir)
    if args.command == "recover":
        totals = recover([Path(p) for p in args.input], cache_dir, _client())
        print(f"recovered {totals['written']} response(s) from {totals['batches']} batch(es)")
        return 0
    if len(args.input) != 1:
        p.error("run takes a single --in")
    items = load_requests(Path(args.input[0]), cache_dir)
    print(f"{len(items)} request(s) still need answers (cached ones skipped)")
    if not items:
        return 0
    totals = run_batches(items, cache_dir, _client(), max_requests=args.max_requests,
                         max_tokens_per_batch=args.max_tokens,
                         max_tokens_in_flight=args.in_flight_tokens,
                         interval=args.interval, timeout=args.timeout)
    print(f"batches={totals['batches']} cached={totals['written']} "
          f"unusable={totals['unusable']} pending={totals['pending']}")
    if args.state:
        Path(args.state).write_text(json.dumps(totals, indent=1), encoding="utf-8")
    return 0 if totals["pending"] == 0 and totals["unusable"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
