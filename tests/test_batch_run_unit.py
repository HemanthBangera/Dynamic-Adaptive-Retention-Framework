"""Collect mode captures exact requests, and batched responses land in the transport's cache."""

from __future__ import annotations

import json

import pytest

from core.llm_transport import OpenAITransport
from scripts.batch_run import cache_path, chunks, load_requests, write_cache_entry

SECRET = "sk-test-secret-value-1234567890"


def _transport(tmp_path, **kw):
    return OpenAITransport("gpt-4o-mini-2024-07-18", api_key=SECRET, cache_dir=tmp_path, **kw)


@pytest.mark.asyncio
async def test_collect_mode_writes_the_exact_request_and_makes_no_call(monkeypatch, tmp_path):
    collected = tmp_path / "collected.jsonl"
    monkeypatch.setenv("DARS_LLM_COLLECT", str(collected))
    t = _transport(tmp_path / "cache")
    out = await t.complete("what is 6*7?", system="Be terse.", seed=2, max_tokens=17)
    assert out["collected"] is True and out["text"] == ""
    assert t.ledger.collected == 1 and t.ledger.calls == 0

    line = json.loads(collected.read_text(encoding="utf-8").strip())
    assert line["url"] == "/v1/chat/completions" and line["method"] == "POST"
    # the custom_id must be the cache key, and the body the request the sync path would send
    assert line["custom_id"] == out["cache_key"]
    assert line["body"] == t.build_request("what is 6*7?", "Be terse.", seed=2, max_tokens=17)
    assert not any((tmp_path / "cache").rglob("*.json"))      # nothing cached by a collect pass


@pytest.mark.asyncio
async def test_collect_mode_can_be_limited_to_one_model(monkeypatch, tmp_path):
    monkeypatch.setenv("DARS_LLM_COLLECT", str(tmp_path / "c.jsonl"))
    monkeypatch.setenv("DARS_LLM_COLLECT_MODEL", "gpt-4o-mini-2024-07-18")
    reader = _transport(tmp_path / "cache")
    assert (await reader.complete("q"))["collected"] is True
    aux = OpenAITransport("gpt-4.1-nano-2025-04-14", api_key=SECRET, cache_dir=tmp_path / "cache",
                          offline=True)
    with pytest.raises(Exception):        # not collected: it tries to run, and offline blocks it
        await aux.complete("q")


@pytest.mark.asyncio
async def test_a_batched_response_is_replayed_as_a_cache_hit(monkeypatch, tmp_path):
    """The end-to-end contract: collect -> batch -> the same call is served from cache."""
    cache = tmp_path / "cache"
    collected = tmp_path / "c.jsonl"
    monkeypatch.setenv("DARS_LLM_COLLECT", str(collected))
    t = _transport(cache)
    first = await t.complete("explain recall", system="Be brief.")
    item = json.loads(collected.read_text(encoding="utf-8").strip())

    written = write_cache_entry(cache, item["custom_id"], item["body"], {
        "choices": [{"message": {"content": " Recall is coverage. "}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 7},
        "model": "gpt-4o-mini-2024-07-18", "system_fingerprint": "fp_batch"})
    assert written

    monkeypatch.delenv("DARS_LLM_COLLECT")
    replay = OpenAITransport("gpt-4o-mini-2024-07-18", api_key=SECRET, cache_dir=cache, offline=True)
    out = await replay.complete("explain recall", system="Be brief.")
    assert out["cached"] is True and out["text"] == "Recall is coverage."
    assert out["cache_key"] == first["cache_key"]
    assert out["usage"]["prompt_tokens"] == 100
    assert out["cost_usd"] == pytest.approx((100 * 0.15 + 7 * 0.60) / 1e6)


def test_already_cached_and_duplicate_requests_are_not_resubmitted(tmp_path):
    cache = tmp_path / "cache"
    line = lambda key: json.dumps({"custom_id": key, "method": "POST",
                                   "url": "/v1/chat/completions", "body": {"model": "m"}})
    src = tmp_path / "c.jsonl"
    src.write_text("\n".join([line("aaa"), line("bbb"), line("aaa")]) + "\n", encoding="utf-8")
    p = cache_path(cache, "bbb")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{}", encoding="utf-8")
    assert [i["custom_id"] for i in load_requests(src, cache)] == ["aaa"]


def test_chunking_respects_both_request_and_token_limits():
    items = [{"custom_id": str(i),
              "body": {"model": "gpt-4o-mini-2024-07-18",
                       "messages": [{"role": "user", "content": "word " * 500}],
                       "max_tokens": 16}} for i in range(10)]
    assert [len(c) for c in chunks(items, max_requests=3, max_tokens=10**9)] == [3, 3, 3, 1]
    by_tokens = list(chunks(items, max_requests=1000, max_tokens=1200))
    assert len(by_tokens) > 1                       # ~500 tokens each, so ~2 per chunk
    assert sum(len(c) for c in by_tokens) == 10     # nothing dropped


def test_a_failed_batch_row_is_not_cached(tmp_path):
    assert write_cache_entry(tmp_path, "k", {"model": "gpt-4o-mini-2024-07-18"}, {}) is False
    assert not list(tmp_path.rglob("*.json"))


class _FakeBatch:
    def __init__(self, bid, status, output_file_id=None, errors=None):
        self.id, self.status, self.output_file_id, self.errors = bid, status, output_file_id, errors


class _FakeClient:
    """Enough of the OpenAI client to drive run_batches, enforcing the enqueued-token limit."""

    LIMIT = 2_000_000

    def __init__(self):
        self.n = 0
        self.in_flight = {}           # id -> tokens
        self.payloads = {}            # id -> request lines
        self.rejected = 0
        self.files = self._Files(self)
        self.batches = self._Batches(self)

    class _Files:
        def __init__(self, outer):
            self.outer = outer

        def create(self, file=None, purpose=None):
            data = file.read().decode("utf-8")
            self.outer.n += 1
            fid = f"file-{self.outer.n}"
            self.outer.payloads[fid] = [json.loads(l) for l in data.splitlines() if l.strip()]
            return type("F", (), {"id": fid})()

        def content(self, fid):
            rows = []
            for item in self.outer.payloads[fid]:
                rows.append(json.dumps({"custom_id": item["custom_id"], "response": {
                    "status_code": 200,
                    "body": {"choices": [{"message": {"content": f"answer {item['custom_id'][:4]}"},
                                          "finish_reason": "stop"}],
                             "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                             "model": "gpt-4o-mini-2024-07-18"}}}))
            return type("C", (), {"text": "\n".join(rows)})()

    class _Batches:
        def __init__(self, outer):
            self.outer = outer

        def create(self, input_file_id=None, **kw):
            o = self.outer
            from core.llm_transport import estimate_tokens
            tokens = sum(estimate_tokens(i["body"]) for i in o.payloads[input_file_id])
            bid = f"batch-{input_file_id}"
            if sum(o.in_flight.values()) + tokens > o.LIMIT:
                o.rejected += 1                      # mirrors the real "token_limit_exceeded"
                o.failed_ids = getattr(o, "failed_ids", set()) | {bid}
            else:
                o.in_flight[bid] = tokens
            o.file_of = getattr(o, "file_of", {})
            o.file_of[bid] = input_file_id
            return _FakeBatch(bid, "validating")

        def retrieve(self, bid):
            o = self.outer
            if bid in getattr(o, "failed_ids", set()):
                return _FakeBatch(bid, "failed", errors="token_limit_exceeded")
            o.in_flight.pop(bid, None)
            return _FakeBatch(bid, "completed", output_file_id=o.file_of[bid])


def test_scheduler_stays_inside_the_enqueued_limit_and_caches_everything(tmp_path):
    from scripts.batch_run import run_batches
    items = [{"custom_id": f"{i:064x}",
              "body": {"model": "gpt-4o-mini-2024-07-18",
                       "messages": [{"role": "user", "content": "word " * 2000}],
                       "max_tokens": 16}} for i in range(12)]
    client = _FakeClient()
    cache = tmp_path / "cache"
    totals = run_batches(items, cache, client, max_tokens_per_batch=900_000,
                         max_tokens_in_flight=1_900_000, interval=0, sleep=lambda _s: None,
                         log=lambda _m: None)
    assert totals["pending"] == 0 and totals["unusable"] == 0
    assert totals["written"] == 12
    assert len(list(cache.rglob("*.json"))) == 12


def test_a_batch_rejected_for_the_token_limit_is_requeued_not_lost(tmp_path):
    from scripts.batch_run import run_batches
    items = [{"custom_id": f"{i:064x}",
              "body": {"model": "gpt-4o-mini-2024-07-18",
                       "messages": [{"role": "user", "content": "word " * 2000}],
                       "max_tokens": 16}} for i in range(6)]
    client = _FakeClient()
    client.LIMIT = 5_000                  # server accepts ~2 items in flight
    # Our scheduler is told it may keep far more in flight than the server allows, so the
    # server rejects batches and the loop has to requeue them instead of losing the work.
    totals = run_batches(items, tmp_path / "cache", client, max_tokens_per_batch=2_500,
                         max_tokens_in_flight=10_000_000, interval=0, sleep=lambda _s: None,
                         log=lambda _m: None)
    assert client.rejected > 0            # the fake server did reject at least one
    assert totals["pending"] == 0 and totals["written"] == 6
    assert len(list((tmp_path / "cache").rglob("*.json"))) == 6


def test_recover_ingests_batches_whose_scheduler_died(tmp_path):
    """A dropped connection must not orphan answers we have already paid for."""
    from scripts.batch_run import recover, run_batches

    items = [{"custom_id": f"{i:064x}",
              "body": {"model": "gpt-4o-mini-2024-07-18",
                       "messages": [{"role": "user", "content": "word " * 100}],
                       "max_tokens": 16}} for i in range(4)]
    collected = tmp_path / "c.jsonl"
    collected.write_text("\n".join(json.dumps(i) for i in items) + "\n", encoding="utf-8")

    client = _FakeClient()
    cache = tmp_path / "cache"
    # submit, but throw the scheduler away before it can ingest (cache stays empty)
    run_batches(items, tmp_path / "elsewhere", client, interval=0, sleep=lambda _s: None,
                log=lambda _m: None)
    assert not list(cache.rglob("*.json"))

    client.batches.list = lambda limit=100: type("L", (), {"data": [
        _FakeBatch(bid, "completed", output_file_id=client.file_of[bid])
        for bid in client.file_of]})()
    totals = recover([collected], cache, client, log=lambda _m: None)
    assert totals["written"] == 4
    assert len(list(cache.rglob("*.json"))) == 4

    # running it again is harmless: already-cached answers are skipped, not rewritten
    again = recover([collected], cache, client, log=lambda _m: None)
    assert again["written"] == 0 and again["skipped"] == 4
