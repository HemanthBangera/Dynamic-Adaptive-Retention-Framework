"""A redacted cache carries no prompt text and still replays every call exactly."""

import asyncio
import json

from core.llm_transport import OpenAITransport
from scripts.batch_run import write_cache_entry
from scripts.redact_cache import redact_record

SECRET = "sk-test-secret-value-1234567890"


def test_redacted_record_keeps_response_and_hashes_messages():
    rec = {"request": {"model": "m", "seed": 0, "messages": [{"role": "user", "content": "a copyrighted page"}]},
           "text": "answer", "usage": {"prompt_tokens": 5, "completion_tokens": 1}}
    red = redact_record(rec)
    assert red["text"] == "answer" and red["usage"] == rec["usage"]
    assert "a copyrighted page" not in json.dumps(red)
    assert red["request"]["messages"][0]["content_chars"] == len("a copyrighted page")
    assert red["request"]["seed"] == 0 and red["redacted"] is True


def test_replay_from_a_redacted_cache_is_identical(tmp_path):
    full, red = tmp_path / "full", tmp_path / "red"
    t = OpenAITransport("gpt-4o-mini-2024-07-18", api_key=SECRET, cache_dir=full, offline=True)
    body = t.build_request("Memory 1: the novel text", "Be brief.", seed=0, max_tokens=16)
    key = t.cache_key(body)
    write_cache_entry(full, key, body, {"choices": [{"message": {"content": "Paris"}, "finish_reason": "stop"}],
                                        "usage": {"prompt_tokens": 12, "completion_tokens": 1},
                                        "model": "gpt-4o-mini-2024-07-18"})
    for path in full.rglob("*.json"):
        target = red / path.relative_to(full)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(redact_record(json.loads(path.read_text(encoding="utf-8")))), encoding="utf-8")
    a = asyncio.run(OpenAITransport("gpt-4o-mini-2024-07-18", api_key=SECRET, cache_dir=full, offline=True)
                    .complete("Memory 1: the novel text", "Be brief.", seed=0, max_tokens=16))
    b = asyncio.run(OpenAITransport("gpt-4o-mini-2024-07-18", api_key=SECRET, cache_dir=red, offline=True)
                    .complete("Memory 1: the novel text", "Be brief.", seed=0, max_tokens=16))
    assert a["text"] == b["text"] == "Paris" and a["usage"] == b["usage"] and a["cache_key"] == b["cache_key"]
    assert "novel text" not in json.dumps(json.loads(next(red.rglob("*.json")).read_text(encoding="utf-8")))
