"""The archive builder keeps secrets, prompts and novel text out of the published archive."""

import json
import subprocess

from scripts.build_archive import (blocking, build_index, env_secret_values, history_secret_values,
                                   looks_like_placeholder, scan_overlap, scan_secrets, select_files,
                                   unredacted_records)
from scripts.redact_cache import redact_record

FAKE_OPENAI = "sk-proj-" + "q7Wz2Lr9Xk4Tn8Vb3Hm6Pc1Yd5Gs0Jf"          # random-looking, not a real key


def test_secret_scan_finds_patterns_and_env_values_without_returning_them(tmp_path):
    (tmp_path / "a.py").write_text(f'KEY = "{FAKE_OPENAI}"\n', encoding="utf-8")
    (tmp_path / "b.txt").write_text("the value is plain-but-secret-123456 here", encoding="utf-8")
    (tmp_path / "clean.md").write_text("task-based-retrieval-and-risk-assessment-framework, hf_snapshots", encoding="utf-8")
    env = tmp_path / "x.env"
    env.write_text("MY_API_KEY='plain-but-secret-123456'\nDEBUG=1\n", encoding="utf-8")
    findings = scan_secrets(tmp_path, env_secret_values(env))
    kinds = {(f["file"], f["kind"]) for f in findings}
    assert ("a.py", "openai_key") in kinds
    assert ("b.txt", "known_secret_0") in kinds and ("x.env", "known_secret_0") in kinds
    assert not any(f["file"] == "clean.md" for f in findings)
    assert "plain-but-secret" not in json.dumps(findings) and FAKE_OPENAI not in json.dumps(findings)


def test_unredacted_cache_records_are_reported(tmp_path):
    rec = {"request": {"model": "m", "messages": [{"role": "user", "content": "novel page"}]}, "text": "x"}
    (tmp_path / "ab").mkdir()
    (tmp_path / "ab" / "k1.json").write_text(json.dumps(redact_record(rec)), encoding="utf-8")
    (tmp_path / "ab" / "k2.json").write_text(json.dumps(rec), encoding="utf-8")
    assert unredacted_records(tmp_path) == ["ab/k2.json".replace("/", __import__("os").sep)]


def test_overlap_scan_flags_twelve_word_spans_only(tmp_path):
    novel = ("It was a bright cold day in April and the clocks were striking thirteen while "
             "the wind swept the dust along the empty street")
    index = build_index([novel])
    quote = " ".join(novel.split()[:20])
    (tmp_path / "quote.jsonl").write_text(json.dumps({"output": "He wrote: " + quote}), encoding="utf-8")
    (tmp_path / "short.json").write_text(json.dumps({"output": "a bright cold day in April"}), encoding="utf-8")
    hits = scan_overlap(tmp_path, {"context": index})
    words_quoted = 20
    assert set(hits) == {"quote.jsonl"}
    assert hits["quote.jsonl"]["context_longest_words"] == words_quoted
    assert hits["quote.jsonl"]["context_shingles"] == words_quoted - 11


def test_select_files_skips_ignored_env_and_caches(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text("benchmark_runs/_llm_cache/\n", encoding="utf-8")
    (tmp_path / "code.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / ".env").write_text("K=v\n", encoding="utf-8")
    (tmp_path / ".env.local").write_text("K=v\n", encoding="utf-8")
    (tmp_path / ".env.example").write_text("K=\n", encoding="utf-8")
    (tmp_path / "benchmark_runs" / "_llm_cache").mkdir(parents=True)
    (tmp_path / "benchmark_runs" / "_llm_cache" / "k.json").write_text("{}", encoding="utf-8")
    (tmp_path / "benchmark_runs" / "_emb_cache").mkdir(parents=True)
    (tmp_path / "benchmark_runs" / "_emb_cache" / "v.sqlite").write_text("", encoding="utf-8")
    names = {p.as_posix() for p in select_files(tmp_path)}
    assert names == {".env.example", ".gitignore", "code.py"}


def test_placeholders_are_listed_but_do_not_block(tmp_path):
    (tmp_path / "t.py").write_text('K = "sk-' + 'test-secret-value-1234567890"\nG = "AIza'
                                   + 'SyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB"\n', encoding="utf-8")
    findings = scan_secrets(tmp_path)
    assert {f["kind"] for f in findings} == {"openai_key:placeholder", "google_api_key:placeholder"}
    assert blocking(findings) == []
    assert not looks_like_placeholder(FAKE_OPENAI.encode())


def test_keys_committed_in_history_become_known_values(tmp_path):
    import os

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    leaked = "AIza" + "Sy9fQ2kLm7Xw3Rt8Zp1Vc6Nb4Hd0Gj5Ks2E"
    ident = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}

    def run(*cmd):
        subprocess.run(list(cmd), cwd=tmp_path, check=True, capture_output=True, env={**os.environ, **ident})

    (tmp_path / "cfg.txt").write_text("GEMINI=" + leaked, encoding="utf-8")
    run("git", "add", "cfg.txt")
    run("git", "commit", "-q", "-m", "leak")
    (tmp_path / "cfg.txt").write_text("GEMINI=", encoding="utf-8")
    run("git", "commit", "-q", "-am", "remove")
    values = history_secret_values(tmp_path)
    assert values == [leaked.encode()]
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "copy.md").write_text("old key: " + leaked, encoding="utf-8")
    findings = scan_secrets(tmp_path / "out", values)
    assert "known_secret_0" in {f["kind"] for f in findings} and blocking(findings)


def test_assignment_fixtures_in_tests_are_not_treated_as_leaked_values(tmp_path):
    import os

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    ident = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text("MY_API_KEY = 'plain-fixture-value-98765'", encoding="utf-8")
    (tmp_path / "cfg.env").write_text("SERVICE_TOKEN=Rq7vLm2Xk9Tz4Wb8Hn3Pc6Yd", encoding="utf-8")
    for cmd in (["git", "add", "."], ["git", "commit", "-q", "-m", "c"]):
        subprocess.run(cmd, cwd=tmp_path, check=True, capture_output=True, env={**os.environ, **ident})
    values = history_secret_values(tmp_path)
    assert b"Rq7vLm2Xk9Tz4Wb8Hn3Pc6Yd" in values
    assert not any(b"plain-fixture-value" in v for v in values)
