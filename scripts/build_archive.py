"""
Build the publication archive (for Zenodo) from the working tree, and check it before release.

The archive holds the code, the run artifacts, the pre-registration, both addenda, the deviations log,
the embedding cache (vectors keyed by hashes, no text) and a redacted LLM response cache
(``scripts/redact_cache.py``: every prompt is replaced by its SHA-256; responses are kept so every
number replays once the prompts are rebuilt from the datasets).

Three checks run on the built tree, and the build fails if the first two find anything:

* secrets: provider key patterns, private keys, and the literal values of the keys in a local ``.env``
  (values are compared, never printed);
* the redacted cache: no record may still carry message content;
* novel text: MemoryAgentBench's novel-derived sources (EventQA, DetectiveQA and the other
  Long-Range-Understanding novels) have no stated licence for the novels themselves, so every archived
  text file is scanned for any 12-word span of those novels ("context") and, separately, of the benchmark's
  own questions, answers and event lists ("qa", MIT). Matches are reported per file with the longest quoted
  span in words (model outputs may quote a phrase of the text they were shown).

    python scripts/build_archive.py --out ../dars_archive --llm-cache benchmark_runs/_llm_cache \
        --emb-cache benchmark_runs/_emb_cache --env ../.env
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from redact_cache import redact_record  # noqa: E402

SHINGLE = 12
TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".txt", ".log", ".py", ".sh", ".csv", ".tsv", ".yaml", ".yml",
                 ".toml", ".cfg", ".ini", ".diff", ".tex", ".bib", ".html"}
EXCLUDED_PARTS = {".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache", "_llm_cache", "_emb_cache"}
EXCLUDED_NAMES = {".env", "keys.txt"}
SECRET_PATTERNS = {
    "openai_key": re.compile(rb"(?<![A-Za-z0-9_-])sk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_-]{20,}"),
    "google_api_key": re.compile(rb"(?<![A-Za-z0-9])A[Il1]za[0-9A-Za-z_-]{35}"),
    "aws_access_key_id": re.compile(rb"(?<![A-Z0-9])(?:AKIA|ASIA)[0-9A-Z]{16}(?![A-Z0-9])"),
    "aws_secret_assignment": re.compile(rb"aws_secret_access_key\s*[=:]\s*\S{20,}", re.I),
    "private_key_block": re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "huggingface_token": re.compile(rb"(?<![A-Za-z0-9_])hf_[A-Za-z0-9]{30,}"),
    "github_token": re.compile(rb"(?<![A-Za-z0-9_])gh[pousr]_[A-Za-z0-9]{36}"),
}
NOVEL_SPLITS = {"Accurate_Retrieval": ("eventqa",), "Long_Range_Understanding": ("",)}   # "" = every source


def select_files(repo: Path) -> List[Path]:
    """Tracked and untracked files that git does not ignore, minus local secrets and caches."""
    out = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=repo,
                         check=True, capture_output=True).stdout.decode("utf-8")
    files = []
    for rel in sorted(set(filter(None, out.split("\0")))):
        path = Path(rel)
        env_like = path.name.startswith(".env") and path.name != ".env.example"
        if path.name in EXCLUDED_NAMES or env_like or EXCLUDED_PARTS & set(path.parts):
            continue
        if (repo / path).is_file():
            files.append(path)
    return files


def copy_tree(repo: Path, files: Sequence[Path], out: Path) -> int:
    for rel in files:
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / rel, dst)
    return len(files)


def unredacted_records(cache: Path) -> List[str]:
    bad = []
    for path in cache.rglob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        messages = (record.get("request") or {}).get("messages") or []
        if not record.get("redacted") or any("content" in m for m in messages):
            bad.append(str(path.relative_to(cache)))
    return bad


def env_secret_values(env_file: Optional[Path]) -> List[bytes]:
    """Values of KEY/TOKEN/SECRET/PASSWORD entries in a dotenv file (kept in memory only)."""
    if env_file is None or not env_file.is_file():
        return []
    values = []
    for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        name, value = line.split("=", 1)
        value = value.strip().strip("'\"")
        if re.search(r"KEY|TOKEN|SECRET|PASSWORD", name, re.I) and len(value) >= 12:
            values.append(value.encode("utf-8"))
    return values


def looks_like_placeholder(match: bytes) -> bool:
    """Test fixtures such as "sk-test-..." or "AIza...AAAAAAAA": marker words, long runs or low diversity."""
    return bool(re.search(rb"test|fake|dummy|example|placeholder|redacted|xxxx", match, re.I)
                or re.search(rb"(.)\1{5}", match) or re.search(rb"0123456789|1234567890", match)
                or len(set(match)) < 16)


ASSIGNMENT = re.compile(rb"([A-Za-z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD)[A-Za-z0-9_]*)[ \t]*[=:][ \t]*[\"']?([^\s\"',]{20,})",
                        re.I)


def history_secret_values(repo: Path) -> List[bytes]:
    """Every non-placeholder key-shaped string or KEY/TOKEN/SECRET assignment value ever committed, so a key
    leaked in history cannot reappear in the archive.

    Provider key patterns are collected from every file. Assignment values are not collected from ``tests/``,
    whose fixtures assign made-up values to KEY variables on purpose (the secret scanner's own tests among them);
    a real provider key committed there is still caught by its pattern."""
    log = subprocess.run(["git", "log", "-p", "--all", "--no-color"], cwd=repo, capture_output=True).stdout
    found = set()
    for pattern in SECRET_PATTERNS.values():
        for m in pattern.finditer(log):
            if not looks_like_placeholder(m.group(0)):
                found.add(m.group(0))
    for block in re.split(rb"(?m)^(?=diff --git )", log):
        header = re.match(rb"diff --git a/(\S+)", block)
        if header and header.group(1).startswith(b"tests/"):
            continue
        for m in ASSIGNMENT.finditer(block):
            value = m.group(2)
            if (not looks_like_placeholder(value) and re.fullmatch(rb"[A-Za-z0-9_\-]{20,}", value)
                    and not re.fullmatch(rb"[a-z_]+", value)):
                found.add(value)
    return sorted(found)


def scan_secrets(root: Path, literals: Sequence[bytes] = ()) -> List[Dict[str, str]]:
    """Findings as {file, kind}; the matched text is never returned. Placeholder-like matches get a
    ``:placeholder`` suffix and do not fail the build; known secret values always do."""
    findings = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        data = path.read_bytes()
        rel = str(path.relative_to(root))
        for kind, pattern in SECRET_PATTERNS.items():
            for m in pattern.finditer(data):
                suffix = ":placeholder" if looks_like_placeholder(m.group(0)) else ""
                findings.append({"file": rel, "kind": kind + suffix})
        for i, value in enumerate(literals):
            if value in data:
                findings.append({"file": rel, "kind": f"known_secret_{i}"})
    unique = {(f["file"], f["kind"]): f for f in findings}
    return list(unique.values())


def blocking(findings: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    return [f for f in findings if not f["kind"].endswith(":placeholder")]


_WORD = re.compile(r"[a-z0-9]+")


def words(text: str) -> List[str]:
    return _WORD.findall(text.lower())


def shingles(text: str, n: int = SHINGLE) -> Iterator[int]:
    w = words(text)
    for i in range(len(w) - n + 1):
        yield int.from_bytes(hashlib.blake2b(" ".join(w[i:i + n]).encode("utf-8"), digest_size=8).digest(), "big")


def build_index(texts: Iterable[str], n: int = SHINGLE) -> Set[int]:
    index: Set[int] = set()
    for text in texts:
        index.update(shingles(text, n))
    return index


def novel_texts(revision: str = "main") -> Iterator[Tuple[str, str]]:
    """(kind, text) for the novel-derived MemoryAgentBench sources: kind "context" is the novel itself; kind "qa" is
    the benchmark's own questions, answers and previous-event lists (MIT, written by the benchmark authors)."""
    from datasets import load_dataset

    for split, prefixes in NOVEL_SPLITS.items():
        for row in load_dataset("ai-hyz/MemoryAgentBench", split=split, revision=revision):
            meta = row.get("metadata") or {}
            if not str(meta.get("source", "")).startswith(prefixes):
                continue
            yield "context", row.get("context") or ""
            for field in (row.get("questions"), row.get("answers"), meta.get("previous_events")):
                for item in field if isinstance(field, list) else [field]:
                    yield "qa", json.dumps(item, ensure_ascii=False) if not isinstance(item, str) else item


def longest_run(positions: Sequence[int]) -> int:
    """Length of the longest run of consecutive integers."""
    best = run = 0
    prev = None
    for p in positions:
        run = run + 1 if prev is not None and p == prev + 1 else 1
        best, prev = max(best, run), p
    return best


def scan_overlap(root: Path, indexes: Dict[str, Set[int]], n: int = SHINGLE) -> Dict[str, Dict[str, int]]:
    """Per text file with any match: distinct matched shingles and the longest quoted span (in words) per index."""
    hits: Dict[str, Dict[str, int]] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES):
        grams = list(shingles(path.read_text(encoding="utf-8", errors="replace"), n))
        entry = {}
        for kind, index in indexes.items():
            pos = [i for i, h in enumerate(grams) if h in index]
            if pos:
                entry[f"{kind}_shingles"] = len({grams[i] for i in pos})
                entry[f"{kind}_longest_words"] = longest_run(pos) + n - 1
        if entry:
            hits[str(path.relative_to(root))] = entry
    return hits


def write_checksums(root: Path) -> int:
    lines = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


DATA_NOTICE = """# Third-party data in this archive

Run artifacts contain short excerpts (questions, answers, memory units, compressed memories) and model
outputs derived from the following datasets, used and redistributed under their licences
(checked 2026-09-13):

| Dataset | Used for | Licence |
|---|---|---|
| MemoryAgentBench (ai-hyz/MemoryAgentBench) | all MAB sources | MIT |
| LongMemEval | LongMemEval (S*) | MIT |
| RULER (SQuAD 2.0 and HotpotQA passages) | RULER QA1 / QA2 | Apache-2.0 (RULER); CC BY-SA 4.0 (SQuAD 2.0, HotpotQA) |
| MQuAKE | FactConsolidation | MIT |
| Banking77 | ICL Banking77 | CC BY 4.0 |
| Multi-Session Chat (via ParlAI) | MSC | MIT (ParlAI) |
| ALFWorld | ALFWorld | MIT |
| EventQA, DetectiveQA (novels) | reader outputs only | novels' copyright not stated |

No context, question or answer text of the novel-derived sources (EventQA, DetectiveQA) is included:
their artifacts hold identifiers, scores and model outputs only, and the LLM response cache stores a
SHA-256 in place of every prompt. `ARCHIVE_REPORT.json` lists every file in which a 12-word span of
those novels was found (model outputs can quote a phrase of the text they were shown).

Excerpts from CC BY-SA 4.0 sources remain under CC BY-SA 4.0.
"""


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--repo", default=".")
    p.add_argument("--out", required=True)
    p.add_argument("--llm-cache", action="append", default=[],
                   help="LLM cache directory; repeat to merge several (later ones do not overwrite)")
    p.add_argument("--emb-cache")
    p.add_argument("--env", help="dotenv file whose secret values must not appear in the archive")
    p.add_argument("--mab-revision", default="main")
    p.add_argument("--skip-novel-scan", action="store_true")
    args = p.parse_args(argv)

    repo, out = Path(args.repo).resolve(), Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    report: Dict[str, object] = {}

    files = select_files(repo)
    report["files_copied"] = copy_tree(repo, files, out)
    report["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                                        text=True).stdout.strip()
    report["git_dirty"] = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo,
                                              capture_output=True, text=True).stdout.strip())

    cache_out = out / "benchmark_runs" / "_llm_cache"
    counts = Counter()
    for src in args.llm_cache:
        for path in sorted(Path(src).rglob("*.json")):
            target = cache_out / path.relative_to(src)
            if target.exists():
                counts["duplicate_keys_skipped"] += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            record = json.loads(path.read_text(encoding="utf-8"))
            target.write_text(json.dumps(redact_record(record), ensure_ascii=False), encoding="utf-8")
            counts["records"] += 1
    report["llm_cache"] = dict(counts)
    if args.emb_cache:
        shutil.copytree(args.emb_cache, out / "benchmark_runs" / "_emb_cache")
    (out / "DATA_NOTICE.md").write_text(DATA_NOTICE, encoding="utf-8")

    report["unredacted_cache_records"] = unredacted_records(cache_out) if cache_out.exists() else []
    literals = env_secret_values(Path(args.env) if args.env else None) + history_secret_values(repo)
    report["known_secret_values_checked"] = len(literals)
    report["secret_findings"] = scan_secrets(out, literals)
    report["blocking_secret_findings"] = blocking(report["secret_findings"])
    if not args.skip_novel_scan:
        indexes: Dict[str, Set[int]] = {"context": set(), "qa": set()}
        for kind, text in novel_texts(args.mab_revision):
            indexes[kind].update(shingles(text))
        indexes["qa"] -= indexes["context"]          # a span found in the novel counts as novel text
        report["novel_shingles_indexed"] = {k: len(v) for k, v in indexes.items()}
        report["novel_overlap_by_file"] = scan_overlap(out, indexes)
        ctx = {f: e for f, e in report["novel_overlap_by_file"].items() if "context_shingles" in e}
        report["novel_context_overlap_summary"] = {
            "files": len(ctx), "longest_words": max((e["context_longest_words"] for e in ctx.values()), default=0),
            "files_over_50_words": sorted(f for f, e in ctx.items() if e["context_longest_words"] > 50)}
    (out / "ARCHIVE_REPORT.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    report["checksummed_files"] = write_checksums(out)

    summary = {k: (len(v) if isinstance(v, (list, dict)) and k != "llm_cache" else v) for k, v in report.items()}
    print(json.dumps(summary, indent=1))
    if report["blocking_secret_findings"] or report["unredacted_cache_records"]:
        raise SystemExit("archive check FAILED: see ARCHIVE_REPORT.json")


if __name__ == "__main__":
    main()
