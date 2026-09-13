"""Summarise the archive replay audit into replay_audit.json (read by scripts/build_si.py).

Inputs, all in this folder: replay_audit.log and compare.txt (written by replay_audit.sh), numeric_diff.json
(numeric_diff.py on the outputs that were not byte-identical), the pytest logs in out/, the archive report of the
audited build, and confirm_from_replayed_data.json (the primary confirmatory family recomputed from data regenerated
in the audit, compared with the archived family).

    python benchmark_runs/revision/test/E13_audit/replay_audit/summarise.py
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def pytest_counts(path: Path) -> dict:
    last = [l for l in path.read_text(encoding="utf-8", errors="replace").splitlines() if re.search(r"\d+ passed", l)][-1]
    return {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|skipped|errors?)", last)}


def main() -> None:
    log = (HERE / "replay_audit.log").read_text(encoding="utf-8", errors="replace")
    compare = (HERE / "compare.txt").read_text(encoding="utf-8", errors="replace").splitlines()
    stages = {}
    for line in compare:
        m = re.match(r"\[(\w+)\] (.+?): (.+)$", line)
        if m and not m.group(2).startswith(("PASS", "FAIL")):
            stages.setdefault(m.group(1), {})[m.group(2)] = m.group(3)
    api = [l.strip() for l in compare if "api_calls" in l]
    numeric = json.loads((HERE / "numeric_diff.json").read_text(encoding="utf-8"))
    report = json.loads((HERE / "ARCHIVE_REPORT_audited_build.json").read_text(encoding="utf-8"))
    out = {
        "archive_sha256": re.search(r"archive sha256 ([0-9a-f]{64})", log).group(1),
        "archive_files_unpacked": int(re.search(r"unpacked: (\d+) files", log).group(1)),
        "environment": ("fresh directory and virtual environment (uv; Python 3.14.3; torch 2.11.0 CPU; "
                        "experiments/requirements-vm.txt); datasets, the embedding model and tokenizer encodings "
                        "taken from the host's local download caches; network then blocked through an unreachable "
                        "proxy; placeholder API key; DARS_LLM_OFFLINE=1 for the replays"),
        "archive_checks": {
            "llm_cache_records": report["llm_cache"]["records"],
            "unredacted_cache_records": len(report["unredacted_cache_records"]),
            "blocking_secret_findings": len(report["blocking_secret_findings"]),
            "placeholder_secret_findings": len(report["secret_findings"]) - len(report["blocking_secret_findings"]),
            "novel_context_overlap_files": report["novel_context_overlap_summary"]["files"],
            "novel_context_longest_words": report["novel_context_overlap_summary"]["longest_words"],
            "novel_context_files_over_50_words": len(report["novel_context_overlap_summary"]["files_over_50_words"]),
        },
        "replayed_manifests_api_calls": api,
        "stages": stages,
        "numeric_differences": {k: {f: v[f] for f in ("leaves", "numeric_diffs", "max_abs", "max_rel",
                                                       "non_numeric_count", "non_numeric", "structural_count", "structural")}
                                for k, v in numeric.items()},
        "pytest": {
            "no_key_network_blocked": pytest_counts(HERE / "out" / "pytest_no_key.log"),
            "placeholder_key_archived_cache_network_blocked": pytest_counts(HERE / "out" / "pytest_placeholder_key.log"),
            "first_run_with_offline_mode_forced_for_all_tests": pytest_counts(HERE / "out" / "pytest.log"),
            "note": ("The first run exported DARS_LLM_OFFLINE=1 for the whole suite, which contradicts 18 transport unit "
                     "tests that exercise the online path against a mocked API; they pass without that override. One "
                     "LLM-backed test lacked its requires_llm marker and failed without a key; the marker was added."),
        },
    }
    checks = HERE / "tie_and_ranking_checks.json"
    if checks.exists():
        out["tie_and_ranking_checks"] = json.loads(checks.read_text(encoding="utf-8"))
    val = HERE / "validation_s3_checks.json"
    if val.exists():
        out["validation_s3_checks"] = json.loads(val.read_text(encoding="utf-8"))
    out["numeric_differences_by_field"] = {k: v.get("by_field", {}) for k, v in numeric.items()
                                           if k in ("E1_per_question", "msc_test_s3_facts")}
    final = HERE / "out" / "final_archive_check.log"
    if final.exists():
        text = final.read_text(encoding="utf-8", errors="replace")
        runs = [{k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|skipped)", l)}
                for l in text.splitlines() if re.search(r"\d+ passed", l)]
        out["final_archive_check"] = {
            "archive_sha256": re.search(r"tarball ([0-9a-f]{64})", text).group(1),
            "checksum_mismatches": int(re.search(r"checksums: (\d+) mismatches", text).group(1)),
            "pytest_placeholder_key_archived_cache": runs[0], "pytest_no_key": runs[1],
            "si_regenerated_identically": "identical to submission SI: True" in text,
        }
    confirm = HERE / "confirm_from_replayed_data.json"
    if confirm.exists():
        out["confirm_from_replayed_data"] = json.loads(confirm.read_text(encoding="utf-8"))
    (HERE / "replay_audit.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("stages", "numeric_differences")}, indent=1))


if __name__ == "__main__":
    main()
