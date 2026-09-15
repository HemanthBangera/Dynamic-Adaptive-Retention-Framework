"""E1 method selection (incl. reader-only baselines) and the pre-registration hash in provenance."""

import hashlib

import pytest

from benchmarks.dars_eval.provenance import PREREGISTRATION, collect_provenance, preregistration_sha256
from benchmarks.dars_eval.run_static import default_methods, reader_only_methods, select_methods


def test_select_methods_defaults_to_the_e1_grid():
    assert [m.name for m in select_methods("")] == [m.name for m in default_methods()]
    assert all(m.kind not in ("none", "full") for m in select_methods(""))


def test_select_methods_accepts_reader_only_baselines_in_canonical_order():
    chosen = select_methods("full_context,no_memory,similarity")
    assert [m.name for m in chosen] == ["similarity", "no_memory", "full_context"]
    assert {m.name: m.kind for m in reader_only_methods()} == {"no_memory": "none", "full_context": "full"}


def test_select_methods_rejects_unknown_names():
    with pytest.raises(ValueError):
        select_methods("similarity,oracle_magic")


def test_provenance_records_preregistration_hash():
    digest = preregistration_sha256()
    assert digest == hashlib.sha256(PREREGISTRATION.read_bytes()).hexdigest()
    assert len(digest) == 64
    assert collect_provenance()["preregistration_sha256"] == digest
