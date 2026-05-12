"""Unit tests for narrative MemoryAgentBench stack (tombstones, chunking, reranker)."""

import asyncio
import time

import pytest

from benchmarks.memory_agent_bench.chunking import chunk_context
from config.settings import DARSConfig
from core.layer_a.reformulator import QueryReformulator, _is_degenerate_expansion
from core.layer_a.reranker import DARSReranker
from core.layer_d.schema import MemoryPayload, MemoryPoint
from core.layer_d.storage import MemoryVault, chunk_index_from_tags


def test_reformulator_whitespace_skips_llm():
    r = QueryReformulator()

    async def _run() -> None:
        assert await r.reformulate_query("") == ""
        assert await r.reformulate_query("   \n\t  ") == "   \n\t  "

    asyncio.run(_run())


def test_is_degenerate_expansion():
    long_q = "What happens next in the story? " * 8
    assert _is_degenerate_expansion(long_q, "ok") is True
    assert _is_degenerate_expansion(long_q, "after sitting on porch Debbie green dress next event") is False


def test_chunk_index_from_tags():
    p = MemoryPayload(text_content="x", tags=["mab", "chunk:7"])
    assert chunk_index_from_tags(p.tags) == 7
    assert chunk_index_from_tags([]) is None


def test_chunk_overlap_produces_more_or_equal_chunks():
    text = "First sentence here. Second sentence here. " * 40
    a = chunk_context(text, chunk_size=80, tiktoken_model="gpt-4o-mini", overlap_tokens=0)
    b = chunk_context(text, chunk_size=80, tiktoken_model="gpt-4o-mini", overlap_tokens=32)
    assert len(b) >= len(a) >= 1


@pytest.mark.usefixtures("vault")
def test_superseded_excluded_from_semantic_search(vault: MemoryVault):
    vault.store_memory(
        "alpha unique z9q2 story beat one",
        source="tomb",
        tags=["chunk:0"],
    )
    pid1 = vault.store_memory(
        "alpha unique z9q2 story beat two extended",
        source="tomb",
        tags=["chunk:2"],
    )
    vault.patch_payload(pid1, {"superseded": True})
    time.sleep(0.3)
    hits = vault.semantic_search("alpha unique z9q2", top_k=5, exclude_superseded=True)
    ids = {h.point_id for h in hits}
    assert pid1 not in ids


def test_rerank_neighbor_expand_requires_tags(vault: MemoryVault):
    vault.store_memory("dogs run fast", tags=["chunk:0"])
    vault.store_memory("dogs run in the park", tags=["chunk:1"])
    vault.store_memory("cats sleep", tags=["chunk:2"])
    time.sleep(0.3)
    r = DARSReranker(vault=vault)
    out = r.rerank("dogs park", fetch_k=6, top_n=1, expand_neighbor_chunks=True)
    assert len(out) >= 1
    indices = {chunk_index_from_tags(m.payload.tags) for m in out}
    assert 0 in indices or 1 in indices or 2 in indices


def test_dual_query_retrieval_merges(vault: MemoryVault):
    prev = DARSConfig.MAB_DUAL_QUERY_RETRIEVAL
    DARSConfig.MAB_DUAL_QUERY_RETRIEVAL = True
    try:
        vault.store_memory("unique alpha zzztoken one", tags=["chunk:0"])
        vault.store_memory("unique beta zzztoken two", tags=["chunk:1"])
        time.sleep(0.35)
        r = DARSReranker(vault=vault)
        out = r.rerank(
            "alpha zzztoken",
            fetch_k=8,
            top_n=4,
            expand_neighbor_chunks=False,
            secondary_query="beta zzztoken",
        )
        blob = " ".join(m.payload.text_content for m in out)
        assert "alpha" in blob and "beta" in blob
    finally:
        DARSConfig.MAB_DUAL_QUERY_RETRIEVAL = prev


def test_prompt_constructor_orders_by_chunk_index():
    from core.layer_a.prompt_constructor import PromptConstructor

    m2 = MemoryPoint(
        point_id="b",
        vector=[],
        payload=MemoryPayload(text_content="second", tags=["chunk:2"]),
        dars_score=0.9,
    )
    m0 = MemoryPoint(
        point_id="a",
        vector=[],
        payload=MemoryPayload(text_content="first", tags=["chunk:0"]),
        dars_score=0.5,
    )
    xml = PromptConstructor.build("q?", [m2, m0])
    pos0 = xml.index("first")
    pos2 = xml.index("second")
    assert pos0 < pos2
