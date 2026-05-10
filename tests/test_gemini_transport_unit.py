"""Unit tests for Gemini key parsing and governed transport (mocked HTTP)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.gemini_transport import (
    GovernedGeminiTransport,
    _is_quota_or_billing_block,
    extract_gemini_api_keys_from_text,
    line_looks_like_gemini_api_key,
    parse_gemini_keys_file,
    redact_key_suffix,
)


def test_redact_key_suffix():
    assert redact_key_suffix("abcdefghijklmnop") == "....mnop"


def test_is_quota_or_billing_block():
    assert _is_quota_or_billing_block('{"error":{"message":"You exceeded your current quota"}}')
    assert not _is_quota_or_billing_block("rate limit try later")


def test_line_looks_like_gemini_api_key():
    assert line_looks_like_gemini_api_key("AIzaSy0123456789012345678901234567890ABCDE")
    assert not line_looks_like_gemini_api_key("API Key")
    assert not line_looks_like_gemini_api_key("API key details")
    assert not line_looks_like_gemini_api_key("projects/104127163399")
    assert not line_looks_like_gemini_api_key("104127163399")


def test_parse_gemini_keys_file_filters_gcp_export(tmp_path):
    p = tmp_path / "keys.txt"
    p.write_text(
        "API key details\nAPI Key\nAIzaSyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n"
        "Name\nMy key\nProject name\nprojects/1\n# comment\n"
        "AIzaSyBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB\n",
        encoding="utf-8",
    )
    keys = parse_gemini_keys_file(str(p))
    assert keys == [
        "AIzaSyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "AIzaSyBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
    ]


def test_parse_gemini_keys_file(tmp_path):
    p = tmp_path / "k.txt"
    p.write_text(
        "# c\n\n"
        "AIzaSyCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC\n"
        "AIzaSyDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD\n",
        encoding="utf-8",
    )
    assert parse_gemini_keys_file(str(p)) == [
        "AIzaSyCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
        "AIzaSyDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD",
    ]


def test_extract_embedded_api_key_lines():
    raw = """
1st one
Api key :   AIzaSyEeEeEeEeEeEeEeEeEeEeEeEeEeEeEeEe
Apu key : AIzaSyFfFfFfFfFfFfFfFfFfFfFfFfFfFfFfFf
"""
    keys = extract_gemini_api_keys_from_text(raw)
    assert keys == [
        "AIzaSyEeEeEeEeEeEeEeEeEeEeEeEeEeEeEeEe",
        "AIzaSyFfFfFfFfFfFfFfFfFfFfFfFfFfFfFfFf",
    ]


@pytest.mark.asyncio
async def test_governed_transport_min_interval():
    keys = ["key-one", "key-two"]
    t = GovernedGeminiTransport(
        keys=keys,
        model="gemini-2.5-flash-lite",
        timeout=30.0,
        min_interval_s=0.05,
        max_retries=2,
    )
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(
        return_value={"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}
    )
    post_cm = MagicMock()
    post_cm.__aenter__ = AsyncMock(return_value=mock_resp)
    post_cm.__aexit__ = AsyncMock(return_value=None)

    class FakeSession:
        closed = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def post(self, *a, **k):
            return post_cm

        async def close(self):
            self.closed = True

    with patch("core.gemini_transport.aiohttp.ClientSession", return_value=FakeSession()):
        out1, i1 = await t.generate_text("p1")
        out2, i2 = await t.generate_text("p2")
    await t.aclose()

    assert out1 == "hi"
    assert out2 == "hi"
    assert i1 == 0
    assert i2 == 0
