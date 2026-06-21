"""Unit tests for the MCP server tools.

We test the underlying ``_compress`` / ``_retrieve`` helper functions directly,
passing explicit CCRStore and ContentRouter instances for isolation.
A smoke test verifies the module-level ``mcp`` FastMCP instance exists and
has the expected tools registered.
"""

from __future__ import annotations

import pytest

from contextly.ccr import CCRStore
from contextly.compressors.registry import ContentRouter
from contextly.mcp_server import _compress, _retrieve, mcp

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def fresh_store() -> CCRStore:
    return CCRStore()


@pytest.fixture
def router() -> ContentRouter:
    from contextly.compressors.code import CodeCompressor
    from contextly.compressors.json_smart import JsonSmartCompressor
    from contextly.compressors.prose import ProseCompressor

    r = ContentRouter()
    r.register(JsonSmartCompressor())
    r.register(CodeCompressor())
    r.register(ProseCompressor())
    return r


# ── _compress ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_compress_returns_required_keys(fresh_store: CCRStore, router: ContentRouter) -> None:
    result = await _compress("Hello world!", "", fresh_store, router)
    required = {
        "compressed",
        "ccr_key",
        "original_chars",
        "compressed_chars",
        "compression_ratio",
        "compressor",
        "metadata",
    }
    assert required.issubset(result.keys())


@pytest.mark.asyncio
async def test_compress_stores_original_in_ccr(
    fresh_store: CCRStore, router: ContentRouter
) -> None:
    content = "Hello world, this is a test message."
    result = await _compress(content, "", fresh_store, router)
    key = result["ccr_key"]
    assert fresh_store.retrieve(key) == content


@pytest.mark.asyncio
async def test_compress_ccr_key_is_16_chars(fresh_store: CCRStore, router: ContentRouter) -> None:
    result = await _compress("short text", "", fresh_store, router)
    assert len(result["ccr_key"]) == 16


@pytest.mark.asyncio
async def test_compress_ratio_between_0_and_1(fresh_store: CCRStore, router: ContentRouter) -> None:
    result = await _compress("Hello world", "", fresh_store, router)
    assert 0.0 < result["compression_ratio"] <= 1.0


@pytest.mark.asyncio
async def test_compress_original_chars_matches_input(
    fresh_store: CCRStore, router: ContentRouter
) -> None:
    content = "Test content for char counting."
    result = await _compress(content, "", fresh_store, router)
    assert result["original_chars"] == len(content)


@pytest.mark.asyncio
async def test_compress_compressor_field_is_string(
    fresh_store: CCRStore, router: ContentRouter
) -> None:
    result = await _compress("some text", "", fresh_store, router)
    assert isinstance(result["compressor"], str)
    assert len(result["compressor"]) > 0


@pytest.mark.asyncio
async def test_compress_prose_reduces_long_text(
    fresh_store: CCRStore, router: ContentRouter
) -> None:
    prose = (
        "The machine learning model processes data efficiently. "
        "It applies various optimization algorithms to minimize error. "
        "The training pipeline includes data augmentation and regularization. "
        "Results are evaluated on a held-out validation dataset. "
        "The final model achieves state-of-the-art performance on benchmarks. "
    ) * 5
    result = await _compress(prose, "summarize this", fresh_store, router)
    assert result["compression_ratio"] < 1.0
    assert result["compressed_chars"] < result["original_chars"]


@pytest.mark.asyncio
async def test_compress_json_array_uses_json_compressor(
    fresh_store: CCRStore, router: ContentRouter
) -> None:
    import json

    records = [{"id": i, "name": f"user_{i}", "score": i * 10} for i in range(50)]
    content = json.dumps(records)
    result = await _compress(content, "", fresh_store, router)
    assert result["compressor"] in ("json_smart", "passthrough")


@pytest.mark.asyncio
async def test_compress_same_content_returns_same_ccr_key(
    fresh_store: CCRStore, router: ContentRouter
) -> None:
    content = "identical content for dedup test"
    r1 = await _compress(content, "", fresh_store, router)
    r2 = await _compress(content, "", fresh_store, router)
    assert r1["ccr_key"] == r2["ccr_key"]


# ── _retrieve ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retrieve_returns_original(fresh_store: CCRStore, router: ContentRouter) -> None:
    content = "Original content to retrieve."
    result = await _compress(content, "", fresh_store, router)
    retrieved = await _retrieve(result["ccr_key"], fresh_store)
    assert retrieved == content


@pytest.mark.asyncio
async def test_retrieve_missing_key_returns_error_json(fresh_store: CCRStore) -> None:
    import json

    response = await _retrieve("0000000000000000", fresh_store)
    data = json.loads(response)
    assert "error" in data
    assert "0000000000000000" in data["error"]


@pytest.mark.asyncio
async def test_retrieve_error_message_is_json_string(fresh_store: CCRStore) -> None:
    import json

    result = await _retrieve("nonexistentkey12", fresh_store)
    # Must be valid JSON
    parsed = json.loads(result)
    assert isinstance(parsed, dict)


# ── Module-level mcp instance ─────────────────────────────────────────────────


def test_mcp_instance_exists() -> None:
    assert mcp is not None


def test_mcp_has_compress_tool() -> None:
    tool_names = [t.name for t in mcp._tool_manager.list_tools()]
    assert "compress_text" in tool_names


def test_mcp_has_retrieve_tool() -> None:
    tool_names = [t.name for t in mcp._tool_manager.list_tools()]
    assert "retrieve_original" in tool_names


def test_mcp_has_stats_tool() -> None:
    tool_names = [t.name for t in mcp._tool_manager.list_tools()]
    assert "compression_stats" in tool_names


def test_mcp_has_three_tools() -> None:
    tools = mcp._tool_manager.list_tools()
    assert len(tools) == 3
