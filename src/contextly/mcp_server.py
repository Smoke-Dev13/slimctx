"""MCP (Model Context Protocol) server exposing Contextly tools.

Exposes three tools to an MCP-compatible client (e.g. Claude Desktop):

  compress_text(content, query="")
      Compress content, store the original in the CCR store, return a dict
      with the compressed text, a CCR retrieval key, and compression metrics.

  retrieve_original(key)
      Retrieve the original content from the CCR store by its key.

  compression_stats()
      Return CCR store statistics (entries, hit rate, etc.).

And one resource:

  contextly://info   — human-readable server description.

For testing, use the private ``_compress`` / ``_retrieve`` helpers directly,
passing explicit ``CCRStore`` and ``ContentRouter`` instances so tests stay
isolated from the module-level singletons.

Run as a stdio MCP server:
    contextly mcp            # via the CLI
    # or directly:
    python -m contextly.mcp_server
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog

from contextly.ccr import CCRStore
from contextly.compressors.code import CodeCompressor
from contextly.compressors.json_table import JsonTableCompressor
from contextly.compressors.logs import LogCompressor
from contextly.compressors.prose import ProseCompressor
from contextly.compressors.registry import ContentRouter
from contextly.expand import filter_original

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as _err:  # pragma: no cover
    raise ImportError(
        "The 'mcp' package is required for MCP server mode. "
        "Install it with: pip install 'contextly[mcp-server]'"
    ) from _err

logger = structlog.get_logger(__name__)


# ── Default module-level singletons (used by CLI) ─────────────────────────────


def _build_default_router() -> ContentRouter:
    router = ContentRouter()
    router.register(JsonTableCompressor())
    router.register(CodeCompressor())
    router.register(LogCompressor())
    router.register(ProseCompressor())
    return router


_default_store: CCRStore = CCRStore()
_default_router: ContentRouter = _build_default_router()


# ── Testable implementation helpers ───────────────────────────────────────────


async def _compress(
    content: str,
    query: str,
    store: CCRStore,
    router: ContentRouter,
) -> dict[str, Any]:
    """Core implementation for the compress_text MCP tool.

    Args:
        content: Text to compress.
        query: Optional user query for aggressiveness tuning.
        store: CCRStore instance to persist the original.
        router: ContentRouter for compressor selection.

    Returns:
        Dict with compressed, ccr_key, original_chars, compressed_chars,
        compression_ratio, and compressor.
    """
    compressor = router.select(content, query)
    result = compressor.compress(content, query)
    ccr_key = store.store(content)

    logger.info(
        "mcp_compress",
        compressor=result.compressor_name,
        ratio=round(result.compression_ratio, 3),
        ccr_key=ccr_key,
    )

    lossless = bool(result.metadata.get("lossless"))
    expandable = result.compressed_length < result.original_length and not lossless

    return {
        "compressed": result.content,
        "ccr_key": ccr_key,
        "original_chars": result.original_length,
        "compressed_chars": result.compressed_length,
        "compression_ratio": round(result.compression_ratio, 4),
        "compressor": result.compressor_name,
        "metadata": result.metadata,
        # Expand-on-demand: when compression dropped information, the agent can
        # pull the full original back by calling expand(expand_ref).
        "expandable": expandable,
        "expand_ref": ccr_key if expandable else None,
        "hint": (
            f"Content was compressed with loss. Call expand('{ccr_key}') for the "
            f"full original, or expand('{ccr_key}', contains='...') to pull back "
            "just the matching records/lines."
            if expandable
            else None
        ),
    }


async def _retrieve(key: str, store: CCRStore) -> str:
    """Core implementation for the retrieve_original MCP tool.

    Args:
        key: The 16-character hex key returned by compress_text.
        store: CCRStore instance to look up the original.

    Returns:
        Original text, or an error message if the key is absent / evicted.
    """
    original = store.retrieve(key)
    if original is None:
        logger.warning("mcp_retrieve_miss", key=key)
        msg = f"Key '{key}' not found in CCR store. It may have been evicted."
        return json.dumps({"error": msg})
    logger.info("mcp_retrieve_hit", key=key)
    return original


# ── FastMCP server ────────────────────────────────────────────────────────────

mcp: FastMCP = FastMCP(
    "contextly",
    instructions=(
        "Contextly compresses context to reduce LLM token costs. "
        "Use compress_text() before sending large content to an LLM; "
        "call retrieve_original() with the returned ccr_key to recover the full text."
    ),
)


@mcp.tool(
    name="compress_text",
    description=(
        "Compress text content (prose, JSON, or code) to reduce token count. "
        "Returns the compressed text plus a ccr_key that can be passed to "
        "retrieve_original() to recover the original. "
        "Set query to the user's question for better compression targeting."
    ),
)
async def compress_text(content: str, query: str = "") -> dict[str, Any]:
    """Compress content and store the original for retrieval."""
    return await _compress(content, query, _default_store, _default_router)


@mcp.tool(
    name="retrieve_original",
    description=(
        "Retrieve the original (pre-compression) content for a given ccr_key. "
        "Use this when the full original text is needed after compression."
    ),
)
async def retrieve_original(key: str) -> str:
    """Return the original content stored under *key*, or an error JSON string."""
    return await _retrieve(key, _default_store)


@mcp.tool(
    name="expand",
    description=(
        "Expand a compressed result back to its original. Pass the expand_ref "
        "(or ccr_key) from compress_text. Provide 'contains' to pull back only "
        "the matching records (JSON) or lines (logs/text) instead of everything "
        "— granular recovery of just the detail you need."
    ),
)
async def expand(ref: str, contains: str = "") -> str:
    """Return the original for an expand_ref / ccr_key, optionally filtered."""
    original = await _retrieve(ref, _default_store)
    if contains:
        filtered, _ = filter_original(original, contains)
        return filtered
    return original


@mcp.tool(
    name="compression_stats",
    description="Return statistics for the in-process CCR store (entries, hit rate, etc.).",
)
async def compression_stats() -> dict[str, Any]:
    """Return a snapshot of CCR store metrics."""
    return _default_store.stats()


@mcp.resource("contextly://info")
async def server_info() -> str:
    """Human-readable Contextly server description."""
    stats = _default_store.stats()
    return (
        "Contextly MCP Server\n"
        "====================\n"
        "Tools: compress_text, retrieve_original, compression_stats\n"
        f"CCR store: {stats['current_entries']}/{stats['max_entries']} entries, "
        f"hit rate {stats['hit_rate']:.1%}"
    )


# ── Standalone entrypoint ─────────────────────────────────────────────────────

if __name__ == "__main__":  # pragma: no cover
    asyncio.run(mcp.run_stdio_async())
