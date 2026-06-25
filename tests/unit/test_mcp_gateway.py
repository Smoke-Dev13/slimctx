"""Tests for the MCP compression gateway."""

from __future__ import annotations

import json

import pytest

from contextly.ccr import CCRStore
from contextly.mcp_gateway import build_gateway_server, build_router, compress_payload

_BIG_LOG = "\n".join(
    f"2026-06-22T10:00:{i % 60:02d} INFO GET /api/x/{i} -> 200 in {i % 50}ms" for i in range(300)
)
_JSON = json.dumps([{"id": i, "city": "Tbilisi", "plan": "gold"} for i in range(60)])


# ── compress_payload (pure) ─────────────────────────────────────────────────────


def test_compress_payload_small_passthrough() -> None:
    store = CCRStore()
    text, ref = compress_payload("hi there", build_router(), store)
    assert text == "hi there"
    assert ref is None


def test_compress_payload_lossless_json_has_no_ref() -> None:
    store = CCRStore()
    text, ref = compress_payload(_JSON, build_router(), store)
    assert ref is None  # lossless table → nothing dropped, no expand needed
    assert len(text) < len(_JSON)
    assert "expand(" not in text


def test_compress_payload_logs_have_ref_and_hint() -> None:
    store = CCRStore()
    text, ref = compress_payload(_BIG_LOG, build_router(), store)
    assert ref is not None
    assert len(text) < len(_BIG_LOG)
    assert f'expand("{ref}")' in text
    assert store.retrieve(ref) == _BIG_LOG  # original recoverable


# ── End-to-end gateway over the in-memory MCP transport ─────────────────────────


@pytest.mark.asyncio
async def test_gateway_forwards_tools_compresses_and_expands() -> None:
    import mcp.types as types
    from mcp.server import Server
    from mcp.shared.memory import create_connected_server_and_client_session as connect

    downstream: Server = Server("downstream")

    @downstream.list_tools()
    async def _lt() -> list[types.Tool]:
        return [
            types.Tool(
                name="get_logs",
                description="Return server logs",
                inputSchema={"type": "object", "properties": {}},
            )
        ]

    @downstream.call_tool()
    async def _ct(name: str, arguments: dict[str, object]) -> list[types.ContentBlock]:
        return [types.TextContent(type="text", text=_BIG_LOG)]

    store = CCRStore()
    async with connect(downstream) as down_session:
        gateway = build_gateway_server(down_session, store, build_router())
        async with connect(gateway) as gw:
            # Downstream tool is forwarded, plus the injected expand tool.
            tools = {t.name for t in (await gw.list_tools()).tools}
            assert "get_logs" in tools
            assert "expand" in tools

            # Tool output is compressed, with an expand hint.
            result = await gw.call_tool("get_logs", {})
            text = result.content[0].text  # type: ignore[union-attr]
            assert len(text) < len(_BIG_LOG)
            assert 'expand("' in text

            # The agent can expand the ref back to the full original.
            ref = text.split('expand("', 1)[1].split('"')[0]
            expanded = await gw.call_tool("expand", {"ref": ref})
            restored = expanded.content[0].text  # type: ignore[union-attr]
            assert restored == _BIG_LOG

            # Granular expand: only matching lines.
            filtered = await gw.call_tool("expand", {"ref": ref, "contains": "/api/x/7 "})
            lines = filtered.content[0].text.splitlines()  # type: ignore[union-attr]
            assert lines and all("/api/x/7 " in ln for ln in lines)
