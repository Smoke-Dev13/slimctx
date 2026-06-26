"""Tests for the gateway savings trackers and their wiring into tool calls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contextly.ccr import CCRStore
from contextly.gateway_stats import GatewayStats, SQLiteStatsStore
from contextly.mcp_gateway import build_gateway_server, build_router, derive_server_name

_JSON = json.dumps([{"id": i, "city": "Tbilisi", "plan": "gold"} for i in range(60)])


# ── GatewayStats (in-memory) ────────────────────────────────────────────────────


def test_empty_snapshot_is_neutral() -> None:
    snap = GatewayStats().snapshot()
    assert snap["tool_calls_total"] == 0
    assert snap["chars_saved_total"] == 0
    assert snap["compression_ratio_mean"] == 1.0
    assert snap["by_tool"] == {}


def test_record_accumulates_totals_and_per_tool() -> None:
    stats = GatewayStats()
    stats.record("query", 1000, 600)
    stats.record("query", 500, 500)  # no savings
    snap = stats.snapshot()

    assert snap["tool_calls_total"] == 2
    assert snap["tool_calls_compressed"] == 1
    assert snap["chars_before_total"] == 1500
    assert snap["chars_after_total"] == 1100
    assert snap["chars_saved_total"] == 400
    assert snap["tokens_saved_estimate_total"] == 100  # 400 // 4
    assert snap["by_tool"]["query"]["calls"] == 2
    assert snap["by_tool"]["query"]["saved_pct"] == pytest.approx(26.7, abs=0.1)


def test_ratio_is_after_over_before() -> None:
    stats = GatewayStats()
    stats.record("t", 1000, 250)
    assert stats.snapshot()["compression_ratio_mean"] == 0.25


# ── SQLiteStatsStore (shared, multi-process) ────────────────────────────────────


def test_sqlite_store_records_and_labels_by_server(tmp_path: Path) -> None:
    store = SQLiteStatsStore(str(tmp_path / "gw.db"), server="nocodb")
    store.record("queryRecords", 10056, 6442)
    store.record("queryRecords", 200, 200)  # no savings
    snap = store.snapshot()

    assert snap["tool_calls_total"] == 2
    assert snap["tool_calls_compressed"] == 1
    assert snap["chars_saved_total"] == 10056 - 6442
    assert snap["by_tool"]["nocodb · queryRecords"]["calls"] == 2
    assert snap["by_tool"]["nocodb · queryRecords"]["server"] == "nocodb"


def test_sqlite_store_aggregates_servers_sharing_one_file(tmp_path: Path) -> None:
    # The whole point: two gateway processes (two stores) → one shared file →
    # a reader sees the union, tagged per server. This is what the dashboard uses.
    db = str(tmp_path / "shared.db")
    SQLiteStatsStore(db, server="nocodb").record("queryRecords", 1000, 600)
    SQLiteStatsStore(db, server="outline").record("search", 500, 300)

    snap = SQLiteStatsStore(db).snapshot()  # third reader, no server label
    assert snap["tool_calls_total"] == 2
    assert snap["chars_saved_total"] == 400 + 200
    assert set(snap["by_tool"]) == {"nocodb · queryRecords", "outline · search"}


def test_sqlite_store_persists_across_reopen(tmp_path: Path) -> None:
    db = str(tmp_path / "persist.db")
    SQLiteStatsStore(db, server="s").record("t", 1000, 100)
    SQLiteStatsStore(db, server="s").record("t", 1000, 100)
    assert SQLiteStatsStore(db).snapshot()["by_tool"]["s · t"]["calls"] == 2


def test_sqlite_store_empty_is_neutral(tmp_path: Path) -> None:
    snap = SQLiteStatsStore(str(tmp_path / "gw.db")).snapshot()
    assert snap["tool_calls_total"] == 0
    assert snap["compression_ratio_mean"] == 1.0
    assert snap["by_tool"] == {}


# ── derive_server_name ──────────────────────────────────────────────────────────


def test_derive_server_name_from_url() -> None:
    name = derive_server_name(
        "mcp-remote.cmd", ["https://nocodb.westdev.duckdns.org/mcp/x", "--header", "tok"]
    )
    assert name == "nocodb"


def test_derive_server_name_falls_back_to_command_basename() -> None:
    assert derive_server_name("npx", ["-y", "@modelcontextprotocol/server-filesystem"]) == "npx"
    assert derive_server_name("server.exe", []) == "server"


# ── Wired through a real gateway tool call ──────────────────────────────────────


@pytest.mark.asyncio
async def test_gateway_records_savings_for_tool_call() -> None:
    import mcp.types as types
    from mcp.server import Server
    from mcp.shared.memory import create_connected_server_and_client_session as connect

    downstream: Server = Server("downstream")

    @downstream.list_tools()
    async def _lt() -> list[types.Tool]:
        return [
            types.Tool(
                name="rows", description="rows", inputSchema={"type": "object", "properties": {}}
            )
        ]

    @downstream.call_tool()
    async def _ct(name: str, arguments: dict[str, object]) -> list[types.ContentBlock]:
        return [types.TextContent(type="text", text=_JSON)]

    store = CCRStore()
    stats = GatewayStats()
    async with connect(downstream) as down_session:
        gateway = build_gateway_server(down_session, store, build_router(), stats=stats)
        async with connect(gateway) as gw:
            await gw.call_tool("rows", {})

    snap = stats.snapshot()
    assert snap["tool_calls_total"] == 1
    assert snap["tool_calls_compressed"] == 1
    assert snap["by_tool"]["rows"]["calls"] == 1
    assert snap["chars_saved_total"] > 0
    # The injected expand tool must not be counted as a downstream tool result.
    assert "contextly_expand" not in snap["by_tool"]


@pytest.mark.asyncio
async def test_gateway_tool_survives_compressor_fault(monkeypatch: pytest.MonkeyPatch) -> None:
    # A fault inside compression must never break the underlying tool call: the
    # client must still receive the original output (best-effort compression).
    import mcp.types as types
    from mcp.server import Server
    from mcp.shared.memory import create_connected_server_and_client_session as connect

    import contextly.mcp_gateway as gw_mod

    def _boom(*_args: object, **_kwargs: object) -> tuple[str, str | None]:
        raise RuntimeError("compressor exploded")

    monkeypatch.setattr(gw_mod, "compress_payload", _boom)

    downstream: Server = Server("downstream")

    @downstream.list_tools()
    async def _lt() -> list[types.Tool]:
        return [
            types.Tool(
                name="rows", description="rows", inputSchema={"type": "object", "properties": {}}
            )
        ]

    @downstream.call_tool()
    async def _ct(name: str, arguments: dict[str, object]) -> list[types.ContentBlock]:
        return [types.TextContent(type="text", text=_JSON)]

    async with connect(downstream) as down_session:
        gateway = gw_mod.build_gateway_server(down_session, CCRStore(), gw_mod.build_router())
        async with connect(gateway) as gw:
            result = await gw.call_tool("rows", {})

    assert not result.isError
    assert result.content[0].text == _JSON  # type: ignore[union-attr]  # raw original survives
