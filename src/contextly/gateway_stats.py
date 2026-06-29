"""Savings trackers for the MCP gateway.

The gateway is a long-lived stdio process that sits between an MCP client and a
downstream server. It has no FastAPI app and no ``/stats`` endpoint of its own,
so these small recorders accumulate the savings of every tool call and expose a
JSON snapshot that the gateway's HTTP dashboard polls.

Two backends share one interface (:class:`StatsRecorder`):

* :class:`GatewayStats` — in-memory, per-process. Tiny and dependency-free.
* :class:`SQLiteStatsStore` — a shared SQLite file. Because Claude Desktop runs
  *one gateway process per wrapped server*, each with its own default dashboard
  port, only one process can bind the port — the rest would show nothing. Pointed
  at one shared file, every gateway records into it and the single dashboard that
  wins the port shows the **aggregate across all wrapped servers**, tagged by a
  ``server`` label so identical tool names don't collide.

``record`` is called from the asyncio event loop (one call per tool result);
``snapshot`` is called from the dashboard's HTTP thread — the lock makes that
cross-thread read safe.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import structlog

from contextly.pricing import tokens_to_dollars

logger = structlog.get_logger(__name__)

# Same chars→tokens heuristic the rest of the codebase uses (≈4 chars/token).
_CHARS_PER_TOKEN = 4


@runtime_checkable
class StatsRecorder(Protocol):
    """The minimal surface the gateway and dashboard need from a stats backend."""

    def record(self, tool: str, chars_before: int, chars_after: int, model: str = "") -> None: ...

    def snapshot(self) -> dict[str, Any]: ...


def default_stats_path() -> str:
    """Shared stats file used by every gateway instance (and read by the proxy)."""
    return str(Path.home() / ".contextly" / "gateway_stats.db")


@dataclass
class _ToolTotals:
    """Running totals for a single downstream tool."""

    calls: int = 0
    chars_before: int = 0
    chars_after: int = 0
    dollars_saved: float = 0.0


@dataclass
class GatewayStats:
    """Thread-safe accumulator of gateway compression savings.

    Mirrors the shape of the proxy's ``/stats`` payload closely enough that a
    dashboard can render either source with the same fields.
    """

    _lock: threading.Lock = field(default_factory=threading.Lock)
    _started: float = field(default_factory=time.time)
    _calls: int = 0
    _compressed_calls: int = 0
    _chars_before: int = 0
    _chars_after: int = 0
    _dollars_saved: float = 0.0
    _by_tool: dict[str, _ToolTotals] = field(default_factory=dict)

    def record(self, tool: str, chars_before: int, chars_after: int, model: str = "") -> None:
        """Record one tool result's before/after character counts."""
        chars_saved = max(0, chars_before - chars_after)
        dollars = tokens_to_dollars(chars_saved // _CHARS_PER_TOKEN, model)
        with self._lock:
            self._calls += 1
            if chars_after < chars_before:
                self._compressed_calls += 1
            self._chars_before += chars_before
            self._chars_after += chars_after
            self._dollars_saved += dollars
            t = self._by_tool.setdefault(tool, _ToolTotals())
            t.calls += 1
            t.chars_before += chars_before
            t.chars_after += chars_after
            t.dollars_saved += dollars

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the accumulated savings."""
        with self._lock:
            chars_saved = self._chars_before - self._chars_after
            ratio = (
                round(self._chars_after / self._chars_before, 4) if self._chars_before > 0 else 1.0
            )
            by_tool = {
                name: {
                    "calls": t.calls,
                    "chars_before": t.chars_before,
                    "chars_after": t.chars_after,
                    "chars_saved": t.chars_before - t.chars_after,
                    "saved_pct": (
                        round(100 * (1 - t.chars_after / t.chars_before), 1)
                        if t.chars_before > 0
                        else 0.0
                    ),
                    "dollars_saved": round(t.dollars_saved, 6),
                }
                for name, t in sorted(self._by_tool.items())
            }
            return {
                "uptime_seconds": round(time.time() - self._started, 1),
                "tool_calls_total": self._calls,
                "tool_calls_compressed": self._compressed_calls,
                "chars_before_total": self._chars_before,
                "chars_after_total": self._chars_after,
                "chars_saved_total": chars_saved,
                "tokens_saved_estimate_total": chars_saved // _CHARS_PER_TOKEN,
                "dollars_saved_total": round(self._dollars_saved, 6),
                "compression_ratio_mean": ratio,
                "by_tool": by_tool,
            }


def _label(server: str, tool: str) -> str:
    """Display key for a (server, tool) pair — prefixed only when a server is set."""
    return f"{server} · {tool}" if server else tool


class SQLiteStatsStore:
    """Shared, multi-process savings store backed by a SQLite file (WAL mode).

    Each gateway process opens the same file and records under its own *server*
    label, so the one dashboard that wins the port reports the union of every
    wrapped server's savings. Totals are cumulative across restarts.

    All operations are best-effort: a stats write must never break a tool call,
    so SQLite errors (e.g. a momentary write lock) are logged and swallowed.

    Args:
        path: Database file path (parent directories are created).
        server: Short label identifying this gateway's downstream server.
    """

    def __init__(self, path: str, server: str = "") -> None:
        self._server = server
        self._lock = threading.Lock()
        self._started = time.time()
        db_path = Path(path).expanduser()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=2000")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS tool_stats ("
            "server TEXT NOT NULL, tool TEXT NOT NULL, calls INTEGER NOT NULL, "
            "compressed_calls INTEGER NOT NULL, chars_before INTEGER NOT NULL, "
            "chars_after INTEGER NOT NULL, dollars_saved REAL NOT NULL DEFAULT 0.0, "
            "PRIMARY KEY (server, tool))"
        )
        # Add dollars_saved column to existing databases that predate this field.
        try:
            self._conn.execute(
                "ALTER TABLE tool_stats ADD COLUMN dollars_saved REAL NOT NULL DEFAULT 0.0"
            )
        except sqlite3.OperationalError:
            pass  # Column already exists
        self._conn.commit()

    def record(self, tool: str, chars_before: int, chars_after: int, model: str = "") -> None:
        """Add one tool result to the running totals for (this server, *tool*)."""
        compressed = 1 if chars_after < chars_before else 0
        chars_saved = max(0, chars_before - chars_after)
        dollars = tokens_to_dollars(chars_saved // _CHARS_PER_TOKEN, model)
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO tool_stats"
                    "(server, tool, calls, compressed_calls,"
                    " chars_before, chars_after, dollars_saved) "
                    "VALUES (?, ?, 1, ?, ?, ?, ?) "
                    "ON CONFLICT(server, tool) DO UPDATE SET "
                    "calls = calls + 1, "
                    "compressed_calls = compressed_calls + excluded.compressed_calls, "
                    "chars_before = chars_before + excluded.chars_before, "
                    "chars_after = chars_after + excluded.chars_after, "
                    "dollars_saved = dollars_saved + excluded.dollars_saved",
                    (self._server, tool, compressed, chars_before, chars_after, dollars),
                )
                self._conn.commit()
        except sqlite3.Error:
            logger.warning("gateway_stats_write_failed", server=self._server, tool=tool)

    def snapshot(self) -> dict[str, Any]:
        """Aggregate savings across every server that shares this file."""
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT server, tool, calls, compressed_calls, chars_before, chars_after, "
                    "dollars_saved FROM tool_stats ORDER BY server, tool"
                ).fetchall()
            except sqlite3.Error:
                rows = []

        calls = compressed = before = after = 0
        dollars_total = 0.0
        by_tool: dict[str, Any] = {}
        for server, tool, c, cc, b, a, d in rows:
            calls += c
            compressed += cc
            before += b
            after += a
            dollars_total += d
            by_tool[_label(server, tool)] = {
                "server": server,
                "calls": c,
                "chars_before": b,
                "chars_after": a,
                "chars_saved": b - a,
                "saved_pct": round(100 * (1 - a / b), 1) if b > 0 else 0.0,
                "dollars_saved": round(d, 6),
            }

        chars_saved = before - after
        ratio = round(after / before, 4) if before > 0 else 1.0
        return {
            "uptime_seconds": round(time.time() - self._started, 1),
            "tool_calls_total": calls,
            "tool_calls_compressed": compressed,
            "chars_before_total": before,
            "chars_after_total": after,
            "chars_saved_total": chars_saved,
            "tokens_saved_estimate_total": chars_saved // _CHARS_PER_TOKEN,
            "dollars_saved_total": round(dollars_total, 6),
            "compression_ratio_mean": ratio,
            "by_tool": by_tool,
        }
