"""In-process savings tracker for the MCP gateway.

The gateway is a long-lived stdio process that sits between an MCP client and a
downstream server. It has no FastAPI app and no ``/stats`` endpoint of its own,
so this small thread-safe accumulator records the savings of every tool call and
exposes a JSON snapshot that the gateway's optional HTTP dashboard can poll.

It is deliberately tiny and dependency-free: a lock, a few running totals, and a
per-tool breakdown. ``record`` is called from the asyncio event loop (one call
per tool result); ``snapshot`` is called from the dashboard's HTTP thread — the
lock makes that cross-thread read safe.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

# Same chars→tokens heuristic the rest of the codebase uses (≈4 chars/token).
_CHARS_PER_TOKEN = 4


@dataclass
class _ToolTotals:
    """Running totals for a single downstream tool."""

    calls: int = 0
    chars_before: int = 0
    chars_after: int = 0


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
    _by_tool: dict[str, _ToolTotals] = field(default_factory=dict)

    def record(self, tool: str, chars_before: int, chars_after: int) -> None:
        """Record one tool result's before/after character counts."""
        with self._lock:
            self._calls += 1
            if chars_after < chars_before:
                self._compressed_calls += 1
            self._chars_before += chars_before
            self._chars_after += chars_after
            t = self._by_tool.setdefault(tool, _ToolTotals())
            t.calls += 1
            t.chars_before += chars_before
            t.chars_after += chars_after

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
                "compression_ratio_mean": ratio,
                "by_tool": by_tool,
            }
