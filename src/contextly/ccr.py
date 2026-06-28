"""CCR (Context Compression Reversibility) store.

Stores the original content keyed by a short SHA-256 digest so it can be
retrieved after compression.  A single CCRStore instance lives in app.state
for the proxy server and as a module-level singleton for the MCP server.

Design:
  - Thread-safe: uses threading.Lock (safe to call from asyncio without
    blocking the event loop because no operation holds the lock across an
    await point).
  - LRU eviction: backed by collections.OrderedDict; oldest entry removed
    when max_entries is reached.
  - Key: first 16 hex chars of SHA-256(content) — 64 bits of entropy,
    collision probability negligible at the configured store sizes.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any


def content_key(content: str) -> str:
    """Return the 16-hex-char retrieval key for *content* (SHA-256 prefix)."""
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:16]


class CCRStore:
    """Thread-safe LRU key-value store for reversible compression.

    Args:
        max_entries: Maximum number of originals to hold in memory before
            the oldest entries are evicted.
    """

    def __init__(self, max_entries: int = 10_000) -> None:
        self._store: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_entries
        self._n_stored: int = 0
        self._n_retrieved: int = 0
        self._n_hits: int = 0
        self._n_misses: int = 0

    # ── Core operations ───────────────────────────────────────────────────────

    def store(self, content: str) -> str:
        """Persist *content* and return its retrieval key.

        If the same content is stored twice the key is identical and no
        duplicate entry is created (the entry is refreshed to most-recent).

        Args:
            content: Original text to preserve.

        Returns:
            16-character hex string that can be passed to retrieve().
        """
        key = content_key(content)
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            else:
                if len(self._store) >= self._max:
                    self._store.popitem(last=False)
                self._store[key] = content
                self._n_stored += 1
        return key

    def retrieve(self, key: str) -> str | None:
        """Return the content stored under *key*, or None if absent / evicted.

        Accessing an entry promotes it to most-recently-used.

        Args:
            key: The retrieval key returned by store().

        Returns:
            Original content string, or None.
        """
        with self._lock:
            self._n_retrieved += 1
            value = self._store.get(key)
            if value is not None:
                self._store.move_to_end(key)
                self._n_hits += 1
                return value
            self._n_misses += 1
            return None

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return a snapshot of store metrics.

        Returns:
            Dict with current_entries, max_entries, total_stored,
            total_retrieved, hits, misses, and hit_rate.
        """
        with self._lock:
            n_entries = len(self._store)
            hit_rate = round(self._n_hits / self._n_retrieved, 4) if self._n_retrieved > 0 else 0.0
            return {
                "current_entries": n_entries,
                "max_entries": self._max,
                "total_stored": self._n_stored,
                "total_retrieved": self._n_retrieved,
                "hits": self._n_hits,
                "misses": self._n_misses,
                "hit_rate": hit_rate,
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


class SQLiteCCRStore(CCRStore):
    """Disk-backed CCR store — survives restarts and is shared across processes.

    Drop-in for :class:`CCRStore` backed by a SQLite file (WAL mode). Because all
    uvicorn workers open the same database file, an original stored by one worker
    is retrievable by any other — unlike the per-process in-memory store, which
    makes ``expand``/``retrieve`` unreliable under ``--workers > 1``.

    Args:
        path: Database file path (parent directories are created).
        max_entries: Soft cap; the oldest rows are evicted past this count.
    """

    def __init__(self, path: str, max_entries: int = 10_000) -> None:
        super().__init__(max_entries)
        Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(Path(path).expanduser()), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS ccr "
            "(key TEXT PRIMARY KEY, content TEXT NOT NULL, created REAL NOT NULL)"
        )
        self._conn.commit()

    def store(self, content: str) -> str:
        key = content_key(content)
        with self._lock:
            exists = self._conn.execute("SELECT 1 FROM ccr WHERE key=?", (key,)).fetchone()
            if exists is None:
                self._conn.execute(
                    "INSERT INTO ccr(key, content, created) VALUES (?, ?, ?)",
                    (key, content, time.time()),
                )
                self._n_stored += 1
                count = self._conn.execute("SELECT COUNT(*) FROM ccr").fetchone()[0]
                if count > self._max:
                    self._conn.execute(
                        "DELETE FROM ccr WHERE key IN "
                        "(SELECT key FROM ccr ORDER BY created ASC LIMIT ?)",
                        (count - self._max,),
                    )
                self._conn.commit()
        return key

    def retrieve(self, key: str) -> str | None:
        with self._lock:
            self._n_retrieved += 1
            row = self._conn.execute("SELECT content FROM ccr WHERE key=?", (key,)).fetchone()
            if row is not None:
                self._n_hits += 1
                return str(row[0])
            self._n_misses += 1
            return None

    def stats(self) -> dict[str, Any]:
        with self._lock:
            n_entries = int(self._conn.execute("SELECT COUNT(*) FROM ccr").fetchone()[0])
            hit_rate = round(self._n_hits / self._n_retrieved, 4) if self._n_retrieved > 0 else 0.0
            return {
                "current_entries": n_entries,
                "max_entries": self._max,
                "total_stored": self._n_stored,
                "total_retrieved": self._n_retrieved,
                "hits": self._n_hits,
                "misses": self._n_misses,
                "hit_rate": hit_rate,
                "backend": "sqlite",
            }

    def __len__(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM ccr").fetchone()[0])


def default_shared_memory_path() -> str:
    """Well-known shared-memory DB used by every agent process on this host."""
    return str(Path.home() / ".contextly" / "shared-memory.db")


class SharedMemoryStore(SQLiteCCRStore):
    """Cross-agent shared CCR store.

    A WAL-mode SQLite store pointed at one well-known file
    (:func:`default_shared_memory_path`) so independent agent processes — Claude
    Code, Cursor, Codex, the proxy, every wrapped gateway — converge on a single
    deduplicated memory pool. Content compressed/stored by one agent is reused by
    any other: the content hash is the key, so an identical original from a second
    agent is a dedup hit, not a new row.

    Adds, over :class:`SQLiteCCRStore`:
      - ``agent_id`` attribution per entry and ``last_accessed`` for access-based LRU,
      - :meth:`lookup` for dedup discovery (does any agent already have this?),
      - cross-agent metrics: ``dedup_hits`` (repeat stores of existing content) and
        ``cross_agent_retrievals`` (a retrieve of content stored by a different agent).
    """

    def __init__(self, path: str, max_entries: int = 10_000, *, agent_id: str = "default") -> None:
        super().__init__(path, max_entries)
        self._agent_id = agent_id
        self._dedup_hits = 0
        self._cross_agent_retrievals = 0
        # Extend the base schema in place (idempotent for existing files).
        self._conn.execute("PRAGMA busy_timeout=2000")
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(ccr)")}
        if "agent_id" not in cols:
            self._conn.execute("ALTER TABLE ccr ADD COLUMN agent_id TEXT")
        if "last_accessed" not in cols:
            self._conn.execute("ALTER TABLE ccr ADD COLUMN last_accessed REAL")
        self._conn.commit()

    def store(self, content: str, *, agent_id: str | None = None) -> str:
        agent_id = agent_id if agent_id is not None else self._agent_id
        key = content_key(content)
        now = time.time()
        with self._lock:
            row = self._conn.execute("SELECT agent_id FROM ccr WHERE key=?", (key,)).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO ccr(key, content, created, agent_id, last_accessed) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (key, content, now, agent_id, now),
                )
                self._n_stored += 1
                count = self._conn.execute("SELECT COUNT(*) FROM ccr").fetchone()[0]
                if count > self._max:
                    # Evict the least-recently-accessed rows (fall back to created).
                    self._conn.execute(
                        "DELETE FROM ccr WHERE key IN "
                        "(SELECT key FROM ccr "
                        "ORDER BY COALESCE(last_accessed, created) ASC LIMIT ?)",
                        (count - self._max,),
                    )
            else:
                # Content already known (this or another agent) → dedup hit.
                self._dedup_hits += 1
                self._conn.execute("UPDATE ccr SET last_accessed=? WHERE key=?", (now, key))
            self._conn.commit()
        return key

    def lookup(self, content: str) -> tuple[str, str] | None:
        """Return ``(key, stored_by_agent_id)`` if *content* is already stored.

        Pure read — does not store or mutate counters. Lets an agent ask "has any
        agent already compressed this?" before doing the work itself.
        """
        key = content_key(content)
        with self._lock:
            row = self._conn.execute("SELECT agent_id FROM ccr WHERE key=?", (key,)).fetchone()
            if row is None:
                return None
            return key, (row[0] if row[0] is not None else "default")

    def retrieve(self, key: str, *, agent_id: str | None = None) -> str | None:
        agent_id = agent_id if agent_id is not None else self._agent_id
        now = time.time()
        with self._lock:
            self._n_retrieved += 1
            row = self._conn.execute(
                "SELECT content, agent_id FROM ccr WHERE key=?", (key,)
            ).fetchone()
            if row is None:
                self._n_misses += 1
                return None
            self._n_hits += 1
            stored_by = row[1] if row[1] is not None else "default"
            if stored_by != agent_id:
                self._cross_agent_retrievals += 1
            self._conn.execute("UPDATE ccr SET last_accessed=? WHERE key=?", (now, key))
            self._conn.commit()
            return str(row[0])

    def stats(self) -> dict[str, Any]:
        base = super().stats()
        with self._lock:
            distinct = int(
                self._conn.execute(
                    "SELECT COUNT(DISTINCT agent_id) FROM ccr WHERE agent_id IS NOT NULL"
                ).fetchone()[0]
            )
            base.update(
                {
                    "backend": "shared-memory",
                    "dedup_hits": self._dedup_hits,
                    "cross_agent_retrievals": self._cross_agent_retrievals,
                    "distinct_agents": distinct,
                }
            )
            return base
