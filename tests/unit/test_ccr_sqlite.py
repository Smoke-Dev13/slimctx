"""Unit tests for the disk-backed SQLiteCCRStore."""

from __future__ import annotations

from pathlib import Path

from contextly.ccr import CCRStore, SQLiteCCRStore, content_key


def test_store_retrieve_roundtrip(tmp_path: Path) -> None:
    store = SQLiteCCRStore(str(tmp_path / "ccr.db"))
    key = store.store("hello world")
    assert store.retrieve(key) == "hello world"


def test_key_matches_memory_store(tmp_path: Path) -> None:
    # Same content must hash to the same key in both backends.
    sqlite_key = SQLiteCCRStore(str(tmp_path / "a.db")).store("payload")
    assert sqlite_key == CCRStore().store("payload") == content_key("payload")


def test_persists_across_instances(tmp_path: Path) -> None:
    path = str(tmp_path / "ccr.db")
    key = SQLiteCCRStore(path).store("durable content")
    # A fresh instance on the same file (e.g. another worker / after restart).
    assert SQLiteCCRStore(path).retrieve(key) == "durable content"


def test_retrieve_missing_returns_none(tmp_path: Path) -> None:
    assert SQLiteCCRStore(str(tmp_path / "ccr.db")).retrieve("deadbeefdeadbeef") is None


def test_duplicate_store_is_idempotent(tmp_path: Path) -> None:
    store = SQLiteCCRStore(str(tmp_path / "ccr.db"))
    k1 = store.store("same")
    k2 = store.store("same")
    assert k1 == k2
    assert len(store) == 1


def test_eviction_past_max(tmp_path: Path) -> None:
    store = SQLiteCCRStore(str(tmp_path / "ccr.db"), max_entries=5)
    keys = [store.store(f"item-{i}") for i in range(8)]
    assert len(store) == 5
    # Oldest evicted, newest retained.
    assert store.retrieve(keys[0]) is None
    assert store.retrieve(keys[-1]) == "item-7"


def test_stats_reports_sqlite_backend(tmp_path: Path) -> None:
    store = SQLiteCCRStore(str(tmp_path / "ccr.db"))
    store.store("x")
    s = store.stats()
    assert s["backend"] == "sqlite"
    assert s["current_entries"] == 1


def test_creates_parent_directory(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "nested" / "ccr.db"
    SQLiteCCRStore(str(nested)).store("x")
    assert nested.exists()
