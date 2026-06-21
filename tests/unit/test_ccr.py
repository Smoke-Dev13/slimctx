"""Unit tests for CCRStore."""

from __future__ import annotations

import threading

import pytest

from contextly.ccr import CCRStore

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def store() -> CCRStore:
    return CCRStore(max_entries=5)


# ── Key format ─────────────────────────────────────────────────────────────────


def test_store_returns_16_char_hex(store: CCRStore) -> None:
    key = store.store("hello world")
    assert len(key) == 16
    assert all(c in "0123456789abcdef" for c in key)


def test_store_deterministic(store: CCRStore) -> None:
    key1 = store.store("same content")
    key2 = store.store("same content")
    assert key1 == key2


def test_different_content_different_keys(store: CCRStore) -> None:
    k1 = store.store("content A")
    k2 = store.store("content B")
    assert k1 != k2


# ── Retrieve ───────────────────────────────────────────────────────────────────


def test_retrieve_returns_original(store: CCRStore) -> None:
    content = "The quick brown fox jumps over the lazy dog."
    key = store.store(content)
    assert store.retrieve(key) == content


def test_retrieve_missing_returns_none(store: CCRStore) -> None:
    assert store.retrieve("deadbeef12345678") is None


def test_retrieve_missing_unknown_key(store: CCRStore) -> None:
    store.store("something")
    assert store.retrieve("0000000000000000") is None


def test_retrieve_unicode_content(store: CCRStore) -> None:
    content = "Hello 世界 🌍 Привет"
    key = store.store(content)
    assert store.retrieve(key) == content


def test_retrieve_empty_string(store: CCRStore) -> None:
    key = store.store("")
    assert store.retrieve(key) == ""


def test_retrieve_large_content(store: CCRStore) -> None:
    content = "x" * 100_000
    key = store.store(content)
    assert store.retrieve(key) == content


# ── Deduplication ──────────────────────────────────────────────────────────────


def test_store_deduplication(store: CCRStore) -> None:
    store.store("repeated content")
    store.store("repeated content")
    assert len(store) == 1


def test_store_deduplication_does_not_increment_n_stored(store: CCRStore) -> None:
    store.store("repeated content")
    store.store("repeated content")
    stats = store.stats()
    assert stats["total_stored"] == 1


# ── LRU eviction ──────────────────────────────────────────────────────────────


def test_lru_evicts_oldest_when_full() -> None:
    s = CCRStore(max_entries=3)
    k1 = s.store("first")
    s.store("second")
    s.store("third")
    # Store a fourth — oldest (first) should be evicted
    s.store("fourth")
    assert s.retrieve(k1) is None
    assert len(s) == 3


def test_lru_promotes_on_retrieve() -> None:
    s = CCRStore(max_entries=3)
    k1 = s.store("first")
    s.store("second")
    s.store("third")
    # Access k1 to make it most-recently-used
    s.retrieve(k1)
    # Now adding a new entry should evict "second" (now oldest), not k1
    s.store("fourth")
    assert s.retrieve(k1) == "first"  # k1 survived


def test_lru_promotes_on_store_duplicate() -> None:
    s = CCRStore(max_entries=3)
    k1 = s.store("first")
    s.store("second")
    s.store("third")
    # Re-store k1 to refresh it
    s.store("first")
    # Add new — "second" should be evicted
    s.store("fourth")
    assert s.retrieve(k1) == "first"


def test_lru_capacity_respected() -> None:
    s = CCRStore(max_entries=2)
    for i in range(10):
        s.store(f"item {i}")
    assert len(s) == 2


# ── Stats ─────────────────────────────────────────────────────────────────────


def test_stats_initial(store: CCRStore) -> None:
    s = store.stats()
    assert s["current_entries"] == 0
    assert s["max_entries"] == 5
    assert s["total_stored"] == 0
    assert s["total_retrieved"] == 0
    assert s["hits"] == 0
    assert s["misses"] == 0
    assert s["hit_rate"] == 0.0


def test_stats_after_store(store: CCRStore) -> None:
    store.store("a")
    store.store("b")
    s = store.stats()
    assert s["current_entries"] == 2
    assert s["total_stored"] == 2


def test_stats_hit_tracking(store: CCRStore) -> None:
    key = store.store("content")
    store.retrieve(key)
    store.retrieve(key)
    store.retrieve("nonexistent")
    s = store.stats()
    assert s["total_retrieved"] == 3
    assert s["hits"] == 2
    assert s["misses"] == 1
    assert s["hit_rate"] == pytest.approx(2 / 3, abs=1e-4)


def test_stats_zero_hit_rate_when_no_retrievals(store: CCRStore) -> None:
    store.store("x")
    assert store.stats()["hit_rate"] == 0.0


def test_stats_returns_snapshot(store: CCRStore) -> None:
    s1 = store.stats()
    store.store("new content")
    s2 = store.stats()
    assert s1["current_entries"] != s2["current_entries"]


# ── __len__ ───────────────────────────────────────────────────────────────────


def test_len_empty(store: CCRStore) -> None:
    assert len(store) == 0


def test_len_after_stores(store: CCRStore) -> None:
    store.store("a")
    store.store("b")
    assert len(store) == 2


# ── Thread safety ─────────────────────────────────────────────────────────────


def test_concurrent_stores_no_corruption() -> None:
    s = CCRStore(max_entries=1000)
    keys: list[str] = []
    lock = threading.Lock()

    def worker(n: int) -> None:
        for i in range(20):
            k = s.store(f"thread-{n}-item-{i}")
            with lock:
                keys.append(k)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(keys) == 200
    # All keys are valid 16-char hex strings
    for k in keys:
        assert len(k) == 16


def test_concurrent_store_retrieve_no_exception() -> None:
    s = CCRStore(max_entries=50)
    errors: list[Exception] = []

    def writer(n: int) -> None:
        try:
            for i in range(30):
                s.store(f"writer-{n}-{i}")
        except Exception as e:
            errors.append(e)

    def reader() -> None:
        try:
            for _ in range(30):
                s.retrieve(f"{'0' * 16}")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(5)]
    threads += [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
