"""Unit tests for SharedMemoryStore (cross-agent shared memory)."""

from __future__ import annotations

from pathlib import Path

from contextly.ccr import SharedMemoryStore, content_key, default_shared_memory_path


def _store(tmp_path: Path, agent_id: str, max_entries: int = 10_000) -> SharedMemoryStore:
    return SharedMemoryStore(str(tmp_path / "shared.db"), max_entries, agent_id=agent_id)


# ── default path ──────────────────────────────────────────────────────────────


def test_default_path_under_contextly_home() -> None:
    path = default_shared_memory_path()
    assert path.endswith("shared-memory.db")
    assert ".contextly" in path


# ── cross-agent sharing ───────────────────────────────────────────────────────


def test_content_stored_by_one_agent_retrieved_by_another(tmp_path: Path) -> None:
    a = _store(tmp_path, "agent_a")
    b = _store(tmp_path, "agent_b")
    key = a.store("shared original content")
    # agent_b, a separate process/instance on the same DB file, reads it back.
    assert b.retrieve(key) == "shared original content"


def test_cross_agent_retrieval_counted(tmp_path: Path) -> None:
    a = _store(tmp_path, "agent_a")
    b = _store(tmp_path, "agent_b")
    key = a.store("some content")
    b.retrieve(key)
    assert b.stats()["cross_agent_retrievals"] == 1


def test_same_agent_retrieval_not_cross(tmp_path: Path) -> None:
    a = _store(tmp_path, "agent_a")
    key = a.store("content x")
    a.retrieve(key)
    assert a.stats()["cross_agent_retrievals"] == 0


# ── dedup ─────────────────────────────────────────────────────────────────────


def test_dedup_hit_on_repeat_store(tmp_path: Path) -> None:
    a = _store(tmp_path, "agent_a")
    b = _store(tmp_path, "agent_b")
    k1 = a.store("dup content")
    k2 = b.store("dup content")  # second agent stores identical content
    assert k1 == k2
    assert b.stats()["dedup_hits"] == 1


def test_key_is_content_hash(tmp_path: Path) -> None:
    a = _store(tmp_path, "agent_a")
    assert a.store("hash me") == content_key("hash me")


# ── lookup (discovery without storing) ────────────────────────────────────────


def test_lookup_miss_returns_none(tmp_path: Path) -> None:
    a = _store(tmp_path, "agent_a")
    assert a.lookup("never stored") is None


def test_lookup_hit_returns_key_and_agent(tmp_path: Path) -> None:
    a = _store(tmp_path, "agent_a")
    b = _store(tmp_path, "agent_b")
    a.store("findable content")
    found = b.lookup("findable content")
    assert found is not None
    key, stored_by = found
    assert key == content_key("findable content")
    assert stored_by == "agent_a"


def test_lookup_does_not_store(tmp_path: Path) -> None:
    a = _store(tmp_path, "agent_a")
    a.lookup("not stored by lookup")
    assert len(a) == 0


# ── eviction ──────────────────────────────────────────────────────────────────


def test_eviction_past_max_entries(tmp_path: Path) -> None:
    a = _store(tmp_path, "agent_a", max_entries=3)
    for i in range(5):
        a.store(f"content number {i}")
    assert len(a) == 3


def test_eviction_keeps_recently_accessed(tmp_path: Path) -> None:
    a = _store(tmp_path, "agent_a", max_entries=2)
    k0 = a.store("zero")
    a.store("one")
    # Touch k0 so it is most-recently-accessed, then overflow.
    a.retrieve(k0)
    a.store("two")  # forces eviction of the LRU entry ("one"), not k0
    assert a.retrieve(k0) == "zero"


# ── stats shape ───────────────────────────────────────────────────────────────


def test_stats_has_cross_agent_fields(tmp_path: Path) -> None:
    a = _store(tmp_path, "agent_a")
    a.store("x")
    st = a.stats()
    assert st["backend"] == "shared-memory"
    assert {"dedup_hits", "cross_agent_retrievals", "distinct_agents"}.issubset(st.keys())


def test_distinct_agents_counted(tmp_path: Path) -> None:
    a = _store(tmp_path, "agent_a")
    b = _store(tmp_path, "agent_b")
    a.store("from a")
    b.store("from b")
    assert a.stats()["distinct_agents"] == 2
