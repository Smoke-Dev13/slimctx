"""Unit tests for AdaptiveController."""

from __future__ import annotations

from contextly.controller import AdaptiveController, session_key_from


def _ctrl(**kw) -> AdaptiveController:  # type: ignore[no-untyped-def]
    return AdaptiveController(**kw)


# ── session key ─────────────────────────────────────────────────────────────────


def test_session_key_from_header() -> None:
    assert session_key_from("sess-123", "ignored") == "sess-123"


def test_session_key_from_system_prompt_stable() -> None:
    k1 = session_key_from(None, "You are helpful.")
    k2 = session_key_from(None, "You are helpful.")
    k3 = session_key_from(None, "Different.")
    assert k1 == k2
    assert k1 != k3
    assert k1.startswith("sys:")


def test_session_key_header_takes_precedence() -> None:
    assert session_key_from("  hdr ", "sys") == "hdr"


# ── choose / default ────────────────────────────────────────────────────────────


def test_default_level_is_default() -> None:
    c = _ctrl()
    assert c.choose("s1") == "default"


# ── compression-paradox guard ───────────────────────────────────────────────────


def test_verbosity_spike_steps_down() -> None:
    c = _ctrl(paradox_threshold=0.25)
    # establish a baseline at a weaker aggression (safe)
    c.observe("s", aggression="safe", completion_tokens=100)
    # now a stronger aggression produces a big jump in output → paradox
    spike = c.observe("s", aggression="default", completion_tokens=200)
    assert spike is True
    # level stepped down from default → safe
    assert c.choose("s") == "safe"


def test_no_spike_when_output_stable() -> None:
    c = _ctrl(paradox_threshold=0.25)
    c.observe("s", aggression="safe", completion_tokens=100)
    spike = c.observe("s", aggression="default", completion_tokens=105)
    assert spike is False


# ── quality floor ───────────────────────────────────────────────────────────────


def test_low_quality_steps_down() -> None:
    c = _ctrl(min_quality=0.6)
    c.observe("s", aggression="default", completion_tokens=50, quality_score=0.9)
    # bad measured quality pulls the level down
    c.observe("s", aggression="default", completion_tokens=50, quality_score=0.4)
    assert c.choose("s") == "safe"


def test_healthy_quality_high_fill_steps_up() -> None:
    c = _ctrl(min_quality=0.6)
    c.observe(
        "s",
        aggression="default",
        completion_tokens=50,
        quality_score=0.95,
        context_fill=0.8,
    )
    assert c.choose("s") == "aggressive"


def test_step_up_clamped_at_aggressive() -> None:
    c = _ctrl()
    for _ in range(5):
        c.observe(
            "s",
            aggression=c.choose("s"),
            completion_tokens=50,
            quality_score=0.95,
            context_fill=0.9,
        )
    assert c.choose("s") == "aggressive"


def test_step_down_clamped_at_safe() -> None:
    c = _ctrl(min_quality=0.6)
    for _ in range(5):
        c.observe("s", aggression=c.choose("s"), completion_tokens=50, quality_score=0.1)
    assert c.choose("s") == "safe"


# ── LRU eviction ─────────────────────────────────────────────────────────────────


def test_lru_eviction_bounds_sessions() -> None:
    c = _ctrl(max_sessions=3)
    for i in range(5):
        c.choose(f"s{i}")
    assert c.stats()["adaptive_sessions_total"] == 3


# ── stats ────────────────────────────────────────────────────────────────────────


def test_stats_counts_steps_and_spikes() -> None:
    c = _ctrl(paradox_threshold=0.25, min_quality=0.6)
    c.observe("s", aggression="safe", completion_tokens=100)
    c.observe("s", aggression="default", completion_tokens=300)  # spike → stepdown
    st = c.stats()
    assert st["verbosity_spikes_total"] == 1
    assert st["adaptive_stepdowns_total"] >= 1


def test_stats_initial() -> None:
    st = _ctrl().stats()
    assert st == {
        "adaptive_sessions_total": 0,
        "adaptive_stepups_total": 0,
        "adaptive_stepdowns_total": 0,
        "verbosity_spikes_total": 0,
    }
