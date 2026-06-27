"""Unit tests for CacheOptimizer."""

from __future__ import annotations

from contextly.cache_opt import CacheOptimizer


def _opt() -> CacheOptimizer:
    return CacheOptimizer()


def _big(text: str = "x", n: int = 5000) -> str:
    return text * n


# ── Anthropic breakpoint injection ─────────────────────────────────────────────


def test_inject_marks_string_system() -> None:
    opt = _opt()
    messages = [
        {"role": "user", "content": _big()},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "now?"},
        {"role": "assistant", "content": "sure"},
    ]
    _new_msgs, new_system, n = opt.inject_anthropic_breakpoints(
        messages, "You are helpful.", min_prefix_chars=1000, recent_window=2
    )
    assert n >= 1
    # system converted from string to block list with a marker
    assert isinstance(new_system, list)
    assert new_system[-1]["cache_control"] == {"type": "ephemeral"}


def test_inject_marks_last_stable_message() -> None:
    opt = _opt()
    messages = [
        {"role": "user", "content": _big()},
        {"role": "assistant", "content": _big()},
        {"role": "user", "content": "latest"},
        {"role": "assistant", "content": "reply"},
    ]
    new_msgs, _system, n = opt.inject_anthropic_breakpoints(
        messages, "sys", min_prefix_chars=1000, recent_window=2
    )
    # boundary = 4 - 2 = 2, so message index 1 (assistant) gets the marker
    marked = new_msgs[1]["content"]
    assert isinstance(marked, list)
    assert marked[-1]["cache_control"] == {"type": "ephemeral"}
    assert n == 2


def test_inject_skips_below_min_prefix() -> None:
    opt = _opt()
    messages = [
        {"role": "user", "content": "tiny"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "more"},
    ]
    new_msgs, new_system, n = opt.inject_anthropic_breakpoints(
        messages, "sys", min_prefix_chars=4096, recent_window=2
    )
    assert n == 0
    assert new_msgs is messages
    assert new_system == "sys"


def test_inject_respects_max_breakpoints() -> None:
    opt = _opt()
    messages = [
        {"role": "user", "content": _big()},
        {"role": "assistant", "content": _big()},
        {"role": "user", "content": "latest"},
        {"role": "assistant", "content": "reply"},
    ]
    _msgs, _system, n = opt.inject_anthropic_breakpoints(
        messages, "sys", min_prefix_chars=1000, recent_window=2, max_breakpoints=1
    )
    assert n == 1


def test_inject_does_not_mutate_original() -> None:
    opt = _opt()
    messages = [
        {"role": "user", "content": _big()},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "latest"},
        {"role": "assistant", "content": "reply"},
    ]
    original_content = messages[1]["content"]
    opt.inject_anthropic_breakpoints(messages, "sys", min_prefix_chars=1000, recent_window=2)
    # original list/objects untouched
    assert messages[1]["content"] == original_content


def test_inject_existing_block_list_system() -> None:
    opt = _opt()
    system = [{"type": "text", "text": _big()}]
    messages = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d"},
    ]
    _msgs, new_system, n = opt.inject_anthropic_breakpoints(
        messages, system, min_prefix_chars=1000, recent_window=2
    )
    assert n >= 1
    assert new_system[-1]["cache_control"] == {"type": "ephemeral"}
    # original system block not mutated
    assert "cache_control" not in system[0]


def test_inject_no_system() -> None:
    opt = _opt()
    messages = [
        {"role": "user", "content": _big()},
        {"role": "assistant", "content": _big()},
        {"role": "user", "content": "latest"},
        {"role": "assistant", "content": "reply"},
    ]
    _msgs, new_system, n = opt.inject_anthropic_breakpoints(
        messages, None, min_prefix_chars=1000, recent_window=2
    )
    assert new_system is None
    # still marks the stable message
    assert n == 1


# ── usage accounting ───────────────────────────────────────────────────────────


def test_openai_cached_tokens_parsed() -> None:
    opt = _opt()
    usage = {"prompt_tokens": 2000, "prompt_tokens_details": {"cached_tokens": 1000}}
    cached = opt.record_openai_usage(usage, price_per_1k=0.0025)
    assert cached == 1000
    st = opt.stats()
    assert st["cache_hit_tokens_total"] == 1000
    # savings = 1000/1000 * 0.0025 * (1 - 0.5) = 0.00125
    assert abs(st["cache_savings_dollars_total"] - 0.00125) < 1e-9


def test_openai_no_cached_tokens() -> None:
    opt = _opt()
    assert opt.record_openai_usage({"prompt_tokens": 100}, price_per_1k=0.0025) == 0


def test_anthropic_cache_read_parsed() -> None:
    opt = _opt()
    usage = {"input_tokens": 50, "cache_read_input_tokens": 2000}
    cached = opt.record_anthropic_usage(usage, price_per_1k=0.015)
    assert cached == 2000
    st = opt.stats()
    # savings = 2000/1000 * 0.015 * (1 - 0.1) = 0.027
    assert abs(st["cache_savings_dollars_total"] - 0.027) < 1e-9


def test_stats_initial() -> None:
    st = _opt().stats()
    assert st["cache_breakpoints_injected_total"] == 0
    assert st["cache_hit_tokens_total"] == 0
    assert st["cache_savings_dollars_total"] == 0.0


def test_breakpoint_counter_accumulates() -> None:
    opt = _opt()
    messages = [
        {"role": "user", "content": _big()},
        {"role": "assistant", "content": _big()},
        {"role": "user", "content": "latest"},
        {"role": "assistant", "content": "reply"},
    ]
    opt.inject_anthropic_breakpoints(messages, "sys", min_prefix_chars=1000, recent_window=2)
    opt.inject_anthropic_breakpoints(messages, "sys", min_prefix_chars=1000, recent_window=2)
    assert opt.stats()["cache_breakpoints_injected_total"] == 4
