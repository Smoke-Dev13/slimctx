"""Prompt-cache optimization for upstream LLM requests.

Two complementary mechanisms:

  1. Anthropic ``cache_control`` auto-injection — marks the longest *stable*
     prefix (system prompt + older messages) with ``{"type": "ephemeral"}`` so
     apps get Anthropic prompt caching (cache read = 0.1x input price) without
     any client changes. The fresh tail (latest turns) is left unmarked so it
     stays outside the cached prefix.

  2. OpenAI cached-token accounting — OpenAI caches eligible prefixes
     automatically (cache read = 0.5x input price); we parse
     ``usage.prompt_tokens_details.cached_tokens`` from the response and report
     the savings.

The proxy's job for both providers is to keep the prefix byte-stable across
turns (cache-aware reorder handles that) and, for Anthropic, to place the
breakpoints. Stdlib only, thread-safe.
"""

from __future__ import annotations

import threading
from typing import Any

# Cache discounts: fraction of the normal input price paid on a cache *read*.
# Anthropic charges 10% on cached reads; OpenAI charges 50%.
_ANTHROPIC_CACHE_READ_FRACTION = 0.1
_OPENAI_CACHE_READ_FRACTION = 0.5

# Anthropic allows at most 4 cache breakpoints per request.
_MAX_ANTHROPIC_BREAKPOINTS = 4


def _to_block_list(content: Any) -> list[dict[str, Any]]:
    """Normalize message/system content to a list of content blocks.

    A bare string becomes ``[{"type": "text", "text": <string>}]``; an existing
    block list is returned shallow-copied (so we never mutate the caller's data).
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return [dict(block) if isinstance(block, dict) else block for block in content]
    return []


def _mark_last_block(blocks: list[dict[str, Any]]) -> bool:
    """Attach an ephemeral cache_control marker to the last dict block.

    Returns True if a marker was placed, False if there was no eligible block.
    """
    for block in reversed(blocks):
        if isinstance(block, dict):
            block["cache_control"] = {"type": "ephemeral"}
            return True
    return False


def _estimate_chars(system: Any, messages: list[dict[str, Any]], upto: int) -> int:
    """Rough char count of the system prompt plus messages[0:upto]."""
    total = 0
    if isinstance(system, str):
        total += len(system)
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict):
                total += len(str(block.get("text", "")))
    for msg in messages[:upto]:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total += len(str(block.get("text", "")))
    return total


class CacheOptimizer:
    """Inject Anthropic cache breakpoints and account for cache savings.

    Thread-safe: counters guarded by a lock. Instantiate once and share.
    """

    def __init__(self) -> None:
        self._breakpoints_injected = 0
        self._cache_hit_tokens = 0
        self._cache_savings_dollars = 0.0
        self._lock = threading.Lock()

    def inject_anthropic_breakpoints(
        self,
        messages: list[dict[str, Any]],
        system: Any,
        *,
        min_prefix_chars: int = 4096,
        recent_window: int = 2,
        max_breakpoints: int = 2,
    ) -> tuple[list[dict[str, Any]], Any, int]:
        """Mark the stable prefix with ephemeral cache_control breakpoints.

        Breakpoint #1 goes on the system prompt (always stable). Breakpoint #2
        goes on the last message *before* the fresh tail (``recent_window``
        messages from the end). No more than ``max_breakpoints`` (capped at the
        Anthropic limit of 4) are placed.

        Injection is skipped entirely when the estimated stable-prefix size is
        below ``min_prefix_chars`` (~1024 tokens), since caching only activates
        above that threshold and a marked-but-too-small prefix wastes a
        breakpoint and incurs the cache-write surcharge.

        Args:
            messages: Chat/messages list (not mutated; a new list is returned).
            system: Anthropic ``system`` field — string, block list, or None.
            min_prefix_chars: Minimum estimated prefix size to bother caching.
            recent_window: Number of trailing messages kept out of the prefix.
            max_breakpoints: Desired breakpoint count (capped at 4).

        Returns:
            (new_messages, new_system, breakpoints_placed).
        """
        cap = min(max_breakpoints, _MAX_ANTHROPIC_BREAKPOINTS)
        if cap <= 0:
            return messages, system, 0

        # Boundary index: messages[:boundary] are the cacheable stable prefix.
        boundary = len(messages) - recent_window
        prefix_chars = _estimate_chars(system, messages, max(boundary, 0))
        if prefix_chars < min_prefix_chars:
            return messages, system, 0

        placed = 0
        new_system = system

        # Breakpoint #1 — system prompt.
        if system:
            sys_blocks = _to_block_list(system)
            if _mark_last_block(sys_blocks):
                new_system = sys_blocks
                placed += 1

        # Breakpoint #2 — last stable message before the fresh tail.
        new_messages = messages
        if placed < cap and 0 < boundary <= len(messages):
            new_messages = [dict(m) for m in messages]
            target = new_messages[boundary - 1]
            blocks = _to_block_list(target.get("content", ""))
            if blocks and _mark_last_block(blocks):
                target["content"] = blocks
                placed += 1

        if placed:
            with self._lock:
                self._breakpoints_injected += placed

        return new_messages, new_system, placed

    def record_openai_usage(self, usage: dict[str, Any], price_per_1k: float) -> int:
        """Account for OpenAI automatic prefix caching from a response usage block.

        Args:
            usage: The ``usage`` object from an OpenAI chat completion response.
            price_per_1k: Normal input price in USD per 1,000 tokens for the model.

        Returns:
            The number of cached prompt tokens reported (0 if none).
        """
        details = usage.get("prompt_tokens_details") or {}
        cached = int(details.get("cached_tokens", 0) or 0)
        if cached <= 0:
            return 0
        # Savings = cached tokens * (1 - read_fraction) * price.
        saved = (cached / 1000.0) * price_per_1k * (1.0 - _OPENAI_CACHE_READ_FRACTION)
        with self._lock:
            self._cache_hit_tokens += cached
            self._cache_savings_dollars += saved
        return cached

    def record_anthropic_usage(self, usage: dict[str, Any], price_per_1k: float) -> int:
        """Account for Anthropic cache reads from a response usage block.

        Args:
            usage: The ``usage`` object from an Anthropic messages response.
            price_per_1k: Normal input price in USD per 1,000 tokens for the model.

        Returns:
            The number of cache-read tokens reported (0 if none).
        """
        cached = int(usage.get("cache_read_input_tokens", 0) or 0)
        if cached <= 0:
            return 0
        saved = (cached / 1000.0) * price_per_1k * (1.0 - _ANTHROPIC_CACHE_READ_FRACTION)
        with self._lock:
            self._cache_hit_tokens += cached
            self._cache_savings_dollars += saved
        return cached

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "cache_breakpoints_injected_total": self._breakpoints_injected,
                "cache_hit_tokens_total": self._cache_hit_tokens,
                "cache_savings_dollars_total": round(self._cache_savings_dollars, 6),
            }
