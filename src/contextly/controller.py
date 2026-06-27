"""Adaptive compression controller — closed-loop per-session aggression tuning.

Compression can backfire: over-compress and the model emits longer or worse
output, costing *more* tokens (the "compression paradox"). This controller
watches the real outcomes Contextly already measures — response length,
A/B quality scores (ROUGE-1 vs an uncompressed shadow), and cache hit rate —
and dials the compressor chain between ``safe`` → ``default`` → ``aggressive``
per session.

Unlike heuristic controllers that tune on output length alone, this one steps
down when *measured* quality drops below a floor and steps up only while quality
stays healthy, optimizing quality-per-dollar directly.

Stdlib only, thread-safe.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Literal

Level = Literal["safe", "default", "aggressive"]

# Ordered weakest → strongest. Index arithmetic drives step up/down.
_LEVELS: tuple[Level, ...] = ("safe", "default", "aggressive")


@dataclass
class _Sample:
    aggression: Level
    completion_tokens: int
    quality_score: float | None
    cached_tokens: int


@dataclass
class _SessionState:
    level: Level = "default"
    samples: deque[_Sample] = field(default_factory=lambda: deque(maxlen=8))
    last_spike: bool = False


def _step(level: Level, delta: int) -> Level:
    """Return the level shifted by *delta* steps, clamped to the valid range."""
    idx = _LEVELS.index(level)
    return _LEVELS[max(0, min(len(_LEVELS) - 1, idx + delta))]


def session_key_from(session_header: str | None, system_prompt: str) -> str:
    """Derive a stable session key from a header or, failing that, the system prompt."""
    if session_header:
        return session_header.strip()
    digest = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    return f"sys:{digest[:16]}"


class AdaptiveController:
    """Per-session adaptive compression level chooser.

    Thread-safe: all session state is guarded by a single lock. Sessions are
    held in a bounded LRU so memory stays flat under churn.
    """

    def __init__(
        self,
        *,
        paradox_threshold: float = 0.25,
        min_quality: float = 0.6,
        window: int = 8,
        max_sessions: int = 1024,
    ) -> None:
        self._paradox_threshold = paradox_threshold
        self._min_quality = min_quality
        self._window = window
        self._max_sessions = max_sessions
        self._sessions: OrderedDict[str, _SessionState] = OrderedDict()
        self._stepups = 0
        self._stepdowns = 0
        self._spikes = 0
        self._lock = threading.Lock()

    def _touch(self, key: str) -> _SessionState:
        """Fetch (or create) a session, marking it most-recently-used. Caller holds lock."""
        state = self._sessions.get(key)
        if state is None:
            state = _SessionState(samples=deque(maxlen=self._window))
            self._sessions[key] = state
            if len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)
        else:
            self._sessions.move_to_end(key)
        return state

    def choose(self, session_key: str) -> Level:
        """Return the compression level to apply for this session's next request."""
        with self._lock:
            return self._touch(session_key).level

    def observe(
        self,
        session_key: str,
        *,
        aggression: Level,
        completion_tokens: int,
        quality_score: float | None = None,
        cached_tokens: int = 0,
        context_fill: float = 0.0,
    ) -> bool:
        """Record a request outcome and adjust the session's level.

        Returns True if a verbosity spike (compression paradox) was detected on
        this observation.
        """
        with self._lock:
            state = self._touch(session_key)
            prev = list(state.samples)
            state.samples.append(
                _Sample(aggression, completion_tokens, quality_score, cached_tokens)
            )

            spike = False
            new_level = state.level

            # Quality floor: measured degradation forces a step down immediately.
            if quality_score is not None and quality_score < self._min_quality:
                new_level = _step(state.level, -1)
            else:
                # Compression-paradox guard: did output balloon relative to this
                # session's recent baseline? When it does and we have room to back
                # off (not already at the weakest level), step down — aggressive
                # compression that inflates output is a net loss.
                baseline = [s.completion_tokens for s in prev if s.completion_tokens > 0]
                if baseline and completion_tokens > 0 and aggression != _LEVELS[0]:
                    base = median(baseline)
                    if base > 0 and (completion_tokens - base) / base >= self._paradox_threshold:
                        spike = True
                        new_level = _step(state.level, -1)

                # Opportunistic tightening: quality healthy and context filling up.
                if (
                    not spike
                    and quality_score is not None
                    and quality_score >= self._min_quality
                    and context_fill >= 0.7
                ):
                    new_level = _step(state.level, +1)

            if new_level != state.level:
                if _LEVELS.index(new_level) > _LEVELS.index(state.level):
                    self._stepups += 1
                else:
                    self._stepdowns += 1
                state.level = new_level

            if spike:
                self._spikes += 1
            state.last_spike = spike
            return spike

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "adaptive_sessions_total": len(self._sessions),
                "adaptive_stepups_total": self._stepups,
                "adaptive_stepdowns_total": self._stepdowns,
                "verbosity_spikes_total": self._spikes,
            }
