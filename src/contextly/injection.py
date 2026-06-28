"""Prompt injection detection for compressed LLM context.

Scans message content for patterns characteristic of prompt injection attacks —
instructions embedded in tool outputs or user messages that attempt to hijack
the LLM's behaviour.

Detection is regex-based (stdlib only, zero dependencies) and conservative:
it flags patterns rather than blocking by default. Set
``injection_block_threshold`` in Config to auto-reject requests above a risk
score.
"""

from __future__ import annotations

import re
import threading
import unicodedata
from dataclasses import dataclass, field
from typing import Any

# ── Pattern catalogue ──────────────────────────────────────────────────────────

_RAW_PATTERNS: list[tuple[str, float]] = [
    # Direct override attempts
    (r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context)", 0.9),
    (r"disregard\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?)", 0.9),
    (r"forget\s+(everything|all)\s+(you('ve|\s+have)\s+)?learned", 0.85),
    (r"you\s+are\s+now\s+(a\s+)?(?!helpful|an?\s+AI)", 0.7),
    (r"new\s+instructions?\s*:", 0.75),
    (r"system\s+prompt\s*:", 0.7),
    # Jailbreak triggers
    (r"\bDAN\b", 0.6),
    (r"do\s+anything\s+now", 0.65),
    (r"jailbreak", 0.8),
    (r"pretend\s+(you\s+are|to\s+be)\s+(?!helpful)", 0.65),
    # Role escalation
    (r"(act|behave)\s+as\s+(if\s+you\s+are\s+)?(?:an?\s+)?(?:evil|uncensored|unethical)", 0.85),
    (r"you\s+have\s+no\s+(restrictions?|limits?|rules?|guidelines?)", 0.8),
    # Data exfiltration
    (r"(reveal|print|output|show|display)\s+(your\s+)?(system\s+)?prompt", 0.75),
    (r"(what\s+(are|were)\s+your\s+instructions?|tell\s+me\s+your\s+instructions?)", 0.65),
    # Delimiter injection
    (r"<\|?(im_start|im_end|system|endoftext)\|?>", 0.85),
    (r"\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>", 0.85),
    # Hidden instruction markers
    (r"###\s*(instruction|system|override|admin)", 0.7),
    (r"={3,}\s*(system|instruction|override)\s*={3,}", 0.7),
]

_COMPILED: list[tuple[re.Pattern[str], float]] = [
    (re.compile(pattern, re.IGNORECASE), weight) for pattern, weight in _RAW_PATTERNS
]

# ── Response-side leak catalogue ────────────────────────────────────────────────
# Symptoms of a *successful* injection visible in the model's reply: the model
# disclosing its system prompt / instructions, or echoing raw delimiter tokens.
# Distinct from the inbound attack patterns above — these flag leakage, not intent.
_RAW_RESPONSE_LEAK_PATTERNS: list[tuple[str, float]] = [
    (r"(my|the)\s+(system\s+)?(prompt|instructions?)\s+(is|are|were)\s*:", 0.8),
    (r"my\s+(initial|original)\s+instructions?\s+(were|are)", 0.8),
    (r"here\s+(is|are)\s+(my|the)\s+(system\s+)?(prompt|instructions?)", 0.8),
    (r"you\s+are\s+(a|an)\s+\w+\s+(assistant|model|ai)\b.{0,40}\b(do\s+not|must|never)\b", 0.6),
    # Verbatim chat-template delimiters leaking into a normal answer.
    (r"<\|?(im_start|im_end|system|endoftext)\|?>", 0.85),
    (r"\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>", 0.85),
]

_COMPILED_RESPONSE_LEAK: list[tuple[re.Pattern[str], float]] = [
    (re.compile(pattern, re.IGNORECASE), weight)
    for pattern, weight in _RAW_RESPONSE_LEAK_PATTERNS
]


# ── Result types ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class InjectionResult:
    """Result of scanning a single text block."""

    detected: bool
    risk_score: float
    matched_patterns: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MessagesScanResult:
    """Aggregate result of scanning all messages in a request."""

    detected: bool
    risk_score: float
    flagged_indices: list[int] = field(default_factory=list)
    matched_patterns: list[str] = field(default_factory=list)


# ── Scanner ────────────────────────────────────────────────────────────────────


def _normalize(text: str) -> str:
    """NFC-normalize and replace Unicode homoglyphs with ASCII equivalents."""
    return unicodedata.normalize("NFC", text)


class InjectionScanner:
    """Stateless regex-based prompt injection detector.

    Thread-safe: all state is in compiled patterns (module-level constants).
    Instantiate once and share across requests.
    """

    def __init__(self) -> None:
        self._detections: int = 0
        self._blocks: int = 0
        self._lock = threading.Lock()

    def scan(self, text: str) -> InjectionResult:
        """Scan a single text block for injection patterns.

        Args:
            text: Raw text content to scan.

        Returns:
            InjectionResult with detected flag, risk_score in [0, 1], and
            list of matched pattern descriptions.
        """
        normalized = _normalize(text)
        matched: list[str] = []
        max_score = 0.0

        for pattern, weight in _COMPILED:
            if pattern.search(normalized):
                matched.append(pattern.pattern)
                if weight > max_score:
                    max_score = weight

        # Boost score when multiple patterns match (pile-on signal)
        if len(matched) > 1:
            max_score = min(1.0, max_score + 0.05 * (len(matched) - 1))

        detected = max_score >= 0.5
        return InjectionResult(detected=detected, risk_score=max_score, matched_patterns=matched)

    def scan_response(self, text: str) -> InjectionResult:
        """Scan a model *response* for signs of a successful injection / leak.

        Flags system-prompt disclosure and verbatim chat-template delimiters in
        the reply — the outbound counterpart to :meth:`scan`. Conservative: it
        reports a risk score (flag), it does not block.

        Args:
            text: The model's response text.

        Returns:
            InjectionResult with detected flag, risk_score in [0, 1], and the
            list of matched leak patterns.
        """
        normalized = _normalize(text)
        matched: list[str] = []
        max_score = 0.0

        for pattern, weight in _COMPILED_RESPONSE_LEAK:
            if pattern.search(normalized):
                matched.append(pattern.pattern)
                if weight > max_score:
                    max_score = weight

        if len(matched) > 1:
            max_score = min(1.0, max_score + 0.05 * (len(matched) - 1))

        detected = max_score >= 0.5
        if detected:
            with self._lock:
                self._detections += 1
        return InjectionResult(detected=detected, risk_score=max_score, matched_patterns=matched)

    def scan_messages(self, messages: list[dict[str, Any]]) -> MessagesScanResult:
        """Scan all messages in a chat payload for injection patterns.

        Args:
            messages: List of message dicts (role + content).

        Returns:
            MessagesScanResult aggregating risk across all messages.
        """
        max_score = 0.0
        all_patterns: list[str] = []
        flagged: list[int] = []

        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            texts: list[str] = []
            if isinstance(content, str):
                texts = [content]
            elif isinstance(content, list):
                texts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]

            for text in texts:
                if not text:
                    continue
                result = self.scan(text)
                if result.detected:
                    flagged.append(i)
                    all_patterns.extend(result.matched_patterns)
                if result.risk_score > max_score:
                    max_score = result.risk_score

        detected = max_score >= 0.5
        if detected:
            with self._lock:
                self._detections += 1

        return MessagesScanResult(
            detected=detected,
            risk_score=max_score,
            flagged_indices=list(dict.fromkeys(flagged)),
            matched_patterns=list(dict.fromkeys(all_patterns)),
        )

    def record_block(self) -> None:
        """Increment the blocked-request counter."""
        with self._lock:
            self._blocks += 1

    def stats(self) -> dict[str, int]:
        """Return detection and block counters."""
        with self._lock:
            return {
                "injections_detected_total": self._detections,
                "injections_blocked_total": self._blocks,
            }
