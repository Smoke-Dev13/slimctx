"""Semantic firewall — outbound secret / PII redaction.

The inverse of ``injection.py``: where ``InjectionScanner`` guards against
malicious instructions coming *in*, ``SecretRedactor`` stops sensitive data
(API keys, private keys, credit cards, SSNs, emails) leaking *out* to the
upstream LLM. Detected secrets are replaced with stable placeholders before the
payload is compressed, stored, or logged, so they never reach the compressor,
the CCR store, or the audit log in the clear.

Stdlib only (``re`` + ``hashlib``), thread-safe. Optionally reversible: when a
CCR store is supplied the original secret is stored and its key embedded in the
placeholder so ``/v1/expand`` can recover it (opt-in — storing secrets is a
deliberate tradeoff).
"""

from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import dataclass, field
from typing import Any

# ── Pattern catalogue (precision over recall) ───────────────────────────────────

_RAW_PATTERNS: list[tuple[str, str]] = [
    # Provider keys — order matters: anthropic before the generic openai sk- rule.
    ("anthropic_key", r"sk-ant-[A-Za-z0-9_-]{20,}"),
    ("openai_key", r"sk-[A-Za-z0-9]{20,}"),
    ("github_token", r"gh[opsu]_[A-Za-z0-9]{36,}"),
    ("aws_access_key", r"AKIA[0-9A-Z]{16}"),
    ("slack_token", r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    ("private_key_block", r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    ("jwt", r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    ("ssn", r"\b\d{3}-\d{2}-\d{4}\b"),
    ("email", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    # Candidate card numbers (13-16 digits, optional separators); Luhn-checked below.
    ("credit_card", r"\b(?:\d[ -]?){13,16}\b"),
]

_COMPILED: list[tuple[str, re.Pattern[str]]] = [
    (name, re.compile(pattern)) for name, pattern in _RAW_PATTERNS
]


def _luhn_valid(candidate: str) -> bool:
    """Return True if the digit string passes the Luhn checksum."""
    digits = [int(c) for c in candidate if c.isdigit()]
    if len(digits) < 13:
        return False
    total = 0
    parity = len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# ── Result types ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Finding:
    """A single detected secret and the placeholder that replaced it."""

    type: str
    placeholder: str


@dataclass(frozen=True)
class RedactionResult:
    """Result of redacting a single text block."""

    redacted_text: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.findings)


# ── Redactor ─────────────────────────────────────────────────────────────────────


class SecretRedactor:
    """Detect and redact secrets / PII in outbound message content.

    Thread-safe: counters guarded by a lock; patterns are module-level constants.
    """

    def __init__(self) -> None:
        self._secrets_redacted = 0
        self._requests_with_secrets = 0
        self._lock = threading.Lock()

    def _placeholder(self, secret: str, kind: str, ccr_store: Any | None) -> str:
        digest = hashlib.sha256(secret.encode("utf-8")).hexdigest()[:8]
        if ccr_store is not None:
            key = ccr_store.store(secret)
            return f'[REDACTED:{kind}:{digest} contextly_expand("{key}")]'
        return f"[REDACTED:{kind}:{digest}]"

    def redact(self, text: str, *, ccr_store: Any | None = None) -> RedactionResult:
        """Redact secrets in *text*, returning the cleaned text and findings."""
        findings: list[Finding] = []
        redacted = text

        for kind, pattern in _COMPILED:

            def _replace(match: re.Match[str], _kind: str = kind) -> str:
                secret = match.group(0)
                if _kind == "credit_card" and not _luhn_valid(secret):
                    return secret  # not a real card number — leave untouched
                placeholder = self._placeholder(secret, _kind, ccr_store)
                findings.append(Finding(type=_kind, placeholder=placeholder))
                return placeholder

            redacted = pattern.sub(_replace, redacted)

        return RedactionResult(redacted_text=redacted, findings=findings)

    def redact_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        ccr_store: Any | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Redact secrets across all message text, returning new messages + count.

        Walks string content and ``{"type": "text"}`` blocks; other parts pass
        through untouched. Increments running counters.
        """
        out: list[dict[str, Any]] = []
        total = 0

        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                result = self.redact(content, ccr_store=ccr_store)
                total += result.count
                out.append({**msg, "content": result.redacted_text})
            elif isinstance(content, list):
                parts: list[Any] = []
                for part in content:
                    if (
                        isinstance(part, dict)
                        and part.get("type") == "text"
                        and isinstance(part.get("text"), str)
                        and part["text"]
                    ):
                        result = self.redact(part["text"], ccr_store=ccr_store)
                        total += result.count
                        parts.append({**part, "text": result.redacted_text})
                    else:
                        parts.append(part)
                out.append({**msg, "content": parts})
            else:
                out.append(msg)

        if total:
            with self._lock:
                self._secrets_redacted += total
                self._requests_with_secrets += 1

        return out, total

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "secrets_redacted_total": self._secrets_redacted,
                "requests_with_secrets_total": self._requests_with_secrets,
            }
