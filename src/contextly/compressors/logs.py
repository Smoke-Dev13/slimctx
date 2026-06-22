"""Log / tool-output compressor — folds repetitive lines by template.

Server logs, tool outputs, and DB dumps are dominated by near-duplicate lines:
the same message with a different timestamp, id, or number on each row. This
compressor extracts a *template* per line (volatile tokens — timestamps, ids,
numbers, IPs, hex — replaced with ``*``), groups lines that share a template,
and emits each unique pattern once with an occurrence count:

    INFO  GET /users/* -> 200 in *ms   (x412)
    ERROR upstream timeout to *         (x7)

The first original line of each group is kept as the representative, so the text
stays concrete and readable. Repetition collapses dramatically (the big win on
machine-generated output) while every *distinct* message is preserved. It is
lossy for the exact values of folded duplicates, but the proxy stores the
original in the CCR store, so ``expand`` recovers the full log verbatim.
"""

from __future__ import annotations

import re

import structlog

from contextly.compressors.base import Compressor, CompressResult

logger = structlog.get_logger(__name__)

_MIN_LINES: int = 6

# A line looks log-like if it starts with a timestamp or a log level. The \b
# guards only the keyword branch (a date like 2026-06-22T has no word boundary
# after it, so it must not be required there).
_LOG_SIGNAL_RE = re.compile(
    r"^\s*[\[(]?(?:\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2}|"
    r"(?:DEBUG|INFO|INFORMATION|WARN|WARNING|ERROR|ERR|TRACE|FATAL|CRITICAL)\b)",
    re.IGNORECASE,
)

# Volatile tokens replaced with "*" when building a line template. Order matters:
# more specific patterns (timestamps, uuids, ip) run before the generic number.
_VOLATILE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?"),
    re.compile(r"\b\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\b"),
    re.compile(r"\b[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\b"),  # uuid
    re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),  # ipv4
    re.compile(r"0x[0-9a-fA-F]+"),
    re.compile(r"\b[0-9a-fA-F]{12,}\b"),  # long hex / hashes
    re.compile(r"-?\d+(?:\.\d+)?"),  # any number last
]


def _make_passthrough(content: str, name: str) -> CompressResult:
    length = len(content)
    return CompressResult(
        content=content,
        original_length=length,
        compressed_length=length,
        compressor_name=name,
    )


def _template(line: str) -> str:
    """Reduce a line to its template by masking volatile tokens with ``*``."""
    out = line
    for pat in _VOLATILE_PATTERNS:
        out = pat.sub("*", out)
    return out


class LogCompressor(Compressor):
    """Fold repetitive log / tool-output lines by template.

    Detection (should_apply): multi-line text where a majority of lines start
    with a timestamp or a log level.

    Compression pipeline:
      1. Split into lines; compute a masked template per line.
      2. Group consecutive-or-not lines by template, preserving first-seen order
         and counting occurrences.
      3. Emit each group's first original line, suffixed with " (xN)" when N > 1.
      4. Use the result only if it is actually shorter than the input.
    """

    @property
    def name(self) -> str:
        return "logs"

    def should_apply(self, content: str, query: str = "") -> bool:
        """Return True if content is multi-line and predominantly log-like."""
        lines = [ln for ln in content.splitlines() if ln.strip()]
        if len(lines) < _MIN_LINES:
            return False
        signal = sum(1 for ln in lines if _LOG_SIGNAL_RE.match(ln))
        return signal >= len(lines) * 0.5

    def compress(self, content: str, query: str = "") -> CompressResult:
        """Fold repeated log patterns into one representative line each."""
        original_length = len(content)
        lines = content.splitlines()

        order: list[str] = []
        counts: dict[str, int] = {}
        first_line: dict[str, str] = {}
        for line in lines:
            if not line.strip():
                continue
            tmpl = _template(line)
            if tmpl not in counts:
                counts[tmpl] = 0
                first_line[tmpl] = line
                order.append(tmpl)
            counts[tmpl] += 1

        if not order:
            return _make_passthrough(content, self.name)

        folded: list[str] = []
        for tmpl in order:
            rep = first_line[tmpl]
            n = counts[tmpl]
            folded.append(f"{rep}  (x{n})" if n > 1 else rep)

        total = sum(counts.values())
        header = f"# contextly: folded {total} log lines into {len(order)} unique patterns"
        compressed_content = header + "\n" + "\n".join(folded)
        compressed_length = len(compressed_content)

        if compressed_length >= original_length:
            return _make_passthrough(content, self.name)

        logger.info(
            "logs_compressed",
            total_lines=total,
            unique_patterns=len(order),
            ratio=round(compressed_length / original_length, 3),
        )

        return CompressResult(
            content=compressed_content,
            original_length=original_length,
            compressed_length=compressed_length,
            compressor_name=self.name,
            metadata={
                "total_lines": total,
                "unique_patterns": len(order),
                "lossless": False,
            },
        )
