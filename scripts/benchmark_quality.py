#!/usr/bin/env python3
"""Offline compression benchmark: token savings vs. information retention.

Contextly's compressors are *lossy* — they drop JSON records and prose
sentences. Raw "tokens saved" numbers are therefore misleading on their own, so
this script reports savings *alongside* a retention metric for each content
type, making the trade-off explicit:

    JSON   — fraction of records and distinct field-keys retained
    prose  — fraction of sentences and of numeric facts retained
    code   — fraction of function/class signatures retained (logic preserved)

No network access and no LLM calls are required; token counts use the bundled
tiktoken encodings when present and fall back to the word tokenizer otherwise.

Usage:
    python scripts/benchmark_quality.py
    python scripts/benchmark_quality.py --model gpt-4o --query "count the users"
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass

from contextly.ab_monitor import _extract_numbers
from contextly.compressors.base import Compressor, CompressResult
from contextly.compressors.code import CodeCompressor
from contextly.compressors.json_smart import JsonSmartCompressor
from contextly.compressors.json_table import JsonTableCompressor
from contextly.compressors.prose import ProseCompressor
from contextly.tokenizer.registry import get_tokenizer

# ── Representative fixtures ─────────────────────────────────────────────────────

_JSON_SAMPLE = json.dumps(
    [
        {"id": i, "user": f"user_{i}", "country": ["US", "DE", "JP"][i % 3], "amount": i * 7.5}
        for i in range(200)
    ]
)

_PROSE_SAMPLE = (
    "The quarterly report shows revenue of 1,240,000 dollars, up 12 percent year over year. "
    "Customer churn fell to 3 percent from 5 percent in the previous quarter. "
    "The engineering team shipped 47 features and resolved 312 support tickets. "
    "Marketing spend was reduced by 18 percent while lead volume grew by 9 percent. "
    "The board approved a hiring plan to add 25 engineers over the next two quarters. "
    "Average response time improved from 850 milliseconds to 420 milliseconds. "
    "Net promoter score increased from 41 to 58 across the measured cohorts. "
    "International revenue now accounts for 34 percent of the total, led by Germany and Japan. "
) * 2

_CODE_SAMPLE = '''\
import os  # standard library


class OrderService:
    """Handles order lifecycle operations."""

    def create_order(self, items, user_id):
        # Validate the cart before persisting anything.
        total = sum(i.price for i in items)
        return {"user": user_id, "total": total}

    def cancel_order(self, order_id):
        # Refund happens asynchronously elsewhere.
        return self._mark_cancelled(order_id)


def load_config(path):
    # Read JSON config from disk.
    with open(path) as f:
        return f.read()
'''


@dataclass(frozen=True)
class Row:
    content_type: str
    compressor: str
    tokens_before: int
    tokens_after: int
    retention_label: str
    retention_pct: float

    @property
    def token_reduction_pct(self) -> float:
        if self.tokens_before == 0:
            return 0.0
        return 100.0 * (1.0 - self.tokens_after / self.tokens_before)


# ── Retention metrics ───────────────────────────────────────────────────────────


def _record_count(parsed: object) -> int:
    """Count records in a list payload or a {"rows": [...]} table payload."""
    if isinstance(parsed, list):
        return len(parsed)
    if isinstance(parsed, dict) and isinstance(parsed.get("rows"), list):
        return len(parsed["rows"])
    return 1


def _json_retention(original: str, result: CompressResult) -> tuple[str, float]:
    """Fraction of records retained in the compressed JSON."""
    try:
        before = json.loads(original)
        after = json.loads(result.content)
    except json.JSONDecodeError:
        return "records retained", 100.0
    n_before = _record_count(before)
    n_after = _record_count(after)
    pct = 100.0 * n_after / n_before if n_before else 100.0
    return f"records retained ({n_after}/{n_before})", pct


def _prose_retention(original: str, result: CompressResult) -> tuple[str, float]:
    """Fraction of numeric facts preserved — the figures an LLM would quote."""
    before_nums = _extract_numbers(original)
    if not sum(before_nums.values()):
        return "numeric facts retained", 100.0
    after_nums = _extract_numbers(result.content)
    preserved = sum((before_nums & after_nums).values())
    pct = 100.0 * preserved / sum(before_nums.values())
    return f"numeric facts retained ({preserved}/{sum(before_nums.values())})", pct


_SIG_RE = re.compile(r"^\s*(?:def |class |func |function )", re.MULTILINE)


def _code_retention(original: str, result: CompressResult) -> tuple[str, float]:
    """Fraction of function/class signatures retained (logic is preserved)."""
    n_before = len(_SIG_RE.findall(original))
    n_after = len(_SIG_RE.findall(result.content))
    pct = 100.0 * n_after / n_before if n_before else 100.0
    return f"signatures retained ({n_after}/{n_before})", pct


# ── Runner ──────────────────────────────────────────────────────────────────────


def _benchmark(model: str, query: str) -> list[Row]:
    tok = get_tokenizer(model)
    cases: list[tuple[str, str, Compressor, object]] = [
        ("JSON (default, lossless)", _JSON_SAMPLE, JsonTableCompressor(), _json_retention),
        ("JSON (opt-in sampling)", _JSON_SAMPLE, JsonSmartCompressor(), _json_retention),
        ("prose", _PROSE_SAMPLE, ProseCompressor(), _prose_retention),
        ("code", _CODE_SAMPLE, CodeCompressor(), _code_retention),
    ]
    rows: list[Row] = []
    for content_type, sample, compressor, retention_fn in cases:
        result = compressor.compress(sample, query)
        label, pct = retention_fn(sample, result)  # type: ignore[operator]
        rows.append(
            Row(
                content_type=content_type,
                compressor=result.compressor_name,
                tokens_before=tok.count(sample),
                tokens_after=tok.count(result.content),
                retention_label=label,
                retention_pct=pct,
            )
        )
    return rows


def _print_markdown(rows: list[Row], model: str, query: str) -> None:
    print(f"\n# Compression benchmark (model={model!r}, query={query!r})\n")
    print("| Content | Compressor | Tokens before | Tokens after | Tokens saved | Retention |")
    print("|---|---|---:|---:|---:|---|")
    for r in rows:
        print(
            f"| {r.content_type} | {r.compressor} | {r.tokens_before} | {r.tokens_after} "
            f"| {r.token_reduction_pct:.0f}% | {r.retention_pct:.0f}% {r.retention_label} |"
        )
    print(
        "\n> Retention is the share of the original information the model still sees. "
        "High token savings with low retention means the proxy is answering from a "
        "fraction of the data — safe for aggregate questions, risky for lookups. "
        "Run the proxy with `--safe-mode` to keep retention at 100% for JSON and prose."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-4o", help="Model name for token counting")
    parser.add_argument("--query", default="", help="User query (affects aggressiveness)")
    args = parser.parse_args()
    rows = _benchmark(args.model, args.query)
    _print_markdown(rows, args.model, args.query)


if __name__ == "__main__":
    main()
