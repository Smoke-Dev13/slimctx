"""Prometheus metrics for Contextly.

All metrics use the ``contextly_`` prefix and the default process-wide registry.
Call ``observe_request`` from the route handler and ``observe_ab_sample`` from
ABMonitor.record_sample(); read ``get_metrics_bytes`` in the /metrics endpoint.

Metric catalogue:
  contextly_requests_total              Counter   [model, compressor]
  contextly_chars_saved_total           Counter   [compressor]
  contextly_compression_ratio           Histogram [compressor]
  contextly_request_latency_seconds     Histogram [model]
  contextly_ab_quality_score            Histogram [compressor]
"""

from __future__ import annotations

from typing import Any

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

__all__ = [
    "CONTENT_TYPE_LATEST",
    "DOLLARS_SAVED_TOTAL",
    "TOKENS_SAVED_TOTAL",
    "get_metrics_bytes",
    "observe_ab_sample",
    "observe_request",
]

# ── Metric definitions ────────────────────────────────────────────────────────

REQUESTS_TOTAL: Any = Counter(
    "contextly_requests_total",
    "Total chat/completions requests proxied (non-streaming only).",
    ["model", "compressor"],
)

CHARS_SAVED_TOTAL: Any = Counter(
    "contextly_chars_saved_total",
    "Cumulative characters saved by compression across all requests.",
    ["compressor"],
)

COMPRESSION_RATIO: Any = Histogram(
    "contextly_compression_ratio",
    "Compression ratio per request (compressed_chars / original_chars). "
    "1.0 means no compression (passthrough).",
    ["compressor"],
    buckets=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
)

REQUEST_LATENCY: Any = Histogram(
    "contextly_request_latency_seconds",
    "End-to-end latency from request arrival to upstream response, in seconds.",
    ["model"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

TOKENS_SAVED_TOTAL: Any = Counter(
    "contextly_tokens_saved_total",
    "Cumulative tokens saved (estimate) by compression across all requests.",
    ["compressor"],
)

DOLLARS_SAVED_TOTAL: Any = Counter(
    "contextly_dollars_saved_total",
    "Cumulative estimated USD saved by compression (tokens_saved * price/token).",
    ["model", "compressor"],
)

AB_QUALITY_SCORE: Any = Histogram(
    "contextly_ab_quality_score",
    "A/B shadow quality score (word-level ROUGE-1 F1) per sampled request. "
    "1.0 = compressed response matches original exactly.",
    ["compressor"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
)


# ── Recording helpers ─────────────────────────────────────────────────────────


def observe_request(
    *,
    model: str,
    compressor: str,
    original_chars: int,
    compressed_chars: int,
    latency_seconds: float,
    tokens_saved_estimate: int = 0,
    dollars_saved: float = 0.0,
) -> None:
    """Update all request-level Prometheus metrics for one proxied request.

    Args:
        model: LLM model identifier (becomes the `model` label).
        compressor: Name of the dominant compressor used (e.g. "prose").
        original_chars: Total chars in original messages before compression.
        compressed_chars: Total chars after compression.
        latency_seconds: Wall-clock seconds from request start to upstream reply.
        tokens_saved_estimate: Estimated tokens saved (chars // 4 when exact count unavailable).
        dollars_saved: Estimated USD saved (tokens_saved * model price/token).
    """
    REQUESTS_TOTAL.labels(model=model, compressor=compressor).inc()

    chars_saved = max(0, original_chars - compressed_chars)
    if chars_saved > 0:
        CHARS_SAVED_TOTAL.labels(compressor=compressor).inc(chars_saved)

    if tokens_saved_estimate > 0:
        TOKENS_SAVED_TOTAL.labels(compressor=compressor).inc(tokens_saved_estimate)

    if dollars_saved > 0:
        DOLLARS_SAVED_TOTAL.labels(model=model, compressor=compressor).inc(dollars_saved)

    if original_chars > 0:
        ratio = compressed_chars / original_chars
        COMPRESSION_RATIO.labels(compressor=compressor).observe(ratio)

    REQUEST_LATENCY.labels(model=model).observe(latency_seconds)


def observe_ab_sample(*, compressor: str, quality_score: float) -> None:
    """Record one A/B quality score in the Prometheus histogram.

    Args:
        compressor: Name of the compressor used (label).
        quality_score: ROUGE-1 F1 score in [0.0, 1.0].
    """
    AB_QUALITY_SCORE.labels(compressor=compressor).observe(quality_score)


def get_metrics_bytes() -> bytes:
    """Return the full Prometheus text exposition as bytes.

    Returns:
        UTF-8 bytes in Prometheus text format 0.0.4.
    """
    return bytes(generate_latest())
