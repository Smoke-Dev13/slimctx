"""Unit tests for Prometheus metrics helpers."""

from __future__ import annotations

from contextly.metrics import (
    get_metrics_bytes,
    observe_ab_sample,
    observe_request,
)

# ── observe_request ────────────────────────────────────────────────────────────


def test_observe_request_does_not_raise() -> None:
    observe_request(
        model="gpt-4o",
        compressor="prose",
        original_chars=1000,
        compressed_chars=300,
        latency_seconds=0.45,
    )


def test_observe_request_passthrough_does_not_raise() -> None:
    observe_request(
        model="gpt-4o-mini",
        compressor="passthrough",
        original_chars=50,
        compressed_chars=50,
        latency_seconds=0.1,
    )


def test_observe_request_zero_chars_does_not_raise() -> None:
    observe_request(
        model="unknown",
        compressor="passthrough",
        original_chars=0,
        compressed_chars=0,
        latency_seconds=0.0,
    )


def test_observe_request_all_compressors() -> None:
    for compressor in ("json_smart", "prose", "code", "passthrough"):
        observe_request(
            model="gpt-4o",
            compressor=compressor,
            original_chars=500,
            compressed_chars=200,
            latency_seconds=0.3,
        )


def test_observe_request_with_tokens_saved_estimate() -> None:
    observe_request(
        model="gpt-4o",
        compressor="prose",
        original_chars=2000,
        compressed_chars=800,
        latency_seconds=0.6,
        tokens_saved_estimate=300,
    )


def test_observe_request_tokens_saved_zero_does_not_raise() -> None:
    observe_request(
        model="gpt-4o",
        compressor="passthrough",
        original_chars=100,
        compressed_chars=100,
        latency_seconds=0.05,
        tokens_saved_estimate=0,
    )


# ── observe_ab_sample ─────────────────────────────────────────────────────────


def test_observe_ab_sample_does_not_raise() -> None:
    observe_ab_sample(compressor="prose", quality_score=0.87)


def test_observe_ab_sample_boundary_values() -> None:
    observe_ab_sample(compressor="json_smart", quality_score=0.0)
    observe_ab_sample(compressor="json_smart", quality_score=1.0)


def test_observe_ab_sample_multiple_compressors() -> None:
    for comp, score in [("prose", 0.9), ("code", 0.85), ("json_smart", 0.95)]:
        observe_ab_sample(compressor=comp, quality_score=score)


# ── get_metrics_bytes ─────────────────────────────────────────────────────────


def test_get_metrics_bytes_returns_bytes() -> None:
    result = get_metrics_bytes()
    assert isinstance(result, bytes)


def test_get_metrics_bytes_nonempty() -> None:
    result = get_metrics_bytes()
    assert len(result) > 0


def test_get_metrics_bytes_contains_requests_total() -> None:
    text = get_metrics_bytes().decode("utf-8")
    assert "contextly_requests_total" in text


def test_get_metrics_bytes_contains_chars_saved() -> None:
    text = get_metrics_bytes().decode("utf-8")
    assert "contextly_chars_saved_total" in text


def test_get_metrics_bytes_contains_compression_ratio() -> None:
    text = get_metrics_bytes().decode("utf-8")
    assert "contextly_compression_ratio" in text


def test_get_metrics_bytes_contains_latency() -> None:
    text = get_metrics_bytes().decode("utf-8")
    assert "contextly_request_latency_seconds" in text


def test_get_metrics_bytes_contains_ab_quality() -> None:
    text = get_metrics_bytes().decode("utf-8")
    assert "contextly_ab_quality_score" in text


def test_get_metrics_bytes_is_valid_prometheus_text() -> None:
    text = get_metrics_bytes().decode("utf-8")
    # Prometheus text format begins with # HELP and # TYPE comment lines
    lines = text.splitlines()
    help_lines = [ln for ln in lines if ln.startswith("# HELP")]
    type_lines = [ln for ln in lines if ln.startswith("# TYPE")]
    assert len(help_lines) >= 5
    assert len(type_lines) >= 5


def test_get_metrics_bytes_utf8_decodable() -> None:
    raw = get_metrics_bytes()
    decoded = raw.decode("utf-8")
    assert len(decoded) > 0
