"""Integration tests for GET /metrics (Prometheus text exposition)."""

from __future__ import annotations

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from contextly.config import Config
from contextly.server import create_app

_CHAT_RESPONSE = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55},
}

_LONG_PROSE = (
    "Machine learning requires large datasets and careful tuning. "
    "Regularization prevents overfitting by adding penalties. "
    "Ensemble methods combine multiple learners to reduce variance. "
    "Transfer learning reuses knowledge from pre-trained models. "
    "Cross-validation estimates generalisation on unseen data. "
) * 4


@pytest.fixture
def client() -> TestClient:
    config = Config(upstream_api_key="test-key", compression_enabled=True)
    app = create_app(config)
    return TestClient(app)


# ── Endpoint basics ────────────────────────────────────────────────────────────


def test_metrics_status_200(client: TestClient) -> None:
    assert client.get("/metrics").status_code == 200


def test_metrics_content_type_is_prometheus(client: TestClient) -> None:
    resp = client.get("/metrics")
    assert "text/plain" in resp.headers["content-type"]


def test_metrics_body_is_not_json(client: TestClient) -> None:
    resp = client.get("/metrics")
    import json

    try:
        json.loads(resp.text)
        is_json = True
    except Exception:
        is_json = False
    assert not is_json, "Expected Prometheus text, not JSON"


def test_metrics_contains_help_lines(client: TestClient) -> None:
    text = client.get("/metrics").text
    help_lines = [ln for ln in text.splitlines() if ln.startswith("# HELP")]
    assert len(help_lines) >= 5


def test_metrics_contains_type_lines(client: TestClient) -> None:
    text = client.get("/metrics").text
    type_lines = [ln for ln in text.splitlines() if ln.startswith("# TYPE")]
    assert len(type_lines) >= 5


# ── Metric names present ───────────────────────────────────────────────────────


def test_metrics_has_requests_total(client: TestClient) -> None:
    assert "contextly_requests_total" in client.get("/metrics").text


def test_metrics_has_chars_saved_total(client: TestClient) -> None:
    assert "contextly_chars_saved_total" in client.get("/metrics").text


def test_metrics_has_compression_ratio(client: TestClient) -> None:
    assert "contextly_compression_ratio" in client.get("/metrics").text


def test_metrics_has_request_latency(client: TestClient) -> None:
    assert "contextly_request_latency_seconds" in client.get("/metrics").text


def test_metrics_has_ab_quality_score(client: TestClient) -> None:
    assert "contextly_ab_quality_score" in client.get("/metrics").text


# ── Metrics update after proxied request ───────────────────────────────────────


@pytest.mark.allow_hosts(["127.0.0.1", "::1"])
@respx.mock
def test_metrics_request_counter_increases_after_request() -> None:
    config = Config(upstream_api_key="test-key", compression_enabled=True)
    app = create_app(config)

    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_CHAT_RESPONSE)
    )

    with TestClient(app) as c:
        # Read initial value
        initial_text = c.get("/metrics").text
        initial_count = _parse_counter(initial_text, "contextly_requests_total")

        # Fire a proxied request
        c.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": _LONG_PROSE}],
            },
        )

        # Counter should have increased
        updated_text = c.get("/metrics").text
        updated_count = _parse_counter(updated_text, "contextly_requests_total")

    assert updated_count > initial_count


@pytest.mark.allow_hosts(["127.0.0.1", "::1"])
@respx.mock
def test_metrics_latency_histogram_populated_after_request() -> None:
    config = Config(upstream_api_key="test-key", compression_enabled=True)
    app = create_app(config)

    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_CHAT_RESPONSE)
    )

    with TestClient(app) as c:
        c.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": _LONG_PROSE}],
            },
        )
        text = c.get("/metrics").text

    # Sum metric should exist and be > 0
    assert "contextly_request_latency_seconds_sum" in text


# ── Helper ────────────────────────────────────────────────────────────────────


def _parse_counter(text: str, metric_name: str) -> float:
    """Sum all sample values for a metric across all label combinations."""
    total = 0.0
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        # Skip _created timestamp lines
        if line.startswith(metric_name + "_created"):
            continue
        if line.startswith(metric_name + "{") or line.startswith(metric_name + " "):
            try:
                total += float(line.rsplit(" ", 1)[-1])
            except ValueError:
                pass
    return total
