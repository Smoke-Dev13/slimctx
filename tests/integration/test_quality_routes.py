"""Integration tests for /stats and /quality observability endpoints (M6)."""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from contextly.ab_monitor import ABMonitor, ABSample
from contextly.config import Config
from contextly.server import create_app

# Long prose that the prose compressor will compress
_LONG_PROSE = (
    "Machine learning models require careful hyperparameter tuning to achieve "
    "optimal performance on the target task. The learning rate is one of the "
    "most critical parameters in gradient-based optimization algorithms. "
    "Regularization techniques such as dropout and weight decay help prevent "
    "overfitting to the training data. Transfer learning leverages pre-trained "
    "weights to reduce the amount of labeled data needed for fine-tuning. "
    "Ensemble methods combine predictions from multiple models to reduce variance. "
) * 3

_CHAT_RESPONSE = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "choices": [
        {"message": {"role": "assistant", "content": "Understood."}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
}


@pytest.fixture
def client() -> TestClient:
    config = Config(upstream_api_key="test-key", compression_enabled=True)
    app = create_app(config)
    return TestClient(app)


# ── /stats ────────────────────────────────────────────────────────────────────


def test_stats_returns_200(client: TestClient) -> None:
    resp = client.get("/stats")
    assert resp.status_code == 200


def test_stats_has_required_fields(client: TestClient) -> None:
    body = client.get("/stats").json()
    assert "requests_total" in body
    assert "requests_compressed" in body
    assert "chars_saved_total" in body
    assert "compression_ratio_mean" in body
    assert "ab_samples_total" in body


def test_stats_initial_values(client: TestClient) -> None:
    body = client.get("/stats").json()
    assert body["requests_total"] == 0
    assert body["requests_compressed"] == 0
    assert body["chars_saved_total"] == 0
    assert body["ab_samples_total"] == 0


@pytest.mark.allow_hosts(["127.0.0.1", "::1"])
@respx.mock
def test_stats_increments_after_compressed_request() -> None:
    config = Config(upstream_api_key="test-key", compression_enabled=True)
    app = create_app(config)

    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_CHAT_RESPONSE)
    )

    with TestClient(app) as c:
        c.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": _LONG_PROSE}]},
        )
        body = c.get("/stats").json()

    assert body["requests_total"] == 1
    assert body["requests_compressed"] == 1
    assert body["chars_saved_total"] > 0


@pytest.mark.allow_hosts(["127.0.0.1", "::1"])
@respx.mock
def test_stats_does_not_increment_for_short_messages() -> None:
    config = Config(upstream_api_key="test-key", compression_enabled=True)
    app = create_app(config)

    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_CHAT_RESPONSE)
    )

    with TestClient(app) as c:
        c.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi!"}]},
        )
        body = c.get("/stats").json()

    # Short content → no chars counted (content_router ran, but no char tracking
    # because total_original_chars > 0 check fails for empty-ish messages)
    assert body["requests_total"] == 0 or body["requests_compressed"] == 0


# ── /quality ──────────────────────────────────────────────────────────────────


def test_quality_returns_200(client: TestClient) -> None:
    assert client.get("/quality").status_code == 200


def test_quality_empty_report(client: TestClient) -> None:
    body = client.get("/quality").json()
    assert body["samples_total"] == 0
    assert body["quality"] is None
    assert body["chars_saved"] is None
    assert body["by_compressor"] == {}


def test_quality_report_after_injected_samples() -> None:
    import time

    config = Config(upstream_api_key="test-key", compression_enabled=True)
    app = create_app(config)

    monitor: ABMonitor = app.state.ab_monitor
    for score in [0.8, 0.85, 0.9, 0.95, 0.75]:
        monitor.record_sample(
            ABSample(
                timestamp=time.time(),
                model="gpt-4o",
                compressor="prose",
                original_chars=1000,
                compressed_chars=300,
                quality_score=score,
                reference_response_len=200,
                compressed_response_len=190,
            )
        )

    with TestClient(app) as c:
        body = c.get("/quality").json()

    assert body["samples_total"] == 5
    assert body["quality"]["mean"] == pytest.approx(0.85, abs=1e-2)
    assert "prose" in body["by_compressor"]
    assert body["by_compressor"]["prose"]["samples"] == 5
    assert body["chars_saved"]["total"] == 5 * 700


def test_quality_report_has_percentile_keys() -> None:
    import time

    config = Config(upstream_api_key="test-key", compression_enabled=True)
    app = create_app(config)

    monitor: ABMonitor = app.state.ab_monitor
    monitor.record_sample(
        ABSample(
            timestamp=time.time(),
            model="gpt-4o",
            compressor="prose",
            original_chars=500,
            compressed_chars=100,
            quality_score=0.9,
            reference_response_len=100,
            compressed_response_len=95,
        )
    )

    with TestClient(app) as c:
        body = c.get("/quality").json()

    q = body["quality"]
    assert "mean" in q
    assert "p10" in q
    assert "p50" in q
    assert "p90" in q


# ── A/B shadow integration ────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.allow_hosts(["127.0.0.1", "::1"])
async def test_ab_shadow_fires_and_records_sample() -> None:
    """With ab_sample_rate=1.0, a shadow request fires on every compressed request."""
    import httpx as _httpx
    from httpx import ASGITransport

    config = Config(
        upstream_api_key="test-key",
        compression_enabled=True,
        ab_sample_rate=1.0,
    )
    app = create_app(config)

    call_count = 0

    async def fake_upstream(request: _httpx.Request) -> _httpx.Response:
        nonlocal call_count
        call_count += 1
        return _httpx.Response(200, json=_CHAT_RESPONSE)

    # Patch the shared http_client with a mock transport
    async with _httpx.AsyncClient(
        transport=_httpx.MockTransport(fake_upstream),
        base_url="https://api.openai.com",
    ) as mock_client:
        # Inject the mock client into app.state BEFORE starting
        app.state.http_client = mock_client  # type: ignore[attr-defined]

        async with _httpx.AsyncClient(transport=ASGITransport(app=app)) as client:
            response = await client.post(
                "http://testserver/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": _LONG_PROSE}],
                },
                headers={"Authorization": "Bearer test"},
            )

        assert response.status_code == 200

        # Let the background shadow task run
        await asyncio.sleep(0.1)

    monitor: ABMonitor = app.state.ab_monitor
    # Both the main request and the shadow request must have hit the upstream
    assert call_count >= 2
    assert len(monitor) == 1
    report = monitor.quality_report()
    assert report["samples_total"] == 1
    assert report["quality"]["mean"] is not None
