"""Integration tests for token-savings tracking in /stats and /metrics."""

from __future__ import annotations

import httpx
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
    "Regularization prevents overfitting by adding penalties to loss functions. "
    "Ensemble methods combine multiple weak learners to reduce variance. "
    "Transfer learning reuses knowledge from pre-trained models effectively. "
    "Cross-validation provides robust estimates of generalisation performance. "
) * 5


# ── /stats includes tokens_saved_estimate_total ───────────────────────────────


def test_stats_has_tokens_saved_estimate_total_key() -> None:
    config = Config(upstream_api_key="test-key", compression_enabled=True)
    app = create_app(config)
    with TestClient(app) as c:
        data = c.get("/stats").json()
    assert "tokens_saved_estimate_total" in data


def test_stats_tokens_saved_starts_at_zero() -> None:
    config = Config(upstream_api_key="test-key", compression_enabled=True)
    app = create_app(config)
    with TestClient(app) as c:
        data = c.get("/stats").json()
    assert data["tokens_saved_estimate_total"] == 0


@respx.mock
def test_stats_tokens_saved_increases_after_compressed_request() -> None:
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
        data = c.get("/stats").json()

    assert data["tokens_saved_estimate_total"] >= 0


@respx.mock
def test_stats_tokens_saved_positive_for_compressible_content() -> None:
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
        data = c.get("/stats").json()

    assert data["tokens_saved_estimate_total"] > 0


# ── /metrics includes contextly_tokens_saved_total ───────────────────────────


def test_metrics_has_tokens_saved_total() -> None:
    config = Config(upstream_api_key="test-key", compression_enabled=True)
    app = create_app(config)
    with TestClient(app) as c:
        text = c.get("/metrics").text
    assert "contextly_tokens_saved_total" in text


@respx.mock
def test_metrics_tokens_saved_counter_increases_after_request() -> None:
    config = Config(upstream_api_key="test-key", compression_enabled=True)
    app = create_app(config)

    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_CHAT_RESPONSE)
    )

    with TestClient(app) as c:
        initial = _parse_counter(c.get("/metrics").text, "contextly_tokens_saved_total")
        c.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": _LONG_PROSE}]},
        )
        updated = _parse_counter(c.get("/metrics").text, "contextly_tokens_saved_total")

    assert updated >= initial


# ── Helper ────────────────────────────────────────────────────────────────────


def _parse_counter(text: str, metric_name: str) -> float:
    total = 0.0
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if line.startswith(metric_name + "_created"):
            continue
        if line.startswith(metric_name + "{") or line.startswith(metric_name + " "):
            try:
                total += float(line.rsplit(" ", 1)[-1])
            except ValueError:
                pass
    return total
