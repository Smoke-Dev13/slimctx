"""Integration tests for prompt-cache optimization."""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from contextly.config import Config
from contextly.server import create_app

_BIG = "x" * 6000

_CHAT_RESPONSE = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "model": "gpt-4o",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}
    ],
    "usage": {
        "prompt_tokens": 2000,
        "completion_tokens": 2,
        "total_tokens": 2002,
        "prompt_tokens_details": {"cached_tokens": 1500},
    },
}

_MESSAGES_RESPONSE = {
    "id": "msg-test",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-4-8",
    "content": [{"type": "text", "text": "OK"}],
    "usage": {"input_tokens": 50, "cache_read_input_tokens": 1800},
}


@pytest.mark.integration
@respx.mock
def test_openai_cache_hit_tokens_header_and_stats() -> None:
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_CHAT_RESPONSE)
    )
    config = Config(upstream_api_key="test-key", cache_optimization_enabled=True)
    with TestClient(create_app(config)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
        )
        assert response.status_code == 200
        assert response.headers.get("x-contextly-cache-hit-tokens") == "1500"

        stats = client.get("/stats").json()
        assert stats["cache_hit_tokens_total"] == 1500
        assert stats["cache_savings_dollars_total"] > 0


@pytest.mark.integration
@respx.mock
def test_anthropic_breakpoints_injected_into_payload() -> None:
    captured: dict = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_MESSAGES_RESPONSE)

    respx.post("https://api.anthropic.com/v1/messages").mock(side_effect=_capture)
    config = Config(
        upstream="anthropic",
        upstream_api_key="test-key",
        cache_optimization_enabled=True,
        cache_min_prefix_chars=1000,
        cache_recent_window=2,
    )
    with TestClient(create_app(config)) as client:
        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-opus-4-8",
                "system": "You are a helpful assistant.",
                "messages": [
                    {"role": "user", "content": _BIG},
                    {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": "latest"},
                    {"role": "assistant", "content": "reply"},
                ],
            },
        )
        assert response.status_code == 200
        assert int(response.headers.get("x-contextly-cache-breakpoints", "0")) >= 1

    body = captured["body"]
    # system converted to block list with a cache_control marker
    assert isinstance(body["system"], list)
    assert body["system"][-1]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.integration
@respx.mock
def test_cache_optimization_disabled_by_default() -> None:
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_CHAT_RESPONSE)
    )
    config = Config(upstream_api_key="test-key")
    with TestClient(create_app(config)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
        )
        assert response.status_code == 200
        assert "x-contextly-cache-hit-tokens" not in response.headers
