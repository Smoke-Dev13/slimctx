"""Integration tests for the adaptive compression controller."""

from __future__ import annotations

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from contextly.config import Config
from contextly.server import create_app


def _response(completion_tokens: int) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": completion_tokens,
            "total_tokens": 100 + completion_tokens,
        },
    }


_PAYLOAD = {
    "model": "gpt-4o",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello there, tell me something."},
    ],
}


@pytest.mark.integration
@respx.mock
def test_aggression_header_present_when_enabled() -> None:
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_response(50))
    )
    config = Config(upstream_api_key="test-key", adaptive_compression_enabled=True)
    with TestClient(create_app(config)) as client:
        resp = client.post(
            "/v1/chat/completions",
            json=_PAYLOAD,
            headers={"X-Contextly-Session": "sess-1"},
        )
        assert resp.status_code == 200
        assert resp.headers.get("x-contextly-aggression") == "default"


@pytest.mark.integration
@respx.mock
def test_verbosity_spike_steps_down_across_requests() -> None:
    # First call establishes a baseline at "safe"-or-equal; second call (after the
    # controller has bumped aggression on a healthy turn) sees a verbosity spike.
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json=_response(100)),
            httpx.Response(200, json=_response(400)),  # ballooned output → paradox
        ]
    )
    config = Config(
        upstream_api_key="test-key",
        adaptive_compression_enabled=True,
        adaptive_paradox_threshold=0.25,
    )
    with TestClient(create_app(config)) as client:
        headers = {"X-Contextly-Session": "sess-spike", "X-Contextly-Mode": "aggressive-noop"}
        # Note: mode header value other than off/safe does not bypass the controller.
        r1 = client.post("/v1/chat/completions", json=_PAYLOAD, headers=headers)
        r2 = client.post("/v1/chat/completions", json=_PAYLOAD, headers=headers)
        assert r1.status_code == 200
        assert r2.status_code == 200
        # second response flags the spike
        assert r2.headers.get("x-contextly-verbosity-spike") == "true"


@pytest.mark.integration
@respx.mock
def test_disabled_by_default_no_aggression_header() -> None:
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_response(50))
    )
    config = Config(upstream_api_key="test-key")
    with TestClient(create_app(config)) as client:
        resp = client.post("/v1/chat/completions", json=_PAYLOAD)
        assert resp.status_code == 200
        assert "x-contextly-aggression" not in resp.headers


@pytest.mark.integration
@respx.mock
def test_explicit_safe_mode_bypasses_controller() -> None:
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_response(50))
    )
    config = Config(upstream_api_key="test-key", adaptive_compression_enabled=True)
    with TestClient(create_app(config)) as client:
        resp = client.post(
            "/v1/chat/completions",
            json=_PAYLOAD,
            headers={"X-Contextly-Mode": "safe"},
        )
        assert resp.status_code == 200
        assert "x-contextly-aggression" not in resp.headers
