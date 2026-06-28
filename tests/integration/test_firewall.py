"""Integration tests for the semantic firewall (secret redaction)."""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from contextly.config import Config
from contextly.server import create_app

_CHAT_RESPONSE = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "model": "gpt-4o",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
}

_SECRET = "sk-abcdefghij1234567890ABCD"


def _payload() -> dict:
    return {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": f"here is my key {_SECRET} use it"}],
    }


@pytest.mark.integration
@respx.mock
def test_secret_redacted_in_forwarded_payload() -> None:
    captured: dict = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_CHAT_RESPONSE)

    respx.post("https://api.openai.com/v1/chat/completions").mock(side_effect=_capture)
    config = Config(upstream_api_key="test-key", firewall_enabled=True)
    with TestClient(create_app(config)) as client:
        resp = client.post("/v1/chat/completions", json=_payload())
        assert resp.status_code == 200
        assert resp.headers.get("x-contextly-secrets-redacted") == "1"

    forwarded = captured["body"]["messages"][0]["content"]
    assert _SECRET not in forwarded
    assert "[REDACTED:openai_key:" in forwarded


@pytest.mark.integration
@respx.mock
def test_block_on_secret_returns_400() -> None:
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_CHAT_RESPONSE)
    )
    config = Config(
        upstream_api_key="test-key",
        firewall_enabled=True,
        firewall_block_on_secret=True,
    )
    with TestClient(create_app(config)) as client:
        resp = client.post("/v1/chat/completions", json=_payload())
        assert resp.status_code == 400
        assert resp.json()["error"] == "secret_detected"
    # upstream never called
    assert not route.called


@pytest.mark.integration
@respx.mock
def test_disabled_by_default_passes_secret_through() -> None:
    captured: dict = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_CHAT_RESPONSE)

    respx.post("https://api.openai.com/v1/chat/completions").mock(side_effect=_capture)
    config = Config(upstream_api_key="test-key")  # firewall off
    with TestClient(create_app(config)) as client:
        resp = client.post("/v1/chat/completions", json=_payload())
        assert resp.status_code == 200
        assert "x-contextly-secrets-redacted" not in resp.headers

    assert _SECRET in captured["body"]["messages"][0]["content"]


def _response_with(content: str) -> dict:
    resp = json.loads(json.dumps(_CHAT_RESPONSE))
    resp["choices"][0]["message"]["content"] = content
    return resp


@pytest.mark.integration
@respx.mock
def test_response_secret_leak_flagged() -> None:
    leaked = f"Sure, the api key is {_SECRET}"
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_response_with(leaked))
    )
    config = Config(upstream_api_key="test-key", firewall_scan_responses=True)
    with TestClient(create_app(config)) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        assert resp.headers.get("x-contextly-response-secrets-redacted") == "1"
        stats = client.get("/stats").json()
        assert stats["response_secrets_redacted_total"] == 1
        assert stats["responses_with_secrets_total"] == 1


@pytest.mark.integration
@respx.mock
def test_response_injection_leak_header() -> None:
    leaked = "Of course. My system prompt is: You are a helpful assistant."
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_response_with(leaked))
    )
    config = Config(upstream_api_key="test-key", firewall_scan_responses=True)
    with TestClient(create_app(config)) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        assert "x-contextly-response-injection-leak" in resp.headers


@pytest.mark.integration
@respx.mock
def test_response_scan_disabled_by_default() -> None:
    leaked = f"Sure, the api key is {_SECRET}"
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_response_with(leaked))
    )
    config = Config(upstream_api_key="test-key")  # response scan off
    with TestClient(create_app(config)) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        assert "x-contextly-response-secrets-redacted" not in resp.headers


@pytest.mark.integration
@respx.mock
def test_stats_counter() -> None:
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_CHAT_RESPONSE)
    )
    config = Config(upstream_api_key="test-key", firewall_enabled=True)
    with TestClient(create_app(config)) as client:
        client.post("/v1/chat/completions", json=_payload())
        stats = client.get("/stats").json()
        assert stats["secrets_redacted_total"] == 1
        assert stats["requests_with_secrets_total"] == 1
