"""Integration tests for OpenAI-compatible routes.

Uses respx to mock upstream calls so no real network I/O happens.

The TestClient is created INSIDE each @respx.mock scope so the httpx.AsyncClient
spawned by the app's lifespan is created while respx patching is active and thus
uses the mock transport.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from contextly.config import Config
from contextly.server import create_app

_CHAT_RESPONSE = {
    "id": "chatcmpl-test123",
    "object": "chat.completion",
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from the mock!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
}

_APP_CONFIG = Config(upstream_api_key="test-key")


def _make_client() -> TestClient:
    """Return a TestClient with respx ALREADY active (call inside @respx.mock)."""
    return TestClient(create_app(_APP_CONFIG), raise_server_exceptions=True)


@pytest.mark.integration
@respx.mock
def test_chat_completions_non_streaming() -> None:
    """Non-streaming request is forwarded and the upstream response returned."""
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_CHAT_RESPONSE)
    )
    with _make_client() as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi!"}]},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "chatcmpl-test123"
    assert data["choices"][0]["message"]["content"] == "Hello from the mock!"


@pytest.mark.integration
@respx.mock
def test_chat_completions_compressed_header_present() -> None:
    """Response carries X-Contextly-Compressed header."""
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_CHAT_RESPONSE)
    )
    with _make_client() as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi!"}]},
        )
    assert "x-contextly-compressed" in response.headers


@pytest.mark.integration
@respx.mock
def test_chat_completions_upstream_4xx_forwarded() -> None:
    """4xx from upstream is forwarded to the client unchanged."""
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            401,
            json={"error": {"message": "Invalid API key.", "type": "invalid_request_error"}},
        )
    )
    with _make_client() as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi!"}]},
        )
    assert response.status_code == 401


@pytest.mark.integration
@respx.mock
def test_compress_endpoint_passthrough() -> None:
    """/v1/compress with no registered compressors returns content unchanged."""
    with _make_client() as client:
        payload = {"content": "Hello world, this is test content.", "query": ""}
        response = client.post("/v1/compress", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["content"] == payload["content"]
    assert data["compressor"] == "passthrough"
    assert data["ratio"] == pytest.approx(1.0)
    assert data["original_length"] == len(payload["content"])


@pytest.mark.integration
@respx.mock
def test_compress_empty_content() -> None:
    with _make_client() as client:
        response = client.post("/v1/compress", json={"content": "", "query": ""})
    assert response.status_code == 200
    data = response.json()
    assert data["ratio"] == pytest.approx(1.0)
    assert data["content"] == ""


@pytest.mark.integration
@respx.mock
def test_compress_with_query_hint() -> None:
    with _make_client() as client:
        response = client.post(
            "/v1/compress",
            json={"content": "some data", "query": "count all users"},
        )
    assert response.status_code == 200
    assert response.json()["compressor"] == "passthrough"


@pytest.mark.integration
@respx.mock
def test_chat_completions_no_compress_passes_messages_verbatim() -> None:
    """compression_enabled=False must forward messages without modification."""
    captured: list[bytes] = []

    def _capture(req: httpx.Request) -> httpx.Response:
        captured.append(req.content)
        return httpx.Response(200, json=_CHAT_RESPONSE)

    respx.post("https://api.openai.com/v1/chat/completions").mock(side_effect=_capture)
    config = Config(upstream_api_key="test-key", compression_enabled=False)
    with TestClient(create_app(config), raise_server_exceptions=True) as client:
        client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "What is 2+2?"}],
            },
        )
    assert captured
    body = json.loads(captured[0])
    assert body["messages"][0]["content"] == "What is 2+2?"
