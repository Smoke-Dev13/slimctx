"""Integration tests for the Anthropic-style /v1/messages route and edge cases."""

from __future__ import annotations

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from contextly.config import Config, UpstreamProvider
from contextly.server import create_app

_MESSAGES_RESPONSE = {
    "id": "msg_test",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hi from Anthropic mock!"}],
    "model": "claude-3-5-sonnet-20241022",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 8},
}


@pytest.mark.integration
@respx.mock
def test_messages_route_non_streaming() -> None:
    """POST /v1/messages is forwarded to the Anthropic upstream."""
    config = Config(upstream=UpstreamProvider.ANTHROPIC, upstream_api_key="test-key")
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json=_MESSAGES_RESPONSE)
    )
    with TestClient(create_app(config), raise_server_exceptions=True) as client:
        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hello!"}],
            },
        )
    assert response.status_code == 200
    assert response.json()["role"] == "assistant"


@pytest.mark.integration
@respx.mock
def test_messages_route_forwards_4xx() -> None:
    config = Config(upstream=UpstreamProvider.ANTHROPIC, upstream_api_key="bad")
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(401, json={"error": {"type": "auth_error"}})
    )
    with TestClient(create_app(config), raise_server_exceptions=True) as client:
        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert response.status_code == 401


@pytest.mark.integration
@respx.mock
def test_chat_completions_list_content_query_extraction() -> None:
    """Extract query from messages with list-style content (multi-modal)."""
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        )
    )
    config = Config(upstream_api_key="test-key")
    with TestClient(create_app(config), raise_server_exceptions=True) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is in this image?"},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
                        ],
                    }
                ],
            },
        )
    assert response.status_code == 200


@pytest.mark.integration
@respx.mock
def test_chat_completions_user_agent_forwarded() -> None:
    """Safe client headers such as User-Agent are forwarded to upstream."""
    captured_headers: list[httpx.Headers] = []

    def _capture(req: httpx.Request) -> httpx.Response:
        captured_headers.append(req.headers)
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        )

    respx.post("https://api.openai.com/v1/chat/completions").mock(side_effect=_capture)
    config = Config(upstream_api_key="test-key")
    with TestClient(create_app(config), raise_server_exceptions=True) as client:
        client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
            headers={"User-Agent": "test-sdk/1.0"},
        )
    assert captured_headers
    assert captured_headers[0].get("user-agent") == "test-sdk/1.0"
