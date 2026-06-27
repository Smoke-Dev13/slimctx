"""Integration tests for multimodal image compression."""

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


def _multimodal_payload() -> dict:
    return {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this picture?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/cat.png", "detail": "high"},
                    },
                ],
            }
        ],
    }


@pytest.mark.integration
@respx.mock
def test_image_detail_downgraded_when_enabled() -> None:
    captured: dict = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_CHAT_RESPONSE)

    respx.post("https://api.openai.com/v1/chat/completions").mock(side_effect=_capture)
    config = Config(upstream_api_key="test-key", image_compression_enabled=True)
    with TestClient(create_app(config)) as client:
        resp = client.post("/v1/chat/completions", json=_multimodal_payload())
        assert resp.status_code == 200

    parts = captured["body"]["messages"][0]["content"]
    image_part = next(p for p in parts if p["type"] == "image_url")
    assert image_part["image_url"]["detail"] == "low"
    # text part still present
    assert any(p["type"] == "text" for p in parts)


@pytest.mark.integration
@respx.mock
def test_image_untouched_when_disabled() -> None:
    captured: dict = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_CHAT_RESPONSE)

    respx.post("https://api.openai.com/v1/chat/completions").mock(side_effect=_capture)
    config = Config(upstream_api_key="test-key")  # image compression off by default
    with TestClient(create_app(config)) as client:
        resp = client.post("/v1/chat/completions", json=_multimodal_payload())
        assert resp.status_code == 200

    parts = captured["body"]["messages"][0]["content"]
    image_part = next(p for p in parts if p["type"] == "image_url")
    assert image_part["image_url"]["detail"] == "high"


@pytest.mark.integration
@respx.mock
def test_image_stats_counter() -> None:
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_CHAT_RESPONSE)
    )
    config = Config(upstream_api_key="test-key", image_compression_enabled=True)
    with TestClient(create_app(config)) as client:
        client.post("/v1/chat/completions", json=_multimodal_payload())
        stats = client.get("/stats").json()
        assert stats["image_parts_compressed_total"] == 1
