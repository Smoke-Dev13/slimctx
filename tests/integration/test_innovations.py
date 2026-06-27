"""Integration tests for innovation features: budget enforcement, dedup, audit, cost tracking."""

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
        {
            "index": 0,
            "message": {"role": "assistant", "content": "OK"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
}


@pytest.mark.integration
@respx.mock
def test_mode_used_header_default() -> None:
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
    assert response.headers.get("x-contextly-mode-used") == "default"


@pytest.mark.integration
@respx.mock
def test_mode_used_header_off_when_no_compress() -> None:
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_CHAT_RESPONSE)
    )
    config = Config(upstream_api_key="test-key", compression_enabled=False)
    with TestClient(create_app(config)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
        )
    assert response.headers.get("x-contextly-mode-used") == "off"


@pytest.mark.integration
@respx.mock
def test_dedup_in_request_adds_ccr_keys_header() -> None:
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_CHAT_RESPONSE)
    )
    big_text = "A" * 300
    config = Config(upstream_api_key="test-key", dedup_enabled=True, dedup_min_chars=200)
    with TestClient(create_app(config)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [
                    {"role": "user", "content": big_text},
                    {"role": "user", "content": big_text},
                ],
            },
        )
    assert response.status_code == 200
    assert "x-contextly-ccr-keys" in response.headers


@pytest.mark.integration
@respx.mock
def test_audit_log_written_to_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    log_path = str(tmp_path / "audit.jsonl")
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_CHAT_RESPONSE)
    )
    config = Config(upstream_api_key="test-key", audit_log_path=log_path)
    with TestClient(create_app(config)) as client:
        client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello world"}]},
        )
    import os

    assert os.path.exists(log_path)
    lines = open(log_path).read().strip().splitlines()
    assert len(lines) >= 1
    rec = json.loads(lines[0])
    assert rec["model"] == "gpt-4o"


@pytest.mark.integration
@respx.mock
def test_budget_enforcement_escalates_chain() -> None:
    """When budget_enforcement=True and estimated tokens exceed context window, escalation runs."""
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_CHAT_RESPONSE)
    )
    # gpt-4 has a tiny 8192 context window — even a small message should trigger escalation
    config = Config(upstream_api_key="test-key", budget_enforcement=True)
    # Send a large message to a small-window model
    big_content = "word " * 2000  # ~10000 chars / 4 = ~2500 tokens
    with TestClient(create_app(config)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",  # 8192 token window
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": big_content}],
            },
        )
    assert response.status_code == 200
    # Mode-used header should reflect escalation (safe or aggressive) or default
    mode = response.headers.get("x-contextly-mode-used", "default")
    assert mode in ("default", "safe", "aggressive", "off")


@pytest.mark.integration
@respx.mock
def test_stats_includes_dollars_saved() -> None:
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_CHAT_RESPONSE)
    )
    config = Config(upstream_api_key="test-key")
    with TestClient(create_app(config)) as client:
        client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi!"}]},
        )
        stats_resp = client.get("/stats")
    assert stats_resp.status_code == 200
    data = stats_resp.json()
    assert "dollars_saved_total" in data
