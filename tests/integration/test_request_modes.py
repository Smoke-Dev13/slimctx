"""Integration tests for X-Contextly-Mode header and Anthropic compression."""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from contextly.config import Config
from contextly.server import create_app

_LONG_PROSE = (
    "Machine learning models require careful hyperparameter tuning to perform well. "
    "The learning rate is one of the most important parameters to configure properly. "
    "Batch size affects both training speed and the quality of gradient estimates. "
    "Regularization techniques such as dropout help prevent overfitting to training data. "
    "Early stopping is a simple but effective technique to improve generalization. "
) * 4

_JSON_ARRAY = json.dumps([{"id": i, "city": "Tbilisi", "plan": "gold"} for i in range(60)])

_CHAT_RESPONSE = {"id": "x", "choices": [{"message": {"role": "assistant", "content": "ok"}}]}


def _app() -> object:
    return create_app(Config(upstream_api_key="test-key", compression_enabled=True))


def _ccr_keys(resp: httpx.Response) -> dict[str, str]:
    header = resp.headers.get("x-contextly-ccr-keys", "")
    return json.loads(header) if header else {}


@pytest.mark.allow_hosts(["127.0.0.1", "::1"])
@respx.mock
def test_mode_off_disables_compression() -> None:
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_CHAT_RESPONSE)
    )
    with TestClient(_app()) as c:
        resp = c.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": _LONG_PROSE}]},
            headers={"X-Contextly-Mode": "off"},
        )
    assert resp.status_code == 200
    assert "x-contextly-ccr-keys" not in resp.headers


@pytest.mark.allow_hosts(["127.0.0.1", "::1"])
@respx.mock
def test_default_mode_compresses_prose() -> None:
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_CHAT_RESPONSE)
    )
    with TestClient(_app()) as c:
        resp = c.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": _LONG_PROSE}]},
        )
    assert "msg:0" in _ccr_keys(resp)


@pytest.mark.allow_hosts(["127.0.0.1", "::1"])
@respx.mock
def test_safe_mode_skips_prose_but_keeps_json() -> None:
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_CHAT_RESPONSE)
    )
    with TestClient(_app()) as c:
        prose = c.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": _LONG_PROSE}]},
            headers={"X-Contextly-Mode": "safe"},
        )
        js = c.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": _JSON_ARRAY}]},
            headers={"X-Contextly-Mode": "safe"},
        )
    # Prose is lossy → skipped in safe mode; JSON is lossless → still compressed.
    assert "x-contextly-ccr-keys" not in prose.headers
    assert "msg:0" in _ccr_keys(js)


@pytest.mark.allow_hosts(["127.0.0.1", "::1"])
@respx.mock
def test_anthropic_messages_now_compresses() -> None:
    respx.post("https://api.openai.com/v1/messages").mock(
        return_value=httpx.Response(200, json={"id": "msg", "content": []})
    )
    with TestClient(_app()) as c:
        resp = c.post(
            "/v1/messages",
            json={
                "model": "claude-3-5-sonnet",
                "messages": [{"role": "user", "content": _LONG_PROSE}],
            },
        )
    assert resp.status_code == 200
    assert "msg:0" in _ccr_keys(resp)


@pytest.mark.allow_hosts(["127.0.0.1", "::1"])
@respx.mock
def test_text_content_blocks_are_compressed() -> None:
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_CHAT_RESPONSE)
    )
    with TestClient(_app()) as c:
        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": [{"type": "text", "text": _LONG_PROSE}]}],
            },
        )
    assert "msg:0:0" in _ccr_keys(resp)
