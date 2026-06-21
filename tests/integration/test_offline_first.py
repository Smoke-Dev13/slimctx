"""Offline-first proof: tokenization and full proxy flow without external network.

Two complementary strategies:
  socket_disabled  — used for pure-Python tokenizer tests (no asyncio event
                     loop needed). Blocks ALL socket creation, so any tiktoken
                     download attempt would raise immediately.

  @pytest.mark.allow_hosts(["127.0.0.1", "::1"])  — used for full-proxy tests
                     that need a TestClient (asyncio needs a loopback socketpair
                     internally on every platform). Allows only loopback; blocks
                     all external connections. Combined with @respx.mock, the
                     mocked upstream call never reaches the network layer.

Both strategies prove the same property: no code path touches the public
internet during a normal proxy request cycle.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from contextly.config import Config
from contextly.server import create_app
from contextly.tokenizer.registry import get_tokenizer
from contextly.tokenizer.tiktoken_bundled import BundledTiktokenTokenizer
from contextly.tokenizer.word_fallback import WordTokenizer

_CHAT_RESPONSE = {
    "id": "chatcmpl-offline",
    "object": "chat.completion",
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "pong"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
}

# ── Pure-Python tokenizer tests (socket_disabled = no socket creation at all) ──


def test_cl100k_tokenizer_loads_without_network(socket_disabled: None) -> None:
    """BundledTiktokenTokenizer construction must not touch the network."""
    tok = BundledTiktokenTokenizer("cl100k_base")
    assert tok.count("hello world") > 0


def test_o200k_tokenizer_loads_without_network(socket_disabled: None) -> None:
    tok = BundledTiktokenTokenizer("o200k_base")
    assert tok.count("The quick brown fox.") > 0


def test_registry_resolves_offline(socket_disabled: None) -> None:
    """get_tokenizer() never performs network I/O."""
    tok = get_tokenizer("gpt-4o")
    assert isinstance(tok, BundledTiktokenTokenizer)
    assert tok.count("contextly") > 0


def test_word_tokenizer_offline(socket_disabled: None) -> None:
    tok = WordTokenizer()
    assert tok.count("no network needed") > 0


# ── Full-proxy tests (allow_hosts = loopback OK, external blocked) ─────────────


@pytest.mark.allow_hosts(["127.0.0.1", "::1"])
@respx.mock
def test_full_proxy_request_offline() -> None:
    """Complete chat/completions proxy round-trip with external connections blocked.

    respx intercepts the httpx call before any socket connect() is made, so
    this test passes even with allow_hosts restricting external IPs.
    Proves: the server starts without downloading anything, and the request
    path does not attempt to reach the public internet.
    """
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_CHAT_RESPONSE)
    )
    config = Config(upstream_api_key="offline-test-key")
    with TestClient(create_app(config), raise_server_exceptions=True) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "ping"}],
            },
        )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "pong"


@pytest.mark.allow_hosts(["127.0.0.1", "::1"])
@respx.mock
def test_compress_endpoint_offline() -> None:
    """/v1/compress works without any external network access."""
    config = Config(upstream_api_key="test-key")
    with TestClient(create_app(config), raise_server_exceptions=True) as client:
        response = client.post(
            "/v1/compress",
            json={"content": "some text to compress", "query": ""},
        )
    assert response.status_code == 200
    assert response.json()["compressor"] == "passthrough"
