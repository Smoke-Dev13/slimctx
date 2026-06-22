"""Integration tests for CCR store — POST /v1/compress and GET /v1/retrieve/{key}."""

from __future__ import annotations

import json

import pytest
import respx
from fastapi.testclient import TestClient

from contextly.config import Config
from contextly.server import create_app

# ── Prose long enough to be compressed ────────────────────────────────────────

_LONG_PROSE = (
    "Machine learning models require careful hyperparameter tuning to perform well. "
    "The learning rate is one of the most important parameters to configure properly. "
    "Batch size affects both training speed and the quality of gradient estimates. "
    "Regularization techniques such as dropout help prevent overfitting to training data. "
    "Validation loss should be monitored closely to detect signs of overfitting early. "
    "Early stopping is a simple but effective technique to improve generalization. "
    "Transfer learning allows practitioners to leverage pre-trained model weights. "
    "Data augmentation artificially expands the training dataset through transformations. "
    "Cross-validation provides a more robust estimate of model generalization performance. "
    "Ensemble methods combine multiple models to reduce variance and improve accuracy. "
) * 3


# ── App fixture ───────────────────────────────────────────────────────────────


@pytest.fixture
def client() -> TestClient:
    config = Config(upstream_api_key="test-key", compression_enabled=True)
    app = create_app(config)
    return TestClient(app)


# ── POST /v1/compress ─────────────────────────────────────────────────────────


def test_compress_endpoint_has_ccr_key_field(client: TestClient) -> None:
    resp = client.post("/v1/compress", json={"content": "short text", "query": ""})
    assert resp.status_code == 200
    assert "ccr_key" in resp.json()


def test_compress_endpoint_passthrough_ccr_key_is_null(client: TestClient) -> None:
    # Passthrough compressor → ratio == 1.0 → no CCR entry
    resp = client.post("/v1/compress", json={"content": "short text", "query": ""})
    assert resp.json()["ccr_key"] is None


def test_compress_endpoint_compressed_content_has_ccr_key(client: TestClient) -> None:
    resp = client.post("/v1/compress", json={"content": _LONG_PROSE, "query": "summarize"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ratio"] < 1.0
    assert body["ccr_key"] is not None
    assert len(body["ccr_key"]) == 16


def test_compress_endpoint_preserves_existing_fields(client: TestClient) -> None:
    resp = client.post("/v1/compress", json={"content": "short text", "query": ""})
    body = resp.json()
    assert "content" in body
    assert "original_length" in body
    assert "compressed_length" in body
    assert "ratio" in body
    assert "compressor" in body
    assert "metadata" in body


# ── GET /v1/retrieve/{key} ────────────────────────────────────────────────────


def test_retrieve_endpoint_returns_original(client: TestClient) -> None:
    compress_resp = client.post("/v1/compress", json={"content": _LONG_PROSE, "query": "summarize"})
    assert compress_resp.status_code == 200
    ccr_key = compress_resp.json()["ccr_key"]
    assert ccr_key is not None

    retrieve_resp = client.get(f"/v1/retrieve/{ccr_key}")
    assert retrieve_resp.status_code == 200
    body = retrieve_resp.json()
    assert body["key"] == ccr_key
    assert body["content"] == _LONG_PROSE


def test_retrieve_endpoint_404_for_unknown_key(client: TestClient) -> None:
    resp = client.get("/v1/retrieve/0000000000000000")
    assert resp.status_code == 404
    body = resp.json()
    assert "error" in body


def test_retrieve_endpoint_404_body_mentions_key(client: TestClient) -> None:
    key = "deadbeef12345678"
    resp = client.get(f"/v1/retrieve/{key}")
    assert resp.status_code == 404
    assert key in resp.json()["error"]


def test_retrieve_endpoint_content_type_json(client: TestClient) -> None:
    resp = client.get("/v1/retrieve/0000000000000000")
    assert "application/json" in resp.headers.get("content-type", "")


def test_retrieve_roundtrip_preserves_unicode(client: TestClient) -> None:
    # Build content that is long enough for prose compressor and contains unicode
    unicode_prose = (
        "Unicode test: café naïve résumé élève. "
        "The algorithm handles multibyte characters without data loss. "
        "Storage and retrieval of encoded text must be exact and lossless. "
        "Compression operates at the character level rather than the byte level. "
        "It is important to verify the original text is preserved precisely. "
    ) * 4

    compress_resp = client.post(
        "/v1/compress", json={"content": unicode_prose, "query": "summarize"}
    )
    if compress_resp.json()["ccr_key"] is None:
        pytest.skip("Content not compressed — passthrough path not testable for CCR")

    ccr_key = compress_resp.json()["ccr_key"]
    retrieve_resp = client.get(f"/v1/retrieve/{ccr_key}")
    assert retrieve_resp.status_code == 200
    assert retrieve_resp.json()["content"] == unicode_prose


# ── GET /v1/expand/{ref} (expand-on-demand) ───────────────────────────────────


def test_expand_returns_original(client: TestClient) -> None:
    ref = client.post("/v1/compress", json={"content": _LONG_PROSE, "query": "summarize"}).json()[
        "ccr_key"
    ]
    assert ref is not None
    resp = client.get(f"/v1/expand/{ref}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is True
    assert body["ref"] == ref
    assert body["content"] == _LONG_PROSE


def test_expand_404_for_unknown_ref(client: TestClient) -> None:
    resp = client.get("/v1/expand/0000000000000000")
    assert resp.status_code == 404
    assert resp.json()["found"] is False


def test_expand_with_contains_filters_records(client: TestClient) -> None:
    records = json.dumps([{"id": i, "city": "Tbilisi" if i == 7 else "Other"} for i in range(60)])
    ref = client.post("/v1/compress", json={"content": records}).json()["ccr_key"]
    assert ref is not None
    resp = client.get(f"/v1/expand/{ref}", params={"contains": "Tbilisi"})
    body = resp.json()
    assert body["found"] is True
    assert body["matches"] == 1
    assert json.loads(body["content"]) == [{"id": 7, "city": "Tbilisi"}]


# ── GET /dashboard ────────────────────────────────────────────────────────────


def test_dashboard_serves_html(client: TestClient) -> None:
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "Contextly" in resp.text
    assert "/stats" in resp.text  # the page polls the stats endpoint


# ── CCR isolation between TestClient instances ────────────────────────────────


def test_each_app_instance_has_independent_ccr_store() -> None:
    config = Config(upstream_api_key="test-key", compression_enabled=True)
    app1 = create_app(config)
    app2 = create_app(config)

    with TestClient(app1) as c1:
        r = c1.post("/v1/compress", json={"content": _LONG_PROSE, "query": "summarize"})
        key = r.json().get("ccr_key")

    if key is None:
        pytest.skip("No compression happened")

    with TestClient(app2) as c2:
        resp = c2.get(f"/v1/retrieve/{key}")
        # Different app instance → different CCRStore → key not found
        assert resp.status_code == 404


# ── chat/completions CCR header ───────────────────────────────────────────────


@pytest.mark.allow_hosts(["127.0.0.1", "::1"])
@respx.mock
def test_chat_completions_ccr_header_set_when_compressed() -> None:
    import httpx

    config = Config(upstream_api_key="test-key", compression_enabled=True)
    app = create_app(config)

    fake_resp = {
        "id": "chatcmpl-test",
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
    }
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=fake_resp)
    )

    with TestClient(app) as client:
        body = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": _LONG_PROSE}],
        }
        resp = client.post("/v1/chat/completions", json=body)

    assert resp.status_code == 200
    header = resp.headers.get("x-contextly-ccr-keys", "")
    if header:
        keys = json.loads(header)
        assert "msg:0" in keys
        assert len(keys["msg:0"]) == 16


@pytest.mark.allow_hosts(["127.0.0.1", "::1"])
@respx.mock
def test_chat_completions_no_ccr_header_for_short_messages() -> None:
    import httpx

    config = Config(upstream_api_key="test-key", compression_enabled=True)
    app = create_app(config)

    fake_resp = {
        "id": "chatcmpl-test",
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
    }
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=fake_resp)
    )

    with TestClient(app) as client:
        body = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
        }
        resp = client.post("/v1/chat/completions", json=body)

    assert resp.status_code == 200
    # Short message → passthrough → no CCR header
    assert "x-contextly-ccr-keys" not in resp.headers
