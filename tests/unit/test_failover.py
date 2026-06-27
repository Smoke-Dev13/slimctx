"""Unit tests for FailoverRouter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from contextly.failover import FailoverRouter, FailoverTarget


def _target(url: str = "https://primary.example.com", provider: str = "primary") -> FailoverTarget:
    return FailoverTarget(url=url, api_key="sk-test", provider=provider)


def _make_response(status: int, body: bytes = b"{}") -> httpx.Response:
    return httpx.Response(status, content=body)


def _router(targets: list[FailoverTarget] | None = None, max_retries: int = 2) -> FailoverRouter:
    if targets is None:
        targets = [_target()]
    return FailoverRouter(targets=targets, max_retries=max_retries)


@pytest.mark.asyncio
async def test_success_on_first_attempt() -> None:
    router = _router()
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = _make_response(200)

    resp, provider = await router.attempt(mock_client, "/v1/chat/completions", {}, {})
    assert resp.status_code == 200
    assert provider == "primary"
    assert mock_client.post.call_count == 1


@pytest.mark.asyncio
async def test_falls_back_on_429() -> None:
    primary = _target("https://primary.example.com", "primary")
    fallback = _target("https://fallback.example.com", "fallback")
    router = FailoverRouter(targets=[primary, fallback], max_retries=2)

    responses = [_make_response(429), _make_response(200)]
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.side_effect = responses

    with patch("contextly.failover.asyncio.sleep", new_callable=AsyncMock):
        resp, provider = await router.attempt(mock_client, "/v1/chat/completions", {}, {})

    assert resp.status_code == 200
    assert provider == "fallback"


@pytest.mark.asyncio
async def test_falls_back_on_connect_error() -> None:
    primary = _target("https://primary.example.com", "primary")
    fallback = _target("https://fallback.example.com", "fallback")
    router = FailoverRouter(targets=[primary, fallback], max_retries=2)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.side_effect = [
        httpx.ConnectError("connection refused"),
        _make_response(200),
    ]

    with patch("contextly.failover.asyncio.sleep", new_callable=AsyncMock):
        resp, provider = await router.attempt(mock_client, "/v1/chat/completions", {}, {})

    assert resp.status_code == 200
    assert provider == "fallback"


@pytest.mark.asyncio
async def test_non_retryable_4xx_returned_immediately() -> None:
    router = _router()
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = _make_response(400)

    resp, _provider = await router.attempt(mock_client, "/v1/chat/completions", {}, {})
    assert resp.status_code == 400
    assert mock_client.post.call_count == 1


@pytest.mark.asyncio
async def test_all_fail_returns_last_response() -> None:
    router = FailoverRouter(targets=[_target()], max_retries=2)
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = _make_response(503)

    with patch("contextly.failover.asyncio.sleep", new_callable=AsyncMock):
        resp, _ = await router.attempt(mock_client, "/v1/chat/completions", {}, {})

    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_api_key_replaced_per_target() -> None:
    primary = FailoverTarget(url="https://p.example.com", api_key="key-primary", provider="p")
    fallback = FailoverTarget(url="https://f.example.com", api_key="key-fallback", provider="f")
    router = FailoverRouter(targets=[primary, fallback], max_retries=2)

    call_headers: list[dict] = []

    async def fake_post(url: str, *, headers: dict, json: dict) -> httpx.Response:
        call_headers.append(dict(headers))
        if "p.example.com" in url:
            raise httpx.ConnectError("fail")
        return _make_response(200)

    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_client.post = fake_post

    with patch("contextly.failover.asyncio.sleep", new_callable=AsyncMock):
        await router.attempt(mock_client, "/v1/chat/completions", {}, {})

    assert call_headers[0]["Authorization"] == "Bearer key-primary"
    assert call_headers[1]["Authorization"] == "Bearer key-fallback"


def test_stats_initial() -> None:
    router = FailoverRouter(
        targets=[_target("https://a.example.com", "a"), _target("https://b.example.com", "b")],
        max_retries=3,
    )
    st = router.stats()
    assert st["failover_events_total"] == 0
    assert st["failover_success_by_target"] == {"a": 0, "b": 0}


def test_has_fallbacks_single_target() -> None:
    assert not FailoverRouter(targets=[_target()], max_retries=2).has_fallbacks


def test_has_fallbacks_multiple_targets() -> None:
    targets = [_target("https://a.com", "a"), _target("https://b.com", "b")]
    assert FailoverRouter(targets=targets, max_retries=2).has_fallbacks
