"""OpenAI-compatible API endpoints.

POST /v1/chat/completions  — proxied with optional compression, streaming OK
POST /v1/messages          — Anthropic-style messages pass-through
POST /v1/compress          — explicit compression; stores original in CCR store
GET  /v1/retrieve/{key}    — retrieve original content by CCR key
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx
import structlog
from fastapi import APIRouter, Request
from fastapi.responses import Response, StreamingResponse

from contextly.ab_monitor import _extract_response_text, run_shadow_ab
from contextly.deps import ABMonitorDep, CCRDep, ConfigDep, ContentRouterDep, HttpClientDep
from contextly.metrics import observe_request

# Keeps strong references to background tasks so they aren't GC'd before completion.
_background_tasks: set[asyncio.Task[None]] = set()

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/v1", tags=["openai-compat"])


def _build_upstream_headers(request: Request, api_key: str) -> dict[str, str]:
    """Build headers for the upstream request, forwarding safe client headers."""
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    for name in ("User-Agent", "X-Request-Id", "X-Stainless-OS"):
        if value := request.headers.get(name):
            headers[name] = value
    return headers


async def _proxy_stream(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> AsyncGenerator[bytes, None]:
    """Yield raw SSE bytes from an upstream streaming response."""
    async with client.stream("POST", url, headers=headers, json=payload) as response:
        async for chunk in response.aiter_bytes():
            yield chunk


def _extract_last_user_query(payload: dict[str, Any]) -> str:
    """Extract the last user-role message text for query-aware compression."""
    for msg in reversed(payload.get("messages", [])):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return str(part.get("text", ""))
    return ""


@router.post("/chat/completions")
async def chat_completions(
    request: Request,
    config: ConfigDep,
    http_client: HttpClientDep,
    content_router: ContentRouterDep,
    ccr_store: CCRDep,
    ab_monitor: ABMonitorDep,
) -> Response:
    """Proxy /v1/chat/completions to the configured upstream with compression.

    Applies the active compressor pipeline to each message's content before
    forwarding. Compressed originals are stored in the CCR store; their keys
    appear in X-Contextly-CCR-Keys (JSON object mapping "msg:{index}" → key).

    When ab_sample_rate > 0 and the request is non-streaming, a shadow A/B
    comparison is fired as a background asyncio task: the original (uncompressed)
    context is sent to the same upstream and the response is quality-scored
    against the compressed-context response.

    Args:
        request: Incoming FastAPI request.
        config: Resolved runtime configuration.
        http_client: Shared async HTTP client.
        content_router: Compressor selection router.
        ccr_store: CCR reversible store.
        ab_monitor: A/B quality monitor.

    Returns:
        Proxied upstream response (StreamingResponse or JSON Response).
    """
    t0 = time.monotonic()
    raw_body = await request.body()
    original_payload: dict[str, Any] = json.loads(raw_body)
    payload = original_payload
    is_streaming: bool = bool(payload.get("stream", False))
    model: str = str(payload.get("model", "unknown"))

    log = logger.bind(model=model, stream=is_streaming, n_messages=len(payload.get("messages", [])))
    log.info("request_received")

    total_original_chars: int = 0
    total_compressed_chars: int = 0
    total_tokens_saved_estimate: int = 0
    dominant_compressor: str = "passthrough"
    ccr_keys: dict[str, str] = {}

    if config.compression_enabled:
        query = _extract_last_user_query(payload)
        compressed_messages: list[dict[str, Any]] = []
        for i, msg in enumerate(payload.get("messages", [])):
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                compressor = content_router.select(content, query)
                result = compressor.compress(content, query)
                total_original_chars += result.original_length
                total_compressed_chars += result.compressed_length
                total_tokens_saved_estimate += result.tokens_saved_estimate
                if result.compression_ratio < 1.0:
                    dominant_compressor = result.compressor_name
                    ccr_keys[f"msg:{i}"] = ccr_store.store(content)
                compressed_messages.append({**msg, "content": result.content})
            else:
                compressed_messages.append(msg)
        payload = {**payload, "messages": compressed_messages}

    if total_original_chars > 0:
        if config.target_token_budget is not None:
            estimated_tokens = total_compressed_chars // 4
            if estimated_tokens > config.target_token_budget:
                log.warning(
                    "token_budget_exceeded",
                    estimated_tokens=estimated_tokens,
                    budget=config.target_token_budget,
                )
        ab_monitor.record_request(
            original_chars=total_original_chars,
            compressed_chars=total_compressed_chars,
            compressor_name=dominant_compressor,
            tokens_saved_estimate=total_tokens_saved_estimate,
        )

    upstream_url = f"{config.resolved_upstream_url()}/v1/chat/completions"
    headers = _build_upstream_headers(request, config.upstream_api_key)
    extra_headers: dict[str, str] = {
        "X-Contextly-Compressed": str(config.compression_enabled).lower(),
    }
    if ccr_keys:
        extra_headers["X-Contextly-CCR-Keys"] = json.dumps(ccr_keys)

    if is_streaming:
        # A/B monitoring skipped for streaming — buffering the response would
        # defeat the purpose of streaming.
        return StreamingResponse(
            _proxy_stream(http_client, upstream_url, headers, payload),
            media_type="text/event-stream",
            headers=extra_headers,
        )

    upstream_resp = await http_client.post(upstream_url, headers=headers, json=payload)
    latency = time.monotonic() - t0
    log.info("upstream_response", status=upstream_resp.status_code, latency=round(latency, 3))

    observe_request(
        model=model,
        compressor=dominant_compressor,
        original_chars=total_original_chars,
        compressed_chars=total_compressed_chars,
        latency_seconds=latency,
        tokens_saved_estimate=total_tokens_saved_estimate,
    )

    chars_saved = total_original_chars - total_compressed_chars
    if (
        chars_saved > 0 and config.ab_sample_rate > 0.0 and random.random() < config.ab_sample_rate  # noqa: S311
    ):
        compressed_response_text = _extract_response_text(upstream_resp.content)
        shadow_task = asyncio.create_task(
            run_shadow_ab(
                http_client=http_client,
                upstream_url=upstream_url,
                headers=headers,
                original_payload=original_payload,
                compressed_response_text=compressed_response_text,
                model=model,
                compressor_name=dominant_compressor,
                original_chars=total_original_chars,
                compressed_chars=total_compressed_chars,
                ab_monitor=ab_monitor,
            )
        )
        _background_tasks.add(shadow_task)
        shadow_task.add_done_callback(_background_tasks.discard)

    return Response(
        content=upstream_resp.content,
        status_code=upstream_resp.status_code,
        media_type="application/json",
        headers=extra_headers,
    )


@router.post("/messages")
async def anthropic_messages(
    request: Request,
    config: ConfigDep,
    http_client: HttpClientDep,
) -> Response:
    """Pass-through for Anthropic-style /v1/messages requests.

    Args:
        request: Incoming FastAPI request.
        config: Resolved runtime configuration.
        http_client: Shared async HTTP client.

    Returns:
        Proxied upstream response.
    """
    raw_body = await request.body()
    payload: dict[str, Any] = json.loads(raw_body)
    is_streaming: bool = bool(payload.get("stream", False))

    upstream_url = f"{config.resolved_upstream_url()}/v1/messages"
    headers = _build_upstream_headers(request, config.upstream_api_key)

    if is_streaming:
        return StreamingResponse(
            _proxy_stream(http_client, upstream_url, headers, payload),
            media_type="text/event-stream",
        )

    upstream_resp = await http_client.post(upstream_url, headers=headers, json=payload)
    return Response(
        content=upstream_resp.content,
        status_code=upstream_resp.status_code,
        media_type="application/json",
    )


@router.post("/compress")
async def compress_endpoint(
    request: Request,
    content_router: ContentRouterDep,
    ccr_store: CCRDep,
) -> Response:
    """Explicitly compress content without forwarding to an LLM.

    Request body: {"content": "...", "query": "..."}
    Response:     {"content": "...", "original_length": N, "compressed_length": M,
                   "ratio": 0.x, "compressor": "...", "metadata": {...},
                   "ccr_key": "<key or null>"}

    When compression actually reduces the content, the original is persisted in
    the CCR store and its 16-character key is returned as ``ccr_key``.  Use
    GET /v1/retrieve/{key} to recover the original later.

    Args:
        request: Incoming FastAPI request.
        content_router: Compressor selection router.
        ccr_store: CCR reversible store.

    Returns:
        Compression result as JSON.
    """
    raw_body = await request.body()
    payload: dict[str, Any] = json.loads(raw_body)
    content: str = payload.get("content", "")
    query: str = payload.get("query", "")

    compressor = content_router.select(content, query)
    result = compressor.compress(content, query)

    ccr_key: str | None = None
    if result.compression_ratio < 1.0:
        ccr_key = ccr_store.store(content)

    return Response(
        content=json.dumps(
            {
                "content": result.content,
                "original_length": result.original_length,
                "compressed_length": result.compressed_length,
                "ratio": round(result.compression_ratio, 4),
                "compressor": result.compressor_name,
                "metadata": result.metadata,
                "ccr_key": ccr_key,
            }
        ),
        status_code=200,
        media_type="application/json",
    )


@router.get("/retrieve/{key}")
async def retrieve_endpoint(key: str, ccr_store: CCRDep) -> Response:
    """Retrieve the original content stored under a CCR key.

    Args:
        key: The 16-character hex key returned by POST /v1/compress or the
             X-Contextly-CCR-Keys response header.
        ccr_store: CCR reversible store.

    Returns:
        {"key": "...", "content": "..."} on success.
        {"error": "..."} with status 404 if the key is absent or evicted.
    """
    original = ccr_store.retrieve(key)
    if original is None:
        return Response(
            content=json.dumps({"error": f"Key '{key}' not found. It may have been evicted."}),
            status_code=404,
            media_type="application/json",
        )
    return Response(
        content=json.dumps({"key": key, "content": original}),
        status_code=200,
        media_type="application/json",
    )
