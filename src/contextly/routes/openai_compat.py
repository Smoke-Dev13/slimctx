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
from contextly.ccr import CCRStore
from contextly.compressors.registry import ContentRouter
from contextly.config import Config
from contextly.ccr import content_key as _ccr_key
from contextly.config import MODEL_CONTEXT_WINDOWS
from contextly.deps import (
    ABMonitorDep,
    AggressiveContentRouterDep,
    AuditWriterDep,
    CCRDep,
    ConfigDep,
    ContentRouterDep,
    HttpClientDep,
    SafeContentRouterDep,
)
from contextly.expand import filter_original
from contextly.metrics import observe_request
from contextly.pricing import tokens_to_dollars

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


async def _compress_stream(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    ccr_store: CCRStore,
    router: ContentRouter,
    flush_sentences: int = 3,
) -> AsyncGenerator[bytes, None]:
    """SSE-aware streaming with compression of tool-call arguments.

    Parses SSE ``data:`` frames as they arrive. When a tool-call delta stream
    completes (signalled by ``finish_reason: "tool_calls"``), the accumulated
    JSON arguments are compressed and the CCR key is injected into a synthetic
    ``data: [CONTEXTLY_KEYS ...]`` comment before ``data: [DONE]``.

    For regular assistant text, a sliding-window sentence buffer flushes
    compressed output every *flush_sentences* sentences rather than buffering
    the whole response.
    """
    tool_args_buf: dict[int, str] = {}  # index → accumulated function.arguments
    tool_ccr_keys: dict[str, str] = {}  # "tool:{index}" → ccr_key
    text_buf: list[str] = []
    sentence_count = 0
    query = _extract_last_user_query(payload)

    def _flush_text() -> bytes:
        nonlocal sentence_count
        combined = "".join(text_buf)
        text_buf.clear()
        sentence_count = 0
        if not combined:
            return b""
        result = router.select(combined, query).compress(combined, query)
        return result.content.encode()

    async with client.stream("POST", url, headers=headers, json=payload) as response:
        async for raw_chunk in response.aiter_bytes():
            for line in raw_chunk.split(b"\n"):
                if not line.startswith(b"data: "):
                    yield line + b"\n"
                    continue
                data_str = line[6:].decode("utf-8", errors="replace").strip()
                if data_str == "[DONE]":
                    # Flush any remaining text buffer
                    leftover = _flush_text()
                    if leftover:
                        yield b"data: " + leftover + b"\n\n"
                    # Compress accumulated tool-call arguments
                    for idx, args_str in tool_args_buf.items():
                        if not args_str:
                            continue
                        result = router.select(args_str, query).compress(args_str, query)
                        if result.compression_ratio < 1.0:
                            key = ccr_store.store(args_str)
                            tool_ccr_keys[f"tool:{idx}"] = key
                    if tool_ccr_keys:
                        keys_json = json.dumps(tool_ccr_keys)
                        yield f"data: [CONTEXTLY_KEYS {keys_json}]\n\n".encode()
                    yield b"data: [DONE]\n\n"
                    return
                try:
                    chunk_obj: dict[str, Any] = json.loads(data_str)
                except (json.JSONDecodeError, ValueError):
                    yield line + b"\n"
                    continue

                # Track tool-call argument deltas
                for choice in chunk_obj.get("choices", []):
                    delta = choice.get("delta", {})
                    for tc in delta.get("tool_calls", []):
                        idx = tc.get("index", 0)
                        fn_args = tc.get("function", {}).get("arguments", "")
                        if fn_args:
                            tool_args_buf[idx] = tool_args_buf.get(idx, "") + fn_args

                    # Track assistant text for sentence-window flushing
                    content_delta = delta.get("content") or ""
                    if content_delta:
                        text_buf.append(content_delta)
                        sentence_count += content_delta.count(".")
                        if sentence_count >= flush_sentences:
                            compressed = _flush_text()
                            if compressed:
                                synthetic = dict(chunk_obj)
                                synthetic["choices"] = [
                                    {**c, "delta": {"content": compressed.decode()}}
                                    for c in chunk_obj.get("choices", [])
                                ]
                                yield b"data: " + json.dumps(synthetic).encode() + b"\n\n"
                            continue

                yield b"data: " + json.dumps(chunk_obj).encode() + b"\n\n"


def _select_router(
    request: Request,
    config: Config,
    default_router: ContentRouter,
    safe_router: ContentRouter,
) -> ContentRouter | None:
    """Pick the compressor chain for a request, honouring X-Contextly-Mode.

    Header values: ``off`` disables compression for the call, ``safe`` forces the
    lossless chain, anything else (or absent) uses the configured default. Returns
    None when no compression should run.
    """
    if not config.compression_enabled:
        return None
    mode = request.headers.get("X-Contextly-Mode", "").strip().lower()
    if mode == "off":
        return None
    if mode == "safe":
        return safe_router
    return default_router


def _compress_messages(
    messages: list[dict[str, Any]],
    query: str,
    router: ContentRouter,
    ccr_store: CCRStore,
    *,
    dedup_enabled: bool = True,
    dedup_min_chars: int = 200,
    model: str = "unknown",
    audit_writer: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    """Compress each message's text, returning new messages, totals, and CCR keys.

    Handles both string content and OpenAI/Anthropic content-block lists (only
    ``{"type": "text", "text": ...}`` parts are compressed). Originals are stored
    in the CCR store whenever compression reduced the text.

    Cross-message deduplication: when dedup_enabled is True and a content block
    is identical to one seen earlier in the same request (by SHA key), it is
    replaced with a sentinel referencing the first occurrence's CCR key instead
    of being compressed again — saving all of its tokens at zero quality risk.
    """
    out: list[dict[str, Any]] = []
    orig = comp = saved = 0
    dominant = "passthrough"
    ccr_keys: dict[str, str] = {}
    seen_hashes: dict[str, str] = {}  # sha_key → slot label of first occurrence

    def _do(text: str, slot: str) -> str:
        nonlocal orig, comp, saved, dominant

        # Cross-message deduplication — exact duplicates need zero inference tokens
        if dedup_enabled and len(text) >= dedup_min_chars:
            sha = _ccr_key(text)
            if sha in seen_hashes:
                first_slot = seen_hashes[sha]
                stored_key = ccr_store.store(text)
                ccr_keys[slot] = stored_key
                sentinel = (
                    f"[duplicate of {first_slot} — identical content omitted to save tokens; "
                    f'contextly_expand("{stored_key}") to recover]'
                )
                orig += len(text)
                comp += len(sentinel)
                saved += max(0, (len(text) - len(sentinel)) // 4)
                dominant = "dedup"
                if audit_writer is not None:
                    audit_writer.record(
                        model=model,
                        msg_index=slot,
                        ccr_key=stored_key,
                        compressor="dedup",
                        original_chars=len(text),
                        compressed_chars=len(sentinel),
                        deduped=True,
                    )
                return sentinel
            seen_hashes[sha] = slot

        result = router.select(text, query).compress(text, query)
        orig += result.original_length
        comp += result.compressed_length
        saved += result.tokens_saved_estimate
        stored_key: str | None = None
        if result.compression_ratio < 1.0:
            dominant = result.compressor_name
            stored_key = ccr_store.store(text)
            ccr_keys[slot] = stored_key
        if audit_writer is not None:
            audit_writer.record(
                model=model,
                msg_index=slot,
                ccr_key=stored_key,
                compressor=result.compressor_name,
                original_chars=result.original_length,
                compressed_chars=result.compressed_length,
            )
        return result.content

    for i, msg in enumerate(messages):
        content = msg.get("content", "")
        if isinstance(content, str) and content:
            out.append({**msg, "content": _do(content, f"msg:{i}")})
        elif isinstance(content, list):
            parts: list[Any] = []
            for j, part in enumerate(content):
                if (
                    isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                    and part["text"]
                ):
                    parts.append({**part, "text": _do(part["text"], f"msg:{i}:{j}")})
                else:
                    parts.append(part)
            out.append({**msg, "content": parts})
        else:
            out.append(msg)

    totals = {
        "original_chars": orig,
        "compressed_chars": comp,
        "tokens_saved": saved,
        "dominant": dominant,
    }
    return out, totals, ccr_keys


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
    safe_content_router: SafeContentRouterDep,
    aggressive_content_router: AggressiveContentRouterDep,
    ccr_store: CCRDep,
    ab_monitor: ABMonitorDep,
    audit_writer: AuditWriterDep,
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

    active_router = _select_router(request, config, content_router, safe_content_router)
    mode_used = "off" if active_router is None else "default"

    if audit_writer is not None:
        audit_writer.new_request()

    if active_router is not None:
        query = _extract_last_user_query(payload)
        messages = payload.get("messages", [])
        compressed_messages, totals, ccr_keys = _compress_messages(
            messages,
            query,
            active_router,
            ccr_store,
            dedup_enabled=config.dedup_enabled,
            dedup_min_chars=config.dedup_min_chars,
            model=model,
            audit_writer=audit_writer,
        )
        total_original_chars = totals["original_chars"]
        total_compressed_chars = totals["compressed_chars"]
        total_tokens_saved_estimate = totals["tokens_saved"]
        dominant_compressor = totals["dominant"]
        payload = {**payload, "messages": compressed_messages}

        # Budget enforcement: escalate compressor chain if estimated tokens
        # would overflow the model's context window.
        if config.budget_enforcement:
            context_window = MODEL_CONTEXT_WINDOWS.get(model, 128_000)
            max_output = int(payload.get("max_tokens") or 4096)
            safety_margin = 512
            estimated_input = total_compressed_chars // 4
            if estimated_input + max_output + safety_margin > context_window:
                # Try safe router first, then aggressive
                for escalation_router, escalation_name in (
                    (safe_content_router, "safe"),
                    (aggressive_content_router, "aggressive"),
                ):
                    esc_msgs, esc_totals, esc_keys = _compress_messages(
                        messages, query, escalation_router, ccr_store,
                        dedup_enabled=config.dedup_enabled,
                        dedup_min_chars=config.dedup_min_chars,
                        model=model,
                    )
                    est = esc_totals["compressed_chars"] // 4
                    if est + max_output + safety_margin <= context_window:
                        compressed_messages, totals, ccr_keys = esc_msgs, esc_totals, esc_keys
                        total_compressed_chars = esc_totals["compressed_chars"]
                        total_tokens_saved_estimate = esc_totals["tokens_saved"]
                        dominant_compressor = esc_totals["dominant"]
                        payload = {**payload, "messages": compressed_messages}
                        mode_used = escalation_name
                        log.info(
                            "budget_enforcement_escalated",
                            chain=escalation_name,
                            estimated_tokens=est,
                            context_window=context_window,
                        )
                        break
                else:
                    log.warning(
                        "budget_enforcement_overflow",
                        estimated_tokens=total_compressed_chars // 4,
                        context_window=context_window,
                    )

    dollars_saved = tokens_to_dollars(
        total_tokens_saved_estimate, model, config.pricing_overrides or {}
    )

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
            dollars_saved=dollars_saved,
        )

    upstream_url = f"{config.resolved_upstream_url()}/v1/chat/completions"
    headers = _build_upstream_headers(request, config.upstream_api_key)
    extra_headers: dict[str, str] = {
        "X-Contextly-Compressed": str(config.compression_enabled).lower(),
        "X-Contextly-Mode-Used": mode_used,
    }
    if ccr_keys:
        extra_headers["X-Contextly-CCR-Keys"] = json.dumps(ccr_keys)

    if is_streaming:
        # A/B monitoring skipped for streaming — buffering the response would
        # defeat the purpose of streaming.
        if config.stream_compression_enabled and active_router is not None:
            stream_gen = _compress_stream(
                http_client,
                upstream_url,
                headers,
                payload,
                ccr_store,
                active_router,
                flush_sentences=config.stream_flush_sentences,
            )
        else:
            stream_gen = _proxy_stream(http_client, upstream_url, headers, payload)
        return StreamingResponse(
            stream_gen,
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
        dollars_saved=dollars_saved,
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
    content_router: ContentRouterDep,
    safe_content_router: SafeContentRouterDep,
    ccr_store: CCRDep,
    ab_monitor: ABMonitorDep,
) -> Response:
    """Proxy Anthropic-style /v1/messages with compression.

    Compresses each message's text (string or text content blocks) the same way
    as /v1/chat/completions, honouring the X-Contextly-Mode header. Originals go
    to the CCR store; keys appear in X-Contextly-CCR-Keys.

    Args:
        request: Incoming FastAPI request.
        config: Resolved runtime configuration.
        http_client: Shared async HTTP client.
        content_router: Default compressor chain.
        safe_content_router: Lossless compressor chain (for mode=safe).
        ccr_store: CCR reversible store.
        ab_monitor: A/B quality monitor (running counters).

    Returns:
        Proxied upstream response.
    """
    raw_body = await request.body()
    payload: dict[str, Any] = json.loads(raw_body)
    is_streaming: bool = bool(payload.get("stream", False))
    ccr_keys: dict[str, str] = {}

    router = _select_router(request, config, content_router, safe_content_router)
    if router is not None:
        query = _extract_last_user_query(payload)
        compressed_messages, totals, ccr_keys = _compress_messages(
            payload.get("messages", []), query, router, ccr_store
        )
        payload = {**payload, "messages": compressed_messages}
        if totals["original_chars"] > 0:
            ab_monitor.record_request(
                original_chars=totals["original_chars"],
                compressed_chars=totals["compressed_chars"],
                compressor_name=totals["dominant"],
                tokens_saved_estimate=totals["tokens_saved"],
            )

    upstream_url = f"{config.resolved_upstream_url()}/v1/messages"
    headers = _build_upstream_headers(request, config.upstream_api_key)
    extra_headers: dict[str, str] = {}
    if ccr_keys:
        extra_headers["X-Contextly-CCR-Keys"] = json.dumps(ccr_keys)

    if is_streaming:
        return StreamingResponse(
            _proxy_stream(http_client, upstream_url, headers, payload),
            media_type="text/event-stream",
            headers=extra_headers,
        )

    upstream_resp = await http_client.post(upstream_url, headers=headers, json=payload)
    return Response(
        content=upstream_resp.content,
        status_code=upstream_resp.status_code,
        media_type="application/json",
        headers=extra_headers,
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


@router.get("/expand/{ref}")
async def expand_endpoint(ref: str, ccr_store: CCRDep, contains: str = "") -> Response:
    """Expand a compressed result back to its original — optionally just a slice.

    The expand-on-demand counterpart to lossy compression: when a message was
    compressed with loss, its ``ccr_key`` (returned in the response body and the
    ``X-Contextly-CCR-Keys`` header) can be expanded here to recover what was
    dropped — so aggressive compression never permanently loses data.

    Pass ``?contains=<substr>`` to pull back only the matching records (for JSON)
    or lines (for logs/text) instead of the whole original — granular recovery so
    the agent spends tokens only on the detail it needs.

    Args:
        ref: The expand reference (a CCR key).
        ccr_store: CCR reversible store.
        contains: Optional case-insensitive substring filter.

    Returns:
        {"ref", "found": true, "content", "matches": N} on success
        (``matches`` is -1 when no filter was applied).
        {"ref", "found": false, "error"} with status 404 otherwise.
    """
    original = ccr_store.retrieve(ref)
    if original is None:
        return Response(
            content=json.dumps(
                {"ref": ref, "found": False, "error": f"Reference '{ref}' not found or evicted."}
            ),
            status_code=404,
            media_type="application/json",
        )
    content, matches = filter_original(original, contains)
    return Response(
        content=json.dumps({"ref": ref, "found": True, "content": content, "matches": matches}),
        status_code=200,
        media_type="application/json",
    )
