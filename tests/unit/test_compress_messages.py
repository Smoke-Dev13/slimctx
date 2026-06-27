"""Unit tests for _compress_messages and helpers in openai_compat."""

from __future__ import annotations

from contextly.audit import AuditWriter
from contextly.ccr import CCRStore
from contextly.compressors.registry import ContentRouter
from contextly.routes.openai_compat import _compress_messages, _extract_last_user_query


def _make_router() -> ContentRouter:
    return ContentRouter()  # passthrough — no compressors registered


def test_compress_messages_string_content() -> None:
    messages = [{"role": "user", "content": "Hello world"}]
    out, totals, _keys = _compress_messages(messages, "", _make_router(), CCRStore())
    assert out[0]["content"] == "Hello world"
    assert totals["original_chars"] == len("Hello world")


def test_compress_messages_empty_content() -> None:
    messages = [{"role": "user", "content": ""}]
    out, totals, _ = _compress_messages(messages, "", _make_router(), CCRStore())
    assert out[0]["content"] == ""
    assert totals["original_chars"] == 0


def test_compress_messages_non_string_content_passthrough() -> None:
    messages = [{"role": "user", "content": None}]
    out, _, _ = _compress_messages(messages, "", _make_router(), CCRStore())
    assert out[0]["content"] is None


def test_compress_messages_list_content() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "image_url", "url": "http://example.com/img.png"},
            ],
        }
    ]
    out, totals, _ = _compress_messages(messages, "", _make_router(), CCRStore())
    content = out[0]["content"]
    assert content[0]["text"] == "Hello"
    assert content[1]["url"] == "http://example.com/img.png"
    assert totals["original_chars"] == len("Hello")


def test_compress_messages_dedup_replaces_duplicate() -> None:
    big_text = "x" * 300
    messages = [
        {"role": "user", "content": big_text},
        {"role": "assistant", "content": big_text},
    ]
    out, totals, _ = _compress_messages(
        messages, "", _make_router(), CCRStore(), dedup_enabled=True, dedup_min_chars=200
    )
    assert out[0]["content"] == big_text
    sentinel = out[1]["content"]
    assert "duplicate" in sentinel
    assert "contextly_expand" in sentinel
    assert totals["dominant"] == "dedup"


def test_compress_messages_dedup_disabled() -> None:
    big_text = "y" * 300
    messages = [
        {"role": "user", "content": big_text},
        {"role": "user", "content": big_text},
    ]
    out, totals, _ = _compress_messages(
        messages, "", _make_router(), CCRStore(), dedup_enabled=False
    )
    assert out[0]["content"] == big_text
    assert out[1]["content"] == big_text
    assert totals["dominant"] == "passthrough"


def test_compress_messages_dedup_below_min_chars() -> None:
    small_text = "short"
    messages = [
        {"role": "user", "content": small_text},
        {"role": "user", "content": small_text},
    ]
    out, _, _ = _compress_messages(
        messages, "", _make_router(), CCRStore(), dedup_enabled=True, dedup_min_chars=200
    )
    assert out[0]["content"] == small_text
    assert out[1]["content"] == small_text


def test_compress_messages_audit_writer_called(tmp_path):  # type: ignore[no-untyped-def]
    log = tmp_path / "audit.jsonl"
    writer = AuditWriter(str(log))
    writer.new_request()
    messages = [{"role": "user", "content": "test content"}]
    _compress_messages(messages, "", _make_router(), CCRStore(), audit_writer=writer)
    writer.close()
    lines = log.read_text().strip().splitlines()
    assert len(lines) == 1
    import json

    rec = json.loads(lines[0])
    assert rec["msg_index"] == "msg:0"
    assert rec["compressor"] == "passthrough"


def test_compress_messages_audit_writer_dedup(tmp_path):  # type: ignore[no-untyped-def]
    log = tmp_path / "audit.jsonl"
    writer = AuditWriter(str(log))
    writer.new_request()
    big_text = "z" * 300
    messages = [
        {"role": "user", "content": big_text},
        {"role": "user", "content": big_text},
    ]
    _compress_messages(
        messages,
        "",
        _make_router(),
        CCRStore(),
        dedup_enabled=True,
        dedup_min_chars=200,
        audit_writer=writer,
    )
    writer.close()
    import json

    lines = log.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["deduped"] is True


def test_extract_last_user_query_string() -> None:
    payload = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is AI?"},
        ]
    }
    assert _extract_last_user_query(payload) == "What is AI?"


def test_extract_last_user_query_list_content() -> None:
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Describe this image"}],
            }
        ]
    }
    assert _extract_last_user_query(payload) == "Describe this image"


def test_extract_last_user_query_no_user_msg() -> None:
    payload = {"messages": [{"role": "system", "content": "sys"}]}
    assert _extract_last_user_query(payload) == ""


def test_extract_last_user_query_empty() -> None:
    assert _extract_last_user_query({}) == ""
