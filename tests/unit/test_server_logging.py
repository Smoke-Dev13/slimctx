"""Unit tests for the uvicorn access-log noise filter."""

from __future__ import annotations

import logging

from contextly.server import _QuietAccessLogFilter


def _access_record(path: str) -> logging.LogRecord:
    # Mirrors uvicorn's access log: args = (client_addr, method, path, http_ver, status).
    return logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg='%s - "%s %s HTTP/%s" %d',
        args=("127.0.0.1:1234", "GET", path, "1.1", 200),
        exc_info=None,
    )


def test_drops_dashboard_polling_paths() -> None:
    f = _QuietAccessLogFilter()
    for path in ("/stats", "/quality", "/dashboard", "/health", "/metrics"):
        assert f.filter(_access_record(path)) is False


def test_drops_path_with_query_string() -> None:
    assert _QuietAccessLogFilter().filter(_access_record("/quality?x=1")) is False


def test_keeps_real_api_traffic() -> None:
    f = _QuietAccessLogFilter()
    assert f.filter(_access_record("/v1/chat/completions")) is True
    assert f.filter(_access_record("/v1/expand/abc123")) is True


def test_keeps_record_with_unexpected_args() -> None:
    rec = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, "plain message", None, None)
    assert _QuietAccessLogFilter().filter(rec) is True
