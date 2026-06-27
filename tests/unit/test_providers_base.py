"""Unit tests for BaseProvider._auth_headers."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import httpx

from contextly.providers.base import BaseProvider


class _ConcreteProvider(BaseProvider):
    @property
    def name(self) -> str:
        return "test"

    async def chat_completions(self, payload: dict[str, Any]) -> httpx.Response:  # pragma: no cover
        raise NotImplementedError


def test_auth_headers_contain_bearer_token() -> None:
    provider = _ConcreteProvider(
        base_url="https://api.example.com",
        api_key="sk-test-key",
        client=MagicMock(),
    )
    headers = provider._auth_headers()
    assert headers["Authorization"] == "Bearer sk-test-key"
    assert headers["Content-Type"] == "application/json"


def test_base_url_trailing_slash_stripped() -> None:
    provider = _ConcreteProvider(
        base_url="https://api.example.com/",
        api_key="key",
        client=MagicMock(),
    )
    assert not provider._base_url.endswith("/")
