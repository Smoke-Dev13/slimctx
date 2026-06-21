"""Abstract base for upstream LLM provider clients.

Each provider (OpenAI, Anthropic, OpenRouter) implements this interface so the
request path can forward calls without knowing which provider is active.
Concrete implementations arrive in M2 (openai.py, anthropic.py, openrouter.py).

For M1, upstream forwarding is handled directly in the route handlers to keep
the dependency graph flat; providers are integrated in M2.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import httpx


class BaseProvider(ABC):
    """Abstract upstream LLM provider."""

    def __init__(self, base_url: str, api_key: str, client: httpx.AsyncClient) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._client = client

    @property
    @abstractmethod
    def name(self) -> str:  # pragma: no cover
        """Provider identifier, e.g. 'openai' or 'anthropic'."""
        ...

    @abstractmethod
    async def chat_completions(self, payload: dict[str, Any]) -> httpx.Response:  # pragma: no cover
        """Forward a chat completions request to the upstream provider.

        Args:
            payload: The JSON body as parsed from the client request.

        Returns:
            The raw httpx.Response from upstream.
        """
        ...

    def _auth_headers(self) -> dict[str, str]:
        """Build the minimal authorization headers required by this provider."""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
