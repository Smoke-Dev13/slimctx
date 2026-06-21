"""Shared pytest fixtures for unit and integration tests."""

from __future__ import annotations

import asyncio
import sys

import pytest
from fastapi.testclient import TestClient

from contextly.config import Config
from contextly.server import create_app

# On Windows, ProactorEventLoop uses socketpair() internally for its self-pipe,
# which conflicts with pytest-socket's socket_disabled fixture. Switch to
# SelectorEventLoop (same as Linux/macOS) so socket_disabled works everywhere.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@pytest.fixture
def config() -> Config:
    """Minimal Config with a fake API key. No real upstream calls expected."""
    return Config(upstream_api_key="test-key-contextly")


@pytest.fixture
def app(config: Config) -> object:
    """Configured FastAPI app instance."""
    return create_app(config)


@pytest.fixture
def client(app: object) -> TestClient:
    """Synchronous TestClient for endpoints that don't need upstream mocking."""
    with TestClient(app, raise_server_exceptions=True) as c:  # type: ignore[arg-type]
        yield c  # type: ignore[misc]
