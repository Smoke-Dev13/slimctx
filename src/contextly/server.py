"""FastAPI application factory and uvicorn entrypoint.

`create_app` wires together routes, shared state, and the async lifespan.
`run` is called by the CLI `proxy` command.

Design notes:
- The httpx.AsyncClient is created once per process in the lifespan and stored
  in app.state so route handlers can share connection pools.
- ContentRouter is also created once and stored in app.state; it is stateless
  after startup so this is safe.
- structlog is configured in the lifespan so log format respects the log_level
  flag passed via CLI.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx
import structlog
import uvicorn
from fastapi import FastAPI

from contextly.ab_monitor import ABMonitor
from contextly.ccr import CCRStore
from contextly.compressors.code import CodeCompressor
from contextly.compressors.json_smart import JsonSmartCompressor
from contextly.compressors.prose import ProseCompressor
from contextly.compressors.registry import ContentRouter
from contextly.config import Config
from contextly.routes.observability import router as obs_router
from contextly.routes.openai_compat import router as openai_router

logger = structlog.get_logger(__name__)


def _configure_logging(log_level: str) -> None:
    """Configure structlog with console rendering at the requested level."""
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(level=log_level.upper())


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage shared resources across the process lifetime."""
    config: Config = app.state.config
    _configure_logging(config.log_level)
    logger.info(
        "contextly_starting",
        host=config.host,
        port=config.port,
        upstream=str(config.upstream),
        upstream_url=config.resolved_upstream_url(),
        compression=config.compression_enabled,
    )
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0),
        follow_redirects=True,
    ) as client:
        app.state.http_client = client
        yield
    logger.info("contextly_stopped")


def create_app(config: Config) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Validated runtime configuration.

    Returns:
        A ready-to-serve FastAPI instance.
    """
    app = FastAPI(
        title="Contextly",
        description="Smart context optimization proxy for LLM APIs",
        version="0.1.0",
        docs_url="/docs",
        redoc_url=None,
        lifespan=_lifespan,
    )
    app.state.config = config
    content_router = ContentRouter()
    if config.compression_enabled:
        content_router.register(JsonSmartCompressor())
        content_router.register(CodeCompressor())
        content_router.register(ProseCompressor())
    app.state.content_router = content_router
    app.state.ccr_store = CCRStore()
    app.state.ab_monitor = ABMonitor()
    app.include_router(obs_router)
    app.include_router(openai_router)
    return app


def run(config: Config) -> None:
    """Start the uvicorn server with the provided configuration.

    Args:
        config: Validated runtime configuration.
    """
    app = create_app(config)
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
    )
