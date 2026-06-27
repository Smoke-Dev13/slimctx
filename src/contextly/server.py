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
from contextly.audit import AuditWriter
from contextly.cache_opt import CacheOptimizer
from contextly.ccr import CCRStore, SQLiteCCRStore
from contextly.compressors.code import CodeCompressor
from contextly.compressors.json_smart import JsonSmartCompressor
from contextly.compressors.json_table import JsonTableCompressor
from contextly.compressors.logs import LogCompressor
from contextly.compressors.prose import ProseCompressor
from contextly.compressors.registry import ContentRouter
from contextly.config import Config
from contextly.controller import AdaptiveController
from contextly.failover import FailoverRouter, FailoverTarget
from contextly.gateway_stats import SQLiteStatsStore, default_stats_path
from contextly.injection import InjectionScanner
from contextly.routes.observability import router as obs_router
from contextly.routes.openai_compat import router as openai_router
from contextly.scorer import MessageScorer

logger = structlog.get_logger(__name__)

# Observability/probe endpoints the dashboard polls; their uvicorn access logs are
# pure noise, so they are filtered out (real API traffic stays logged).
_QUIET_ACCESS_PATHS: frozenset[str] = frozenset(
    {
        "/",
        "/favicon.ico",
        "/stats",
        "/quality",
        "/gateway-stats",
        "/dashboard",
        "/health",
        "/metrics",
    }
)


class _QuietAccessLogFilter(logging.Filter):
    """Drop uvicorn access-log records for the dashboard's polling endpoints."""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and len(args) >= 3 and isinstance(args[2], str):
            path = args[2].split("?", 1)[0]
            if path in _QUIET_ACCESS_PATHS:
                return False
        return True


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
    # uvicorn configures its access logger before the lifespan runs; attaching the
    # filter here (idempotently) keeps the dashboard's 2s polling out of the logs.
    access_logger = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, _QuietAccessLogFilter) for f in access_logger.filters):
        access_logger.addFilter(_QuietAccessLogFilter())


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


def _build_router(
    compression_enabled: bool, *, safe: bool, aggressive: bool = False
) -> ContentRouter:
    """Build a content router.

    json_table (lossless JSON) and code compression run in every mode. The lossy
    compressors — log folding and prose sentence-dropping — are added only when
    not ``safe``. The aggressive chain additionally enables json_smart record
    sampling; it is used only by budget enforcement when context would otherwise
    overflow.
    """
    router = ContentRouter()
    if compression_enabled:
        router.register(JsonTableCompressor())
        router.register(CodeCompressor())
        if not safe:
            router.register(LogCompressor())
            router.register(ProseCompressor())
            if aggressive:
                router.register(JsonSmartCompressor())
    return router


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
    # Two routers are kept so a request can override the global mode per call via
    # the X-Contextly-Mode header: the default chain and a lossless "safe" chain.
    app.state.content_router = _build_router(config.compression_enabled, safe=config.safe_mode)
    app.state.content_router_safe = _build_router(config.compression_enabled, safe=True)
    app.state.content_router_aggressive = _build_router(
        config.compression_enabled, safe=False, aggressive=True
    )
    app.state.ccr_store = (
        SQLiteCCRStore(config.ccr_path) if config.ccr_backend == "sqlite" else CCRStore()
    )
    app.state.ab_monitor = ABMonitor()
    # Read-only view onto the shared file the MCP gateway writes, so the proxy
    # dashboard can surface gateway savings alongside its own (server label "" —
    # snapshot() aggregates every server that wrote to the file).
    app.state.gateway_stats = SQLiteStatsStore(config.gateway_stats_path or default_stats_path())
    app.state.audit_writer = AuditWriter(config.audit_log_path) if config.audit_log_path else None
    app.state.injection_scanner = InjectionScanner()
    app.state.message_scorer = MessageScorer()
    app.state.cache_optimizer = CacheOptimizer()
    app.state.adaptive_controller = AdaptiveController(
        paradox_threshold=config.adaptive_paradox_threshold,
        min_quality=config.adaptive_min_quality,
        window=config.adaptive_window,
    )
    primary = FailoverTarget(
        url=config.resolved_upstream_url(),
        api_key=config.upstream_api_key,
        provider=str(config.upstream),
    )
    fallbacks = [
        FailoverTarget(url=t["url"], api_key=t["api_key"], provider=t.get("provider", t["url"]))
        for t in config.failover_upstreams
    ]
    app.state.failover_router = FailoverRouter(
        targets=[primary, *fallbacks],
        max_retries=config.failover_max_retries,
    )
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
