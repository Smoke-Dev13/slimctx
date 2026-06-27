"""Health, stats, metrics, and quality observability endpoints.

Mounted at the root level (no prefix) so probe URLs stay short:
  GET /health   — liveness probe
  GET /stats    — aggregate compression statistics (JSON)
  GET /metrics  — Prometheus exposition (M7)
  GET /quality  — A/B quality regression report
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from contextly.dashboard import DASHBOARD_HTML
from contextly.deps import (
    ABMonitorDep,
    AdaptiveControllerDep,
    CacheOptimizerDep,
    ConfigDep,
    FailoverRouterDep,
    GatewayStatsDep,
    ImageCompressorDep,
    InjectionScannerDep,
    SecretRedactorDep,
)
from contextly.metrics import CONTENT_TYPE_LATEST, get_metrics_bytes

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["observability"])


@router.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Send the bare base URL to the dashboard so it just works in a browser."""
    return RedirectResponse(url="/dashboard")


@router.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    """Silence the browser's automatic favicon request (no icon to serve)."""
    return Response(status_code=204)


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    """Live token-savings dashboard (polls /stats and /quality in the browser)."""
    return HTMLResponse(DASHBOARD_HTML)


@router.get("/health")
async def health(config: ConfigDep) -> JSONResponse:
    """Liveness probe — returns 200 as long as the server is up."""
    return JSONResponse(
        {
            "status": "ok",
            "version": "0.1.0",
            "upstream": str(config.upstream),
            "upstream_url": config.resolved_upstream_url(),
            "compression_enabled": config.compression_enabled,
        }
    )


@router.get("/stats")
async def stats(
    ab_monitor: ABMonitorDep,
    injection_scanner: InjectionScannerDep,
    failover_router: FailoverRouterDep,
    cache_optimizer: CacheOptimizerDep,
    adaptive_controller: AdaptiveControllerDep,
    image_compressor: ImageCompressorDep,
    secret_redactor: SecretRedactorDep,
) -> JSONResponse:
    """Aggregate runtime statistics.

    Returns compression throughput counters accumulated since server start.
    A/B quality samples are summarised in /quality for the full breakdown.

    Args:
        ab_monitor: A/B monitor holding running counters.

    Returns:
        JSON with request and compression aggregate stats.
    """
    return JSONResponse(
        {
            **ab_monitor.stats(),
            **injection_scanner.stats(),
            **failover_router.stats(),
            **cache_optimizer.stats(),
            **adaptive_controller.stats(),
            **image_compressor.stats(),
            **secret_redactor.stats(),
        }
    )


@router.get("/gateway-stats")
async def gateway_stats(gateway: GatewayStatsDep) -> JSONResponse:
    """Aggregate savings recorded by the MCP gateway(s) sharing the stats file.

    Lets the proxy dashboard surface gateway tool-output compression even though
    the gateway is a separate stdio process. Returns neutral zeros when no gateway
    has run yet.

    Args:
        gateway: Read-only view of the shared SQLite gateway stats store.

    Returns:
        JSON with combined gateway totals and a per-(server, tool) breakdown.
    """
    return JSONResponse(gateway.snapshot())


@router.get("/metrics")
async def metrics() -> Response:
    """Prometheus text format exposition for scraping.

    Returns all ``contextly_*`` metrics in Prometheus text format 0.0.4.
    Suitable for direct scraping by a Prometheus server.

    Returns:
        Plain-text Prometheus metrics in ``text/plain; version=0.0.4`` format.
    """
    return Response(content=get_metrics_bytes(), media_type=CONTENT_TYPE_LATEST)


@router.get("/quality")
async def quality(ab_monitor: ABMonitorDep) -> JSONResponse:
    """A/B quality regression report.

    Returns a detailed breakdown of compression quality scores derived from
    shadow requests (original vs compressed context sent to the same upstream).
    Only populated when ab_sample_rate > 0 in the proxy configuration.

    Args:
        ab_monitor: A/B monitor holding the sample ring buffer.

    Returns:
        JSON with score distribution, savings summary, and per-compressor stats.
    """
    return JSONResponse(ab_monitor.quality_report())
