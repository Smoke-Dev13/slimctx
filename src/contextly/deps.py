"""FastAPI dependency providers for shared app-level resources.

Starlette's `State` exposes arbitrary attributes via __getattr__, which mypy
types as returning `Any`. We use `cast()` to narrow the return type instead
of suppressing with type: ignore, keeping the rest of the codebase fully typed.
"""

from __future__ import annotations

from typing import Annotated, cast

import httpx
from fastapi import Depends, Request

from contextly.ab_monitor import ABMonitor
from contextly.audit import AuditWriter
from contextly.ccr import CCRStore
from contextly.compressors.registry import ContentRouter
from contextly.config import Config
from contextly.failover import FailoverRouter
from contextly.gateway_stats import SQLiteStatsStore
from contextly.injection import InjectionScanner
from contextly.scorer import MessageScorer


def _get_config(request: Request) -> Config:
    return cast(Config, request.app.state.config)


def _get_http_client(request: Request) -> httpx.AsyncClient:
    return cast(httpx.AsyncClient, request.app.state.http_client)


def _get_content_router(request: Request) -> ContentRouter:
    return cast(ContentRouter, request.app.state.content_router)


def _get_safe_content_router(request: Request) -> ContentRouter:
    return cast(ContentRouter, request.app.state.content_router_safe)


def _get_ccr_store(request: Request) -> CCRStore:
    return cast(CCRStore, request.app.state.ccr_store)


def _get_ab_monitor(request: Request) -> ABMonitor:
    return cast(ABMonitor, request.app.state.ab_monitor)


def _get_gateway_stats(request: Request) -> SQLiteStatsStore:
    return cast(SQLiteStatsStore, request.app.state.gateway_stats)


def _get_aggressive_content_router(request: Request) -> ContentRouter:
    return cast(ContentRouter, request.app.state.content_router_aggressive)


def _get_audit_writer(request: Request) -> AuditWriter | None:
    return cast(AuditWriter | None, getattr(request.app.state, "audit_writer", None))


def _get_injection_scanner(request: Request) -> InjectionScanner:
    return cast(InjectionScanner, request.app.state.injection_scanner)


def _get_message_scorer(request: Request) -> MessageScorer:
    return cast(MessageScorer, request.app.state.message_scorer)


def _get_failover_router(request: Request) -> FailoverRouter:
    return cast(FailoverRouter, request.app.state.failover_router)


ConfigDep = Annotated[Config, Depends(_get_config)]
HttpClientDep = Annotated[httpx.AsyncClient, Depends(_get_http_client)]
ContentRouterDep = Annotated[ContentRouter, Depends(_get_content_router)]
SafeContentRouterDep = Annotated[ContentRouter, Depends(_get_safe_content_router)]
AggressiveContentRouterDep = Annotated[ContentRouter, Depends(_get_aggressive_content_router)]
CCRDep = Annotated[CCRStore, Depends(_get_ccr_store)]
ABMonitorDep = Annotated[ABMonitor, Depends(_get_ab_monitor)]
GatewayStatsDep = Annotated[SQLiteStatsStore, Depends(_get_gateway_stats)]
AuditWriterDep = Annotated[AuditWriter | None, Depends(_get_audit_writer)]
InjectionScannerDep = Annotated[InjectionScanner, Depends(_get_injection_scanner)]
MessageScorerDep = Annotated[MessageScorer, Depends(_get_message_scorer)]
FailoverRouterDep = Annotated[FailoverRouter, Depends(_get_failover_router)]
