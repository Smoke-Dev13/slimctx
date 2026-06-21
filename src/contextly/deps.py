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
from contextly.ccr import CCRStore
from contextly.compressors.registry import ContentRouter
from contextly.config import Config


def _get_config(request: Request) -> Config:
    return cast(Config, request.app.state.config)


def _get_http_client(request: Request) -> httpx.AsyncClient:
    return cast(httpx.AsyncClient, request.app.state.http_client)


def _get_content_router(request: Request) -> ContentRouter:
    return cast(ContentRouter, request.app.state.content_router)


def _get_ccr_store(request: Request) -> CCRStore:
    return cast(CCRStore, request.app.state.ccr_store)


def _get_ab_monitor(request: Request) -> ABMonitor:
    return cast(ABMonitor, request.app.state.ab_monitor)


ConfigDep = Annotated[Config, Depends(_get_config)]
HttpClientDep = Annotated[httpx.AsyncClient, Depends(_get_http_client)]
ContentRouterDep = Annotated[ContentRouter, Depends(_get_content_router)]
CCRDep = Annotated[CCRStore, Depends(_get_ccr_store)]
ABMonitorDep = Annotated[ABMonitor, Depends(_get_ab_monitor)]
