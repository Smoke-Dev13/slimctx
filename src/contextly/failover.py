"""Multi-model failover routing for upstream LLM requests.

Tries a primary upstream first, then rotates through fallbacks on transient
errors (429, 5xx, connection failures). Tracks per-target success/failure
counts for /stats reporting.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any

import httpx

_RETRY_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


@dataclass
class FailoverTarget:
    url: str
    api_key: str
    provider: str


@dataclass
class _TargetStats:
    successes: int = 0
    failures: int = 0


class FailoverRouter:
    """Ordered list of upstream targets; retries on transient errors.

    Thread-safe: counters protected by a lock.
    Instantiate once and share across requests.
    """

    def __init__(
        self,
        targets: list[FailoverTarget],
        max_retries: int = 3,
    ) -> None:
        self._targets = targets
        self._max_retries = max_retries
        self._stats: dict[str, _TargetStats] = {t.provider: _TargetStats() for t in targets}
        self._events_total: int = 0
        self._lock = threading.Lock()

    @property
    def has_fallbacks(self) -> bool:
        return len(self._targets) > 1

    async def attempt(
        self,
        http_client: httpx.AsyncClient,
        path: str,
        base_headers: dict[str, str],
        payload: dict[str, Any],
    ) -> tuple[httpx.Response, str]:
        """Try each target in order, retrying on transient failures.

        Args:
            http_client: Shared async HTTP client.
            path: URL path appended to each target's base URL (e.g. /v1/chat/completions).
            base_headers: Request headers; Authorization is replaced per target.
            payload: JSON body to POST.

        Returns:
            (response, provider_name) — the first successful-or-non-retryable response
            and the name of the target that handled it.
        """
        last_exc: Exception | None = None
        last_resp: httpx.Response | None = None

        for attempt_idx in range(self._max_retries * len(self._targets)):
            target = self._targets[attempt_idx % len(self._targets)]
            url = target.url.rstrip("/") + path
            headers = {**base_headers, "Authorization": f"Bearer {target.api_key}"}

            try:
                resp = await http_client.post(url, headers=headers, json=payload)
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                last_exc = exc
                with self._lock:
                    self._stats[target.provider].failures += 1
                    self._events_total += 1
                backoff = 0.5 * (2 ** (attempt_idx % self._max_retries))
                await asyncio.sleep(backoff)
                continue

            if resp.status_code not in _RETRY_STATUSES:
                with self._lock:
                    self._stats[target.provider].successes += 1
                return resp, target.provider

            last_resp = resp
            with self._lock:
                self._stats[target.provider].failures += 1
                self._events_total += 1
            backoff = 0.5 * (2 ** (attempt_idx % self._max_retries))
            await asyncio.sleep(backoff)

        # All attempts exhausted — return last response or raise last exception.
        if last_resp is not None:
            return last_resp, self._targets[-1].provider
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("failover: no targets configured")

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "failover_events_total": self._events_total,
                "failover_success_by_target": {p: s.successes for p, s in self._stats.items()},
                "failover_failure_by_target": {p: s.failures for p, s in self._stats.items()},
            }
