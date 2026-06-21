"""Integration tests for observability endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
def test_health_status_200(client: TestClient) -> None:
    assert client.get("/health").status_code == 200


@pytest.mark.integration
def test_health_body_schema(client: TestClient) -> None:
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"
    assert "upstream" in data
    assert "upstream_url" in data
    assert "compression_enabled" in data


@pytest.mark.integration
def test_health_upstream_defaults_to_openai(client: TestClient) -> None:
    data = client.get("/health").json()
    assert data["upstream"] == "openai"
    assert "openai.com" in data["upstream_url"]


@pytest.mark.integration
def test_stats_status_200(client: TestClient) -> None:
    assert client.get("/stats").status_code == 200


@pytest.mark.integration
def test_stats_body_schema(client: TestClient) -> None:
    data = client.get("/stats").json()
    assert "requests_total" in data
    assert "chars_saved_total" in data


@pytest.mark.integration
def test_metrics_status_200(client: TestClient) -> None:
    assert client.get("/metrics").status_code == 200


@pytest.mark.integration
def test_quality_status_200(client: TestClient) -> None:
    assert client.get("/quality").status_code == 200
