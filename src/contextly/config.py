"""Runtime configuration for Contextly, sourced from environment variables and .env files.

All settings are accessible via the CONTEXTLY_ prefix (e.g., CONTEXTLY_PORT=8080)
or by passing keyword arguments directly when constructing Config in tests.
Precedence: explicit kwargs > env vars > .env file > defaults.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class UpstreamProvider(StrEnum):
    """Named upstream LLM providers with well-known base URLs."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    CUSTOM = "custom"


_PROVIDER_BASE_URLS: dict[UpstreamProvider, str] = {
    UpstreamProvider.OPENAI: "https://api.openai.com",
    UpstreamProvider.ANTHROPIC: "https://api.anthropic.com",
    UpstreamProvider.OPENROUTER: "https://openrouter.ai/api",
}


class Config(BaseSettings):
    """All Contextly runtime settings, validated by pydantic-settings."""

    model_config = SettingsConfigDict(
        env_prefix="CONTEXTLY_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        populate_by_name=True,
    )

    # ── Server ──────────────────────────────────────────────────────────────
    host: str = "127.0.0.1"
    port: Annotated[int, Field(ge=1, le=65535)] = 4000
    workers: Annotated[int, Field(ge=1)] = 1
    log_level: str = "info"

    # ── Upstream ────────────────────────────────────────────────────────────
    upstream: UpstreamProvider = UpstreamProvider.OPENAI
    upstream_base_url: AnyHttpUrl | None = None
    upstream_api_key: str = Field(default="", repr=False)

    # ── Compression ─────────────────────────────────────────────────────────
    compression_enabled: bool = True
    target_token_budget: int | None = None
    # Safe mode guarantees the model still sees every JSON record and every
    # prose sentence: the lossy compressors (json_smart record sampling, prose
    # sentence dropping) are disabled, leaving only structure-preserving code
    # compression (comment/whitespace stripping). Trades savings for fidelity.
    safe_mode: bool = False

    # ── A/B Quality ─────────────────────────────────────────────────────────
    ab_sample_rate: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0

    def resolved_upstream_url(self) -> str:
        """Return the effective upstream base URL (no trailing slash).

        Returns:
            The configured upstream_base_url, or the well-known URL for the
            selected upstream provider.
        """
        if self.upstream_base_url is not None:
            return str(self.upstream_base_url).rstrip("/")
        return _PROVIDER_BASE_URLS.get(
            self.upstream,
            _PROVIDER_BASE_URLS[UpstreamProvider.OPENAI],
        )
