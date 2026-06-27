"""Runtime configuration for Contextly, sourced from environment variables and .env files.

All settings are accessible via the CONTEXTLY_ prefix (e.g., CONTEXTLY_PORT=8080)
or by passing keyword arguments directly when constructing Config in tests.
Precedence: explicit kwargs > env vars > .env file > defaults.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Context window sizes (input tokens) for known models.
# Used by budget enforcement to detect when a request would overflow.
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
    "o1": 200_000,
    "o1-mini": 128_000,
    "o3": 200_000,
    "o3-mini": 200_000,
    "o4-mini": 200_000,
    "claude-opus-4-8": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-5-haiku-20241022": 200_000,
    "claude-3-opus-20240229": 200_000,
    "claude-3-haiku-20240307": 200_000,
    "gemini-1.5-pro": 2_000_000,
    "gemini-1.5-flash": 1_000_000,
    "gemini-2.0-flash": 1_000_000,
    "llama-3.3-70b-versatile": 128_000,
}


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
    # CCR (reversible store) backend: "memory" (per-process, default) or "sqlite"
    # (a file shared across workers and persisted across restarts — required for
    # correct expand/retrieve when running with --workers > 1).
    ccr_backend: Literal["memory", "sqlite"] = "memory"
    ccr_path: str = ".contextly/ccr.db"

    # ── A/B Quality ─────────────────────────────────────────────────────────
    ab_sample_rate: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0

    # ── Budget enforcement ───────────────────────────────────────────────────
    # When True, Contextly will automatically escalate the compressor chain
    # (safe → default → aggressive) to keep the estimated input token count
    # within the model's context window. Eliminates context_length_exceeded
    # errors without any app changes.
    budget_enforcement: bool = False

    # ── Cost tracking ────────────────────────────────────────────────────────
    # Per-model price overrides in USD per 1,000 input tokens.
    # Example: CONTEXTLY_PRICING_OVERRIDES={"my-model": 0.002}
    pricing_overrides: dict[str, float] = Field(default_factory=dict)

    # ── Deduplication ────────────────────────────────────────────────────────
    # Cross-message deduplication: replace exact-duplicate content blocks with
    # a sentinel referencing the CCR key of the first occurrence.
    dedup_enabled: bool = True
    dedup_min_chars: Annotated[int, Field(ge=1)] = 200

    # ── Streaming compression ─────────────────────────────────────────────────
    # When True, compress tool-call arguments and assistant text in SSE streams.
    # Off by default — enable once tested against your streaming client.
    stream_compression_enabled: bool = False
    stream_flush_sentences: Annotated[int, Field(ge=1)] = 3

    # ── Audit log ────────────────────────────────────────────────────────────
    # When set, every compression event is appended to a JSONL audit log.
    # Empty string disables auditing.
    audit_log_path: str = ""

    # ── Context reordering ───────────────────────────────────────────────────
    context_reorder_enabled: bool = False
    context_reorder_min_messages: int = 5
    # When reordering is on, keep the older prefix chronological and sort only
    # the fresh tail, so the prompt-cache prefix stays byte-stable across turns.
    context_reorder_cache_stable: bool = True

    # ── ML compression (optional, heavy deps) ─────────────────────────────────
    # Neural extractive compression via LLMLingua-2. Requires the `ml` extra;
    # when the dependency is absent the compressor is simply never selected.
    ml_compression_enabled: bool = False
    ml_compression_model: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"

    # ── Image compression (multimodal) ────────────────────────────────────────
    # Downgrade image detail (and optionally downscale inline images) to cut the
    # token cost of vision inputs. Pillow (the `image` extra) enables downscale.
    image_compression_enabled: bool = False
    image_detail_level: Literal["low", "auto"] = "low"
    image_max_dimension: int = 512

    # ── Adaptive compression controller ──────────────────────────────────────
    # Closed-loop per-session tuning of the compressor chain based on measured
    # response length, A/B quality, and cache hit rate.
    adaptive_compression_enabled: bool = False
    adaptive_paradox_threshold: float = 0.25
    adaptive_min_quality: float = 0.6
    adaptive_window: int = 8

    # ── Prompt-cache optimization ─────────────────────────────────────────────
    # Auto-inject Anthropic cache_control breakpoints and account for cache
    # savings on both OpenAI and Anthropic responses.
    cache_optimization_enabled: bool = False
    cache_min_prefix_chars: int = 4096
    cache_recent_window: int = 2

    # ── Prompt injection detection ───────────────────────────────────────────
    injection_detection_enabled: bool = False
    injection_block_threshold: float | None = None

    # ── Semantic firewall (outbound secret / PII redaction) ───────────────────
    # Redact API keys, private keys, cards, SSNs, emails before they reach the
    # upstream LLM. ``reversible`` stores originals in the CCR store (opt-in).
    firewall_enabled: bool = False
    firewall_block_on_secret: bool = False
    firewall_reversible: bool = False

    # ── Multi-model failover ─────────────────────────────────────────────────
    # Ordered list of fallback upstreams. Each entry: {"url": "...", "api_key": "...",
    # "provider": "openrouter"}. Primary upstream is always tried first; these are
    # used only when the primary returns a retryable error (429/5xx/connect failure).
    failover_upstreams: list[dict[str, str]] = Field(default_factory=list)
    failover_max_retries: int = 3

    # ── Gateway stats bridge ────────────────────────────────────────────────
    # The proxy dashboard also surfaces the MCP gateway's savings by reading the
    # shared stats file the gateway writes (so one dashboard shows both). Empty
    # means the default ~/.contextly/gateway_stats.db used by ``mcp-gateway``.
    gateway_stats_path: str = ""

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
