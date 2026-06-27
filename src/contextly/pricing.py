"""Model pricing table for dollar-savings tracking.

Prices are per 1,000 input tokens (USD). Override with
CONTEXTLY_PRICING_OVERRIDES={"my-model": 0.002} in the environment.

The table is intentionally approximate — it gives teams a business-case
number in the dashboard rather than precise billing (providers change prices;
use your actual invoice for accounting).
"""

from __future__ import annotations

# Dollars per 1,000 input tokens.
# Source: public pricing pages, approximate as of 2026.
_PRICING: dict[str, float] = {
    # OpenAI
    "gpt-4o": 0.0025,
    "gpt-4o-mini": 0.00015,
    "gpt-4-turbo": 0.01,
    "gpt-4": 0.03,
    "gpt-3.5-turbo": 0.0005,
    "o1": 0.015,
    "o1-mini": 0.003,
    "o3": 0.01,
    "o3-mini": 0.0011,
    "o4-mini": 0.0011,
    # Anthropic Claude
    "claude-opus-4-8": 0.015,
    "claude-sonnet-4-6": 0.003,
    "claude-haiku-4-5-20251001": 0.0008,
    "claude-3-5-sonnet-20241022": 0.003,
    "claude-3-5-haiku-20241022": 0.0008,
    "claude-3-opus-20240229": 0.015,
    "claude-3-haiku-20240307": 0.00025,
    # Groq (free-tier; approximate retail equivalent)
    "llama-3.3-70b-versatile": 0.00059,
    "llama3-8b-8192": 0.00005,
    "mixtral-8x7b-32768": 0.00024,
    # Google
    "gemini-1.5-pro": 0.00125,
    "gemini-1.5-flash": 0.000075,
    "gemini-2.0-flash": 0.0001,
}

# Tokens saved * this multiplier = dollars saved estimate
_DOLLARS_PER_1K = 0.002  # conservative fallback


def price_per_1k_tokens(model: str, overrides: dict[str, float] | None = None) -> float:
    """Return the input price (USD/1K tokens) for *model*.

    Checks *overrides* first, then the built-in table, then returns a
    conservative fallback so the counter never goes to zero on unknown models.

    Args:
        model: Model identifier string (e.g. ``"gpt-4o"``).
        overrides: Optional per-model price overrides from config.

    Returns:
        Dollars per 1,000 input tokens.
    """
    if overrides:
        # Exact match first
        if model in overrides:
            return overrides[model]
        # Prefix match (e.g. "gpt-4o-2024-11-20" → "gpt-4o")
        for key, val in overrides.items():
            if model.startswith(key):
                return val

    if model in _PRICING:
        return _PRICING[model]

    # Prefix match against built-in table
    for key, val in _PRICING.items():
        if model.startswith(key):
            return val

    return _DOLLARS_PER_1K


def tokens_to_dollars(
    tokens_saved: int, model: str, overrides: dict[str, float] | None = None
) -> float:
    """Convert *tokens_saved* to an estimated dollar saving for *model*.

    Args:
        tokens_saved: Number of input tokens saved by compression.
        model: Model identifier used to look up price.
        overrides: Optional price overrides from config.

    Returns:
        Estimated dollars saved (non-negative float).
    """
    return max(0.0, tokens_saved / 1000.0 * price_per_1k_tokens(model, overrides))
