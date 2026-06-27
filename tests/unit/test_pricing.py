"""Unit tests for pricing.py."""

from __future__ import annotations

import pytest

from contextly.pricing import price_per_1k_tokens, tokens_to_dollars


def test_known_model_exact() -> None:
    p = price_per_1k_tokens("gpt-4o")
    assert p == pytest.approx(0.0025)


def test_known_model_prefix_match() -> None:
    # "gpt-4o-2024-11-20" should prefix-match "gpt-4o"
    p = price_per_1k_tokens("gpt-4o-2024-11-20")
    assert p == pytest.approx(0.0025)


def test_unknown_model_fallback() -> None:
    p = price_per_1k_tokens("totally-unknown-model-xyz")
    assert p == pytest.approx(0.002)


def test_overrides_exact_match() -> None:
    overrides = {"my-model": 0.005}
    assert price_per_1k_tokens("my-model", overrides) == pytest.approx(0.005)


def test_overrides_prefix_match() -> None:
    overrides = {"my-model": 0.005}
    assert price_per_1k_tokens("my-model-v2", overrides) == pytest.approx(0.005)


def test_overrides_fallback_to_builtin() -> None:
    overrides = {"other-model": 0.001}
    assert price_per_1k_tokens("gpt-4o", overrides) == pytest.approx(0.0025)


def test_tokens_to_dollars_basic() -> None:
    dollars = tokens_to_dollars(1000, "gpt-4o")
    assert dollars == pytest.approx(0.0025)


def test_tokens_to_dollars_zero() -> None:
    assert tokens_to_dollars(0, "gpt-4o") == pytest.approx(0.0)


def test_tokens_to_dollars_with_overrides() -> None:
    overrides = {"my-model": 0.01}
    dollars = tokens_to_dollars(2000, "my-model", overrides)
    assert dollars == pytest.approx(0.02)
