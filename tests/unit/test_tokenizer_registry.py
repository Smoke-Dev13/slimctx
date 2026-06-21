"""Unit tests for the tokenizer registry / factory."""

from __future__ import annotations

import pytest

from contextly.tokenizer.registry import get_tokenizer
from contextly.tokenizer.tiktoken_bundled import BundledTiktokenTokenizer
from contextly.tokenizer.word_fallback import WordTokenizer


@pytest.mark.parametrize(
    "model,expected_encoding",
    [
        ("gpt-4o", "o200k_base"),
        ("gpt-4o-mini", "o200k_base"),
        ("gpt-4o-2024-08-06", "o200k_base"),
        ("o1", "o200k_base"),
        ("o1-mini", "o200k_base"),
        ("o3", "o200k_base"),
        ("gpt-4", "cl100k_base"),
        ("gpt-4-turbo", "cl100k_base"),
        ("gpt-3.5-turbo", "cl100k_base"),
        ("gpt-3.5-turbo-0125", "cl100k_base"),
        ("claude-3-5-sonnet-20241022", "cl100k_base"),
    ],
)
def test_known_model_returns_bundled_tokenizer(model: str, expected_encoding: str) -> None:
    tok = get_tokenizer(model)
    assert isinstance(tok, BundledTiktokenTokenizer)
    assert tok.encoding_name == expected_encoding


def test_unknown_model_returns_word_tokenizer() -> None:
    tok = get_tokenizer("some-unknown-llm-9000")
    assert isinstance(tok, WordTokenizer)


def test_prefix_matching_gpt4_variant() -> None:
    tok = get_tokenizer("gpt-4-custom-finetune-v3")
    assert isinstance(tok, BundledTiktokenTokenizer)
    assert tok.encoding_name == "cl100k_base"


def test_prefix_matching_gpt4o_variant() -> None:
    tok = get_tokenizer("gpt-4o-2025-01-15")
    assert isinstance(tok, BundledTiktokenTokenizer)
    assert tok.encoding_name == "o200k_base"


def test_tokenizer_can_count(request: pytest.FixtureRequest) -> None:
    tok = get_tokenizer("gpt-4o")
    assert tok.count("hello world") > 0


def test_fallback_tokenizer_can_count() -> None:
    tok = get_tokenizer("completely-unknown-model")
    assert tok.count("hello world") > 0
