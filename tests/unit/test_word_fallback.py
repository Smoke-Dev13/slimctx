"""Unit tests for WordTokenizer."""

from __future__ import annotations

import pytest

from contextly.tokenizer.word_fallback import WordTokenizer


@pytest.fixture
def tok() -> WordTokenizer:
    return WordTokenizer()


def test_encoding_name(tok: WordTokenizer) -> None:
    assert tok.encoding_name == "word"


def test_empty_string_is_zero(tok: WordTokenizer) -> None:
    assert tok.count("") == 0


def test_single_word(tok: WordTokenizer) -> None:
    assert tok.count("hello") == 1


def test_two_words(tok: WordTokenizer) -> None:
    assert tok.count("hello world") == 2


def test_punctuation_counted_separately(tok: WordTokenizer) -> None:
    assert tok.count("Hello, world!") == 4  # Hello , world !


def test_only_whitespace_is_zero(tok: WordTokenizer) -> None:
    assert tok.count("   \t\n  ") == 0


def test_equation_tokens(tok: WordTokenizer) -> None:
    # "x=1+2" → x, =, 1, +, 2
    assert tok.count("x=1+2") == 5


def test_longer_text_positive(tok: WordTokenizer) -> None:
    count = tok.count("The quick brown fox jumps over the lazy dog.")
    assert count > 0


def test_count_monotonic_with_length(tok: WordTokenizer) -> None:
    short = tok.count("hello")
    long = tok.count("hello world foo bar baz")
    assert long > short


def test_encode_returns_integers(tok: WordTokenizer) -> None:
    ids = tok.encode("hello world")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)


def test_encode_length_matches_count(tok: WordTokenizer) -> None:
    text = "The quick, brown fox!"
    assert len(tok.encode(text)) == tok.count(text)


def test_encode_empty_is_empty_list(tok: WordTokenizer) -> None:
    assert tok.encode("") == []


def test_unicode_text(tok: WordTokenizer) -> None:
    count = tok.count("Привет мир")
    assert count == 2


def test_mixed_unicode_punct(tok: WordTokenizer) -> None:
    count = tok.count("café, naïve!")
    assert count >= 2
