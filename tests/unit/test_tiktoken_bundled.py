"""Unit tests for BundledTiktokenTokenizer.

These tests prove that:
  1. The tokenizer loads from package_data (is_bundled returns True).
  2. Token counts match tiktoken's reference implementation.
  3. No network access is required (see test_offline_first.py for the
     full-flow socket-disabled proof).
"""

from __future__ import annotations

import pytest
import tiktoken

from contextly.tokenizer.tiktoken_bundled import BundledTiktokenTokenizer, is_bundled


@pytest.mark.parametrize("encoding", ["cl100k_base", "o200k_base"])
def test_is_bundled(encoding: str) -> None:
    assert is_bundled(encoding) is True


def test_is_bundled_unknown_returns_false() -> None:
    assert is_bundled("nonexistent_encoding_xyz") is False


@pytest.mark.parametrize("encoding", ["cl100k_base", "o200k_base"])
def test_encoding_name(encoding: str) -> None:
    tok = BundledTiktokenTokenizer(encoding)
    assert tok.encoding_name == encoding


def test_unknown_encoding_raises() -> None:
    with pytest.raises(KeyError, match="Unsupported encoding"):
        BundledTiktokenTokenizer("gpt2")


@pytest.mark.parametrize("encoding", ["cl100k_base", "o200k_base"])
def test_count_matches_reference(encoding: str) -> None:
    """BundledTokenizer must produce the same count as tiktoken.get_encoding."""
    tok = BundledTiktokenTokenizer(encoding)
    ref = tiktoken.get_encoding(encoding)
    samples = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "SELECT * FROM users WHERE id = 42;",
        "def foo(x: int) -> str:\n    return str(x)\n",
        "안녕하세요! 日本語テスト. Привет мир.",
        "",
    ]
    for text in samples:
        bundled_count = tok.count(text)
        ref_count = len(ref.encode(text, disallowed_special=()))
        assert bundled_count == ref_count, (
            f"[{encoding}] count mismatch for {text!r}: "
            f"bundled={bundled_count}, reference={ref_count}"
        )


@pytest.mark.parametrize("encoding", ["cl100k_base", "o200k_base"])
def test_encode_matches_reference(encoding: str) -> None:
    tok = BundledTiktokenTokenizer(encoding)
    ref = tiktoken.get_encoding(encoding)
    text = "contextly reduces LLM token costs."
    assert tok.encode(text) == ref.encode(text, disallowed_special=())


def test_count_empty_string() -> None:
    tok = BundledTiktokenTokenizer("cl100k_base")
    assert tok.count("") == 0


def test_count_unicode_cl100k() -> None:
    tok = BundledTiktokenTokenizer("cl100k_base")
    assert tok.count("hello") > 0


def test_count_increases_with_text_length() -> None:
    tok = BundledTiktokenTokenizer("cl100k_base")
    short_count = tok.count("hello")
    long_count = tok.count("hello world foo bar baz qux")
    assert long_count > short_count
