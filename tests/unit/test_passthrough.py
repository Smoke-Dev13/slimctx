"""Unit tests for PassthroughCompressor."""

from __future__ import annotations

import pytest

from contextly.compressors.base import CompressResult
from contextly.compressors.passthrough import PassthroughCompressor


@pytest.fixture
def compressor() -> PassthroughCompressor:
    return PassthroughCompressor()


def test_name(compressor: PassthroughCompressor) -> None:
    assert compressor.name == "passthrough"


def test_should_apply_always_true(compressor: PassthroughCompressor) -> None:
    assert compressor.should_apply("anything") is True
    assert compressor.should_apply("") is True
    assert compressor.should_apply("x" * 10_000) is True
    assert compressor.should_apply("data", query="find anomalies") is True


def test_compress_returns_content_unchanged(compressor: PassthroughCompressor) -> None:
    content = "Hello, world! Some test content here."
    result = compressor.compress(content)
    assert result.content == content


def test_compress_result_type(compressor: PassthroughCompressor) -> None:
    result = compressor.compress("some text")
    assert isinstance(result, CompressResult)


def test_compression_ratio_is_one(compressor: PassthroughCompressor) -> None:
    result = compressor.compress("some content here")
    assert result.compression_ratio == pytest.approx(1.0)


def test_compress_empty_string(compressor: PassthroughCompressor) -> None:
    result = compressor.compress("")
    assert result.content == ""
    assert result.original_length == 0
    assert result.compressed_length == 0
    assert result.compression_ratio == 1.0


def test_compress_reports_correct_lengths(compressor: PassthroughCompressor) -> None:
    content = "hello world"
    result = compressor.compress(content)
    assert result.original_length == len(content)
    assert result.compressed_length == len(content)


def test_compress_metadata_empty(compressor: PassthroughCompressor) -> None:
    result = compressor.compress("data")
    assert result.metadata == {}
    assert result.compressor_name == "passthrough"


def test_compress_with_query_hint_still_passthrough(compressor: PassthroughCompressor) -> None:
    result = compressor.compress("some data", query="find all anomalies please")
    assert result.content == "some data"
    assert result.compression_ratio == pytest.approx(1.0)


def test_tokens_saved_estimate_is_zero(compressor: PassthroughCompressor) -> None:
    result = compressor.compress("some content text here")
    assert result.tokens_saved_estimate == 0


def test_compress_unicode(compressor: PassthroughCompressor) -> None:
    content = "Привет мир 🌍"
    result = compressor.compress(content)
    assert result.content == content


def test_compress_large_content(compressor: PassthroughCompressor) -> None:
    content = "word " * 5000
    result = compressor.compress(content)
    assert result.content == content
    assert result.compression_ratio == pytest.approx(1.0)
