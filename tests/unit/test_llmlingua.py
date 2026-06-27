"""Unit tests for LLMLinguaCompressor (optional ML compressor)."""

from __future__ import annotations

from contextly.compressors.llmlingua import (
    LLMLinguaCompressor,
    _rate_from_query,
)

_PROSE = (
    "The quick brown fox jumps over the lazy dog. "
    "Compression should preserve the essential meaning of this text. "
    "Neural token classification drops low-information words. "
    "This sentence exists to make the content long enough to qualify."
)


def test_rate_from_query_default() -> None:
    assert _rate_from_query("") == 0.6


def test_rate_from_query_aggressive() -> None:
    assert _rate_from_query("please summarize this") == 0.4


def test_rate_from_query_conservative() -> None:
    assert _rate_from_query("find the exact value") == 0.8


def test_should_apply_false_when_dependency_absent() -> None:
    # In CI the `llmlingua` package is not installed → model is None → never selected.
    c = LLMLinguaCompressor()
    assert c.should_apply(_PROSE) is False


def test_compress_passthrough_when_model_unavailable() -> None:
    c = LLMLinguaCompressor()
    result = c.compress(_PROSE)
    assert result.content == _PROSE
    assert result.compression_ratio == 1.0
    assert result.compressor_name == "llmlingua"


def test_name() -> None:
    assert LLMLinguaCompressor().name == "llmlingua"


def test_compress_uses_model_when_available() -> None:
    """With the model mocked, compress returns the model's output."""

    class _FakeModel:
        def compress_prompt(self, content: str, rate: float) -> dict[str, str]:
            return {"compressed_prompt": "short version"}

    c = LLMLinguaCompressor()
    c._model = _FakeModel()  # type: ignore[assignment]
    c._load_attempted = True
    assert c.should_apply(_PROSE) is True
    result = c.compress(_PROSE, "summarize")
    assert result.content == "short version"
    assert result.compression_ratio < 1.0
    assert result.metadata["rate"] == 0.4


def test_compress_guards_against_larger_output() -> None:
    class _BloatModel:
        def compress_prompt(self, content: str, rate: float) -> dict[str, str]:
            return {"compressed_prompt": content + " padding padding padding"}

    c = LLMLinguaCompressor()
    c._model = _BloatModel()  # type: ignore[assignment]
    c._load_attempted = True
    result = c.compress(_PROSE)
    # larger output is rejected → passthrough
    assert result.content == _PROSE
    assert result.compression_ratio == 1.0


def test_compress_handles_model_error() -> None:
    class _BrokenModel:
        def compress_prompt(self, content: str, rate: float) -> dict[str, str]:
            raise RuntimeError("boom")

    c = LLMLinguaCompressor()
    c._model = _BrokenModel()  # type: ignore[assignment]
    c._load_attempted = True
    result = c.compress(_PROSE)
    assert result.content == _PROSE
    assert result.compression_ratio == 1.0
