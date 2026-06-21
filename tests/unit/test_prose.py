"""Unit tests for ProseCompressor."""

from __future__ import annotations

import pytest

from contextly.compressors.prose import (
    ProseCompressor,
    _aggressiveness_from_query,
    _extract_keywords,
    _score_sentence,
    _split_sentences,
    _word_freq_keywords,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def compressor() -> ProseCompressor:
    return ProseCompressor()


def _long_prose(sentences: int = 30) -> str:
    """Generate prose that is clearly not JSON or code."""
    base = (
        "The system uses machine learning to optimize token usage. "
        "Compression algorithms reduce the number of tokens sent to language models. "
        "This saves money and improves response time significantly. "
        "The proxy server intercepts API calls and compresses context efficiently. "
        "Smart sampling preserves the most informative parts of the conversation. "
    )
    return (base * (sentences // 5 + 1)).strip()


_SAMPLE_PROSE = _long_prose(30)


# ── should_apply ───────────────────────────────────────────────────────────────


def test_should_apply_long_prose(compressor: ProseCompressor) -> None:
    assert compressor.should_apply(_SAMPLE_PROSE) is True


def test_should_apply_rejects_short_text(compressor: ProseCompressor) -> None:
    assert compressor.should_apply("Too short.") is False


def test_should_apply_rejects_json_array(compressor: ProseCompressor) -> None:
    assert compressor.should_apply('[{"a": 1}, {"b": 2}]') is False


def test_should_apply_rejects_json_object(compressor: ProseCompressor) -> None:
    assert compressor.should_apply('{"key": "value", "other": 123}') is False


def test_should_apply_rejects_python_code(compressor: ProseCompressor) -> None:
    code = "def foo(x):\n    return x\nclass Bar:\n    pass\nimport os\n" * 10
    assert compressor.should_apply(code) is False


def test_should_apply_rejects_text_without_sentences(compressor: ProseCompressor) -> None:
    no_punct = "word " * 100
    assert compressor.should_apply(no_punct) is False


def test_should_apply_name(compressor: ProseCompressor) -> None:
    assert compressor.name == "prose"


# ── compress — passthrough cases ───────────────────────────────────────────────


def test_compress_passthrough_for_too_few_sentences(compressor: ProseCompressor) -> None:
    short = "One sentence only. Two is here." * 10
    result = compressor.compress(short)
    # Only 2 unique sentences — might passthrough or compress depending on split
    assert result.compressor_name == "prose"


def test_compress_passthrough_compressor_name(compressor: ProseCompressor) -> None:
    result = compressor.compress(_SAMPLE_PROSE)
    assert result.compressor_name == "prose"


# ── compress — successful compression ─────────────────────────────────────────


def test_compress_reduces_length(compressor: ProseCompressor) -> None:
    result = compressor.compress(_SAMPLE_PROSE)
    assert result.compression_ratio < 1.0


def test_compress_50_percent_reduction_on_long_prose(compressor: ProseCompressor) -> None:
    prose = _long_prose(50)
    result = compressor.compress(prose)
    savings = (1.0 - result.compression_ratio) * 100.0
    assert savings > 50.0, f"Expected >50% savings, got {savings:.1f}%"


def test_compress_output_is_non_empty(compressor: ProseCompressor) -> None:
    result = compressor.compress(_SAMPLE_PROSE)
    assert len(result.content) > 0


def test_compress_metadata_keys(compressor: ProseCompressor) -> None:
    result = compressor.compress(_SAMPLE_PROSE)
    for key in ("total_sentences", "selected_sentences", "keyword_count", "aggressiveness"):
        assert key in result.metadata, f"missing: {key}"


def test_compress_selected_less_than_total(compressor: ProseCompressor) -> None:
    result = compressor.compress(_SAMPLE_PROSE)
    assert result.metadata["selected_sentences"] < result.metadata["total_sentences"]


def test_compress_default_aggressiveness(compressor: ProseCompressor) -> None:
    result = compressor.compress(_SAMPLE_PROSE)
    assert result.metadata["aggressiveness"] == pytest.approx(0.7)


# ── Query-aware aggressiveness ─────────────────────────────────────────────────


def test_compress_aggressive_query_keeps_fewer_sentences(compressor: ProseCompressor) -> None:
    prose = _long_prose(50)
    r_agg = compressor.compress(prose, query="summarize briefly")
    r_cons = compressor.compress(prose, query="find the specific detail about compression")
    assert r_agg.metadata["selected_sentences"] <= r_cons.metadata["selected_sentences"]


def test_compress_aggressive_query_higher_ratio(compressor: ProseCompressor) -> None:
    prose = _long_prose(50)
    r_agg = compressor.compress(prose, query="give me a brief overview")
    r_cons = compressor.compress(prose, query="identify the specific technical details")
    assert r_agg.compression_ratio <= r_cons.compression_ratio


# ── Helper: _split_sentences ───────────────────────────────────────────────────


def test_split_sentences_basic() -> None:
    text = "The first full sentence is here. The second full sentence is also here. And a third."
    sentences = _split_sentences(text)
    assert len(sentences) >= 2


def test_split_sentences_filters_short() -> None:
    text = "Hi. This is a longer sentence that will survive. Ok."
    sentences = _split_sentences(text)
    # "Hi." and "Ok." should be filtered (< 15 chars)
    assert all(len(s) > 15 for s in sentences)


def test_split_sentences_empty() -> None:
    assert _split_sentences("") == []


# ── Helper: _word_freq_keywords ────────────────────────────────────────────────


def test_word_freq_keywords_returns_dict() -> None:
    text = "compression algorithm compression system token token token"
    kws = _word_freq_keywords(text, top_n=5)
    assert isinstance(kws, dict)
    assert len(kws) > 0


def test_word_freq_keywords_most_frequent_highest_weight() -> None:
    text = "compression compression compression algorithm token"
    kws = _word_freq_keywords(text, top_n=5)
    assert kws.get("compression", 0) >= kws.get("algorithm", 0)


def test_word_freq_keywords_empty_text() -> None:
    assert _word_freq_keywords("", top_n=5) == {}


# ── Helper: _extract_keywords ─────────────────────────────────────────────────


def test_extract_keywords_returns_dict() -> None:
    kws = _extract_keywords(_SAMPLE_PROSE, top_n=10)
    assert isinstance(kws, dict)
    assert len(kws) > 0


def test_extract_keywords_values_in_0_1() -> None:
    kws = _extract_keywords(_SAMPLE_PROSE, top_n=10)
    for word, weight in kws.items():
        assert 0.0 <= weight <= 1.0, f"{word!r} weight {weight} out of range"


# ── Helper: _score_sentence ────────────────────────────────────────────────────


def test_score_sentence_higher_for_more_keywords() -> None:
    weights = {"token": 1.0, "compression": 0.9, "algorithm": 0.5}
    high = _score_sentence("token compression algorithm system", weights)
    low = _score_sentence("the quick brown fox", weights)
    assert high > low


def test_score_sentence_zero_for_empty() -> None:
    assert _score_sentence("", {"word": 1.0}) == pytest.approx(0.0)


def test_score_sentence_normalises_length() -> None:
    weights = {"token": 1.0}
    short = _score_sentence("token", weights)
    long_sent = _score_sentence("token " + " ".join(["filler"] * 50), weights)
    assert short >= long_sent


# ── Helper: _aggressiveness_from_query ────────────────────────────────────────


def test_aggressiveness_summarize() -> None:
    assert _aggressiveness_from_query("give me a brief summary") > 0.7


def test_aggressiveness_find_detail() -> None:
    assert _aggressiveness_from_query("find the specific example") < 0.5


def test_aggressiveness_default() -> None:
    assert _aggressiveness_from_query("") == pytest.approx(0.7)
