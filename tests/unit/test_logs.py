"""Unit tests for LogCompressor."""

from __future__ import annotations

from contextly.compressors.logs import LogCompressor, _template

# 200 repetitive log lines: same two messages with varying timestamp/number.
_LOG = (
    "\n".join(
        f"2026-06-22T10:00:{i % 60:02d} INFO  GET /users/{i} -> 200 in {i % 90}ms"
        for i in range(200)
    )
    + "\n2026-06-22T10:05:01 ERROR upstream timeout to 10.0.0.5"
)

_PROSE = (
    "Machine learning models require careful tuning. The learning rate matters most. "
    "Regularization prevents overfitting on the training data. Transfer learning helps."
)


def _compress(text: str) -> object:
    return LogCompressor().compress(text)


# ── _template ───────────────────────────────────────────────────────────────────


def test_template_masks_timestamp_and_numbers() -> None:
    t = _template("2026-06-22T10:00:01 INFO GET /users/42 -> 200 in 13ms")
    assert "2026" not in t and "42" not in t and "200" not in t
    assert "INFO GET /users/" in t


def test_template_groups_varying_lines() -> None:
    a = _template("12:00:01 INFO request 5 done")
    b = _template("12:59:59 INFO request 9999 done")
    assert a == b


# ── should_apply ────────────────────────────────────────────────────────────────


def test_should_apply_on_logs() -> None:
    assert LogCompressor().should_apply(_LOG) is True


def test_should_not_apply_on_prose() -> None:
    assert LogCompressor().should_apply(_PROSE) is False


def test_should_not_apply_on_few_lines() -> None:
    assert LogCompressor().should_apply("2026-01-01 INFO one line") is False


# ── compress ────────────────────────────────────────────────────────────────────


def test_folds_repeated_lines() -> None:
    result = _compress(_LOG)
    assert result.compressor_name == "logs"
    assert result.compressed_length < result.original_length
    # 200 GET lines + 1 ERROR line collapse to 2 patterns.
    assert result.metadata["unique_patterns"] == 2
    assert result.metadata["total_lines"] == 201
    assert "(x200)" in result.content


def test_marks_lossy() -> None:
    assert _compress(_LOG).metadata["lossless"] is False


def test_passthrough_when_no_repetition() -> None:
    # All-distinct log lines: folding cannot beat the original, so passthrough.
    distinct = "\n".join(f"2026-06-22 INFO unique message {chr(65 + i)}" for i in range(8))
    result = _compress(distinct)
    assert result.content == distinct
