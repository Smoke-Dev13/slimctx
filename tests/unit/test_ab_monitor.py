"""Unit tests for ABMonitor, quality scorer, and shadow A/B helper."""

from __future__ import annotations

import time

import pytest

from contextly.ab_monitor import (
    ABMonitor,
    ABSample,
    _extract_response_text,
    _numeric_consistency,
    _percentile,
    _quality_score,
)

# ── _quality_score ─────────────────────────────────────────────────────────────


def test_quality_score_identical_texts() -> None:
    assert _quality_score("the cat sat on the mat", "the cat sat on the mat") == pytest.approx(1.0)


def test_quality_score_no_overlap() -> None:
    assert _quality_score("apple banana cherry", "dog elephant frog") == pytest.approx(0.0)


def test_quality_score_partial_overlap() -> None:
    score = _quality_score("hello world goodbye", "hello world")
    assert 0.0 < score < 1.0


def test_quality_score_symmetric() -> None:
    s1 = _quality_score("hello world", "world hello universe")
    s2 = _quality_score("world hello universe", "hello world")
    assert s1 == pytest.approx(s2, abs=1e-6)


def test_quality_score_both_empty() -> None:
    assert _quality_score("", "") == pytest.approx(1.0)


def test_quality_score_reference_empty() -> None:
    assert _quality_score("", "some words here") == pytest.approx(0.0)


def test_quality_score_candidate_empty() -> None:
    assert _quality_score("some words here", "") == pytest.approx(0.0)


def test_quality_score_handles_duplicates() -> None:
    # Counter intersection: "the the the" ∩ "the the" = 2 "the"s
    score = _quality_score("the the the cat", "the the dog")
    assert 0.0 < score < 1.0


def test_quality_score_in_range() -> None:
    ref = "machine learning models require careful hyperparameter tuning"
    cand = "learning models require careful configuration and tuning"
    score = _quality_score(ref, cand)
    assert 0.0 <= score <= 1.0


def test_quality_score_case_insensitive() -> None:
    s1 = _quality_score("Hello World", "hello world")
    assert s1 == pytest.approx(1.0)


# ── _numeric_consistency ───────────────────────────────────────────────────────


def test_numeric_consistency_no_numbers_in_reference() -> None:
    assert _numeric_consistency("no digits here", "still none") == pytest.approx(1.0)


def test_numeric_consistency_all_preserved() -> None:
    ref = "Revenue was 1,240,000 and churn was 3 percent."
    cand = "They reported 1240000 in revenue at 3 percent churn."
    assert _numeric_consistency(ref, cand) == pytest.approx(1.0)


def test_numeric_consistency_thousands_separator_normalised() -> None:
    assert _numeric_consistency("total 1,000", "total 1000") == pytest.approx(1.0)


def test_numeric_consistency_corrupted_number_detected() -> None:
    # ROUGE-1 would score this highly; the numeric figure is wrong.
    score = _numeric_consistency("the count is 312", "the count is 47")
    assert score == pytest.approx(0.0)


def test_numeric_consistency_partial() -> None:
    score = _numeric_consistency("values 10 and 20 and 30", "values 10 and 20")
    assert score == pytest.approx(2.0 / 3.0, abs=1e-4)


# ── _percentile ────────────────────────────────────────────────────────────────


def test_percentile_empty() -> None:
    assert _percentile([], 50) == 0.0


def test_percentile_single_value() -> None:
    assert _percentile([0.5], 50) == 0.5


def test_percentile_p50_of_ordered() -> None:
    values = [0.1, 0.3, 0.5, 0.7, 0.9]
    assert _percentile(values, 50) == 0.5


def test_percentile_p0_returns_minimum() -> None:
    values = [0.1, 0.5, 0.9]
    assert _percentile(values, 0) == 0.1


def test_percentile_p100_returns_maximum() -> None:
    values = [0.1, 0.5, 0.9]
    assert _percentile(values, 100) == 0.9


# ── _extract_response_text ─────────────────────────────────────────────────────


def test_extract_response_text_openai_format() -> None:
    import json

    body = json.dumps(
        {"choices": [{"message": {"role": "assistant", "content": "Hello!"}}]}
    ).encode()
    assert _extract_response_text(body) == "Hello!"


def test_extract_response_text_empty_choices() -> None:
    import json

    body = json.dumps({"choices": []}).encode()
    assert _extract_response_text(body) == ""


def test_extract_response_text_invalid_json() -> None:
    assert _extract_response_text(b"not json") == ""


def test_extract_response_text_null_content() -> None:
    import json

    body = json.dumps({"choices": [{"message": {"role": "assistant", "content": None}}]}).encode()
    assert _extract_response_text(body) == ""


def test_extract_response_text_missing_content_key() -> None:
    import json

    body = json.dumps({"choices": [{"message": {"role": "assistant"}}]}).encode()
    assert _extract_response_text(body) == ""


# ── ABSample ──────────────────────────────────────────────────────────────────


def test_ab_sample_is_frozen() -> None:
    s = ABSample(
        timestamp=time.time(),
        model="gpt-4o",
        compressor="prose",
        original_chars=1000,
        compressed_chars=300,
        quality_score=0.85,
        reference_response_len=200,
        compressed_response_len=190,
    )
    with pytest.raises(AttributeError):
        s.quality_score = 0.99  # type: ignore[misc]


def test_ab_sample_chars_accessible() -> None:
    s = ABSample(
        timestamp=1.0,
        model="gpt-4o-mini",
        compressor="json_smart",
        original_chars=5000,
        compressed_chars=100,
        quality_score=0.92,
        reference_response_len=400,
        compressed_response_len=410,
    )
    assert s.original_chars - s.compressed_chars == 4900


# ── ABMonitor — record_request ────────────────────────────────────────────────


@pytest.fixture
def monitor() -> ABMonitor:
    return ABMonitor(max_samples=10)


def test_record_request_increments_total(monitor: ABMonitor) -> None:
    monitor.record_request(1000, 300, "prose")
    assert monitor.stats()["requests_total"] == 1


def test_record_request_increments_compressed_when_savings(monitor: ABMonitor) -> None:
    monitor.record_request(1000, 300, "prose")
    assert monitor.stats()["requests_compressed"] == 1


def test_record_request_not_compressed_when_equal(monitor: ABMonitor) -> None:
    monitor.record_request(500, 500, "passthrough")
    assert monitor.stats()["requests_compressed"] == 0


def test_record_request_accumulates(monitor: ABMonitor) -> None:
    monitor.record_request(1000, 300, "prose")
    monitor.record_request(500, 500, "passthrough")
    monitor.record_request(2000, 400, "json_smart")
    s = monitor.stats()
    assert s["requests_total"] == 3
    assert s["requests_compressed"] == 2


def test_record_request_chars_saved(monitor: ABMonitor) -> None:
    monitor.record_request(1000, 300, "prose")
    monitor.record_request(800, 200, "prose")
    s = monitor.stats()
    assert s["chars_saved_total"] == (700 + 600)


def test_stats_initial(monitor: ABMonitor) -> None:
    s = monitor.stats()
    assert s["requests_total"] == 0
    assert s["requests_compressed"] == 0
    assert s["chars_saved_total"] == 0
    assert s["tokens_saved_estimate_total"] == 0
    assert s["compression_ratio_mean"] == 1.0
    assert s["ab_samples_total"] == 0


def test_record_request_tokens_saved_estimate(monitor: ABMonitor) -> None:
    monitor.record_request(1000, 300, "prose", tokens_saved_estimate=175)
    assert monitor.stats()["tokens_saved_estimate_total"] == 175


def test_record_request_tokens_saved_default_zero(monitor: ABMonitor) -> None:
    monitor.record_request(1000, 300, "prose")
    assert monitor.stats()["tokens_saved_estimate_total"] == 0


def test_record_request_tokens_saved_accumulates(monitor: ABMonitor) -> None:
    monitor.record_request(1000, 300, "prose", tokens_saved_estimate=100)
    monitor.record_request(500, 200, "prose", tokens_saved_estimate=75)
    assert monitor.stats()["tokens_saved_estimate_total"] == 175


def test_stats_compression_ratio_mean(monitor: ABMonitor) -> None:
    monitor.record_request(1000, 500, "prose")
    s = monitor.stats()
    assert s["compression_ratio_mean"] == pytest.approx(0.5, abs=1e-4)


# ── ABMonitor — record_sample ─────────────────────────────────────────────────


def _make_sample(
    quality: float, compressor: str = "prose", numeric_consistency: float = 1.0
) -> ABSample:
    return ABSample(
        timestamp=time.time(),
        model="gpt-4o",
        compressor=compressor,
        original_chars=1000,
        compressed_chars=300,
        quality_score=quality,
        reference_response_len=200,
        compressed_response_len=180,
        numeric_consistency=numeric_consistency,
    )


def test_quality_report_includes_numeric_consistency() -> None:
    monitor = ABMonitor(max_samples=10)
    monitor.record_sample(_make_sample(0.9, "prose", numeric_consistency=0.5))
    monitor.record_sample(_make_sample(0.8, "prose", numeric_consistency=1.0))
    report = monitor.quality_report()
    assert report["numeric_consistency"]["mean"] == pytest.approx(0.75, abs=1e-4)
    assert report["by_compressor"]["prose"]["mean_numeric_consistency"] == pytest.approx(
        0.75, abs=1e-4
    )


def test_record_sample_increments_len(monitor: ABMonitor) -> None:
    monitor.record_sample(_make_sample(0.9))
    assert len(monitor) == 1


def test_record_sample_evicts_when_full() -> None:
    m = ABMonitor(max_samples=3)
    for i in range(5):
        m.record_sample(_make_sample(float(i) / 4))
    assert len(m) == 3


# ── ABMonitor — quality_report ────────────────────────────────────────────────


def test_quality_report_empty(monitor: ABMonitor) -> None:
    report = monitor.quality_report()
    assert report["samples_total"] == 0
    assert report["quality"] is None
    assert report["chars_saved"] is None
    assert report["by_compressor"] == {}


def test_quality_report_single_sample(monitor: ABMonitor) -> None:
    monitor.record_sample(_make_sample(0.8))
    report = monitor.quality_report()
    assert report["samples_total"] == 1
    assert report["quality"]["mean"] == pytest.approx(0.8, abs=1e-4)
    assert report["quality"]["p10"] == pytest.approx(0.8, abs=1e-4)
    assert report["quality"]["p50"] == pytest.approx(0.8, abs=1e-4)
    assert report["quality"]["p90"] == pytest.approx(0.8, abs=1e-4)


def test_quality_report_percentiles(monitor: ABMonitor) -> None:
    scores = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.4, 0.3, 0.85, 0.75]
    for s in scores:
        monitor.record_sample(_make_sample(s))
    report = monitor.quality_report()
    assert report["quality"]["p10"] < report["quality"]["p50"]
    assert report["quality"]["p50"] < report["quality"]["p90"]


def test_quality_report_by_compressor(monitor: ABMonitor) -> None:
    monitor.record_sample(_make_sample(0.9, "prose"))
    monitor.record_sample(_make_sample(0.7, "prose"))
    monitor.record_sample(_make_sample(0.95, "json_smart"))
    report = monitor.quality_report()
    assert "prose" in report["by_compressor"]
    assert "json_smart" in report["by_compressor"]
    assert report["by_compressor"]["prose"]["samples"] == 2
    assert report["by_compressor"]["json_smart"]["samples"] == 1


def test_quality_report_chars_saved_total(monitor: ABMonitor) -> None:
    for _ in range(3):
        monitor.record_sample(_make_sample(0.9))  # each saves 700 chars
    report = monitor.quality_report()
    assert report["chars_saved"]["total"] == 2100


def test_quality_report_chars_saved_mean(monitor: ABMonitor) -> None:
    monitor.record_sample(_make_sample(0.9))
    report = monitor.quality_report()
    assert report["chars_saved"]["mean"] == pytest.approx(700.0, abs=1e-1)


# ── run_shadow_ab ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_shadow_ab_records_sample() -> None:
    import json
    from unittest.mock import AsyncMock, MagicMock

    from contextly.ab_monitor import run_shadow_ab

    fake_body = json.dumps(
        {"choices": [{"message": {"role": "assistant", "content": "The answer is 42."}}]}
    ).encode()
    mock_resp = MagicMock()
    mock_resp.content = fake_body

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)

    monitor = ABMonitor()
    await run_shadow_ab(
        http_client=mock_client,
        upstream_url="https://api.openai.com/v1/chat/completions",
        headers={"Authorization": "Bearer test"},
        original_payload={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "What is 6*7?"}],
        },
        compressed_response_text="The answer is 42.",
        model="gpt-4o",
        compressor_name="prose",
        original_chars=1000,
        compressed_chars=300,
        ab_monitor=monitor,
    )

    assert len(monitor) == 1
    report = monitor.quality_report()
    assert report["samples_total"] == 1
    assert report["quality"]["mean"] == pytest.approx(1.0, abs=1e-4)


@pytest.mark.asyncio
async def test_run_shadow_ab_handles_upstream_error_silently() -> None:
    from unittest.mock import AsyncMock

    from contextly.ab_monitor import run_shadow_ab

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=Exception("connection refused"))

    monitor = ABMonitor()
    await run_shadow_ab(
        http_client=mock_client,
        upstream_url="https://api.openai.com/v1/chat/completions",
        headers={},
        original_payload={},
        compressed_response_text="some response",
        model="gpt-4o",
        compressor_name="prose",
        original_chars=1000,
        compressed_chars=300,
        ab_monitor=monitor,
    )

    # No sample recorded — error was silently swallowed
    assert len(monitor) == 0


@pytest.mark.asyncio
async def test_run_shadow_ab_partial_overlap_quality() -> None:
    import json
    from unittest.mock import AsyncMock, MagicMock

    from contextly.ab_monitor import run_shadow_ab

    reference = "The model was trained on large datasets with many epochs."
    candidate = "The model learned from datasets using fewer training steps."

    fake_body = json.dumps(
        {"choices": [{"message": {"role": "assistant", "content": reference}}]}
    ).encode()
    mock_resp = MagicMock()
    mock_resp.content = fake_body

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)

    monitor = ABMonitor()
    await run_shadow_ab(
        http_client=mock_client,
        upstream_url="https://example.com",
        headers={},
        original_payload={},
        compressed_response_text=candidate,
        model="gpt-4o-mini",
        compressor_name="prose",
        original_chars=500,
        compressed_chars=200,
        ab_monitor=monitor,
    )

    report = monitor.quality_report()
    score = report["quality"]["mean"]
    assert 0.0 < score < 1.0
