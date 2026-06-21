"""Benchmark and acceptance tests for JsonSmartCompressor.

Acceptance criteria (run as part of normal test suite):
  - >85% compression on a 1000-record homogeneous JSON array
  - <10% compression on heterogeneous prose (passthrough)

Benchmark tests (require pytest-benchmark) measure throughput at 100 and 1000
records. Mark them with ``pytest.mark.benchmark`` so they can be excluded from
the fast CI run with ``-m 'not benchmark'``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from contextly.compressors.json_smart import JsonSmartCompressor

# ── Test data generators ───────────────────────────────────────────────────────


def _make_homogeneous_records(n: int = 1000) -> list[dict[str, Any]]:
    """Generate *n* user records with bounded categorical diversity.

    Categorical fields: status (3 values) x country (5 values) = 15 combos.
    Numeric fields are uniformly distributed so z-score anomalies don't appear.
    ID fields are high-cardinality and should be skipped by the clustering step.
    """
    statuses = ["active", "inactive", "pending"]
    countries = ["US", "CA", "UK", "DE", "FR"]
    return [
        {
            "id": i + 1,
            "username": f"user{i:05d}",
            "email": f"u{i}@example.com",
            "status": statuses[i % len(statuses)],
            "country": countries[i % len(countries)],
            "age": 25 + (i % 40),
            "balance": 1000.0 + (i % 500),
        }
        for i in range(n)
    ]


def _make_prose(words: int = 500) -> str:
    sentence = "The quick brown fox jumps over the lazy dog near the riverbank. "
    text = sentence * (words // len(sentence.split()) + 1)
    return " ".join(text.split()[:words])


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def compressor() -> JsonSmartCompressor:
    return JsonSmartCompressor()


@pytest.fixture(scope="module")
def homogeneous_1000() -> str:
    return json.dumps(_make_homogeneous_records(1000))


@pytest.fixture(scope="module")
def prose_content() -> str:
    return _make_prose(500)


# ── Acceptance tests (always run) ─────────────────────────────────────────────


def test_acceptance_85_percent_compression_on_1000_homogeneous_records(
    compressor: JsonSmartCompressor,
    homogeneous_1000: str,
) -> None:
    """M3 acceptance gate: homogeneous 1000-record array compresses >85% by bytes."""
    result = compressor.compress(homogeneous_1000, "")
    savings_pct = (1.0 - result.compression_ratio) * 100.0
    assert savings_pct > 85.0, (
        f"Expected >85% byte savings, got {savings_pct:.1f}%. "
        f"Kept {result.metadata.get('selected_records')} of "
        f"{result.metadata.get('total_records')} records."
    )


def test_acceptance_less_than_10_percent_compression_on_prose(
    compressor: JsonSmartCompressor,
    prose_content: str,
) -> None:
    """M3 acceptance gate: prose is not a JSON array, so passthrough applies (<10% savings)."""
    result = compressor.compress(prose_content, "")
    savings_pct = (1.0 - result.compression_ratio) * 100.0
    assert savings_pct < 10.0, (
        f"Expected <10% savings on heterogeneous prose, got {savings_pct:.1f}%"
    )


def test_acceptance_sample_meta_present(
    compressor: JsonSmartCompressor,
    homogeneous_1000: str,
) -> None:
    result = compressor.compress(homogeneous_1000, "")
    output = json.loads(result.content)
    meta_entries = [r for r in output if "_sample_meta" in r]
    assert len(meta_entries) == 1
    meta = meta_entries[0]["_sample_meta"]
    assert meta["total"] == 1000
    assert meta["shown"] < 1000
    assert meta["strategy"] == "stratified+outliers"


def test_acceptance_shown_count_matches_metadata(
    compressor: JsonSmartCompressor,
    homogeneous_1000: str,
) -> None:
    result = compressor.compress(homogeneous_1000, "")
    output = json.loads(result.content)
    meta = next(r["_sample_meta"] for r in output if "_sample_meta" in r)
    real_records = [r for r in output if "_sample_meta" not in r]
    assert meta["shown"] == len(real_records)
    assert result.metadata["selected_records"] == len(real_records)


def test_acceptance_query_aware_aggregate_more_aggressive(
    compressor: JsonSmartCompressor,
    homogeneous_1000: str,
) -> None:
    r_agg = compressor.compress(homogeneous_1000, "count how many users are active")
    r_cons = compressor.compress(homogeneous_1000, "find the user with unusual activity")
    assert r_agg.metadata["selected_records"] <= r_cons.metadata["selected_records"]


# ── Benchmark tests (pytest-benchmark, excluded from fast CI with -m 'not benchmark') ──


@pytest.mark.benchmark
def test_benchmark_compression_1000_records(
    benchmark: Any,
    compressor: JsonSmartCompressor,
    homogeneous_1000: str,
) -> None:
    """Throughput benchmark for 1000-record homogeneous JSON array."""
    result = benchmark(compressor.compress, homogeneous_1000, "")
    assert result.compressor_name == "json_smart"
    assert (1.0 - result.compression_ratio) > 0.85


@pytest.mark.benchmark
def test_benchmark_compression_100_records(
    benchmark: Any,
    compressor: JsonSmartCompressor,
) -> None:
    """Throughput benchmark for 100-record array (typical LLM prompt size)."""
    content = json.dumps(_make_homogeneous_records(100))
    result = benchmark(compressor.compress, content, "")
    assert result.compressor_name == "json_smart"
    # Smaller arrays → still expect meaningful compression
    assert result.compression_ratio < 1.0


@pytest.mark.benchmark
def test_benchmark_should_apply(
    benchmark: Any,
    compressor: JsonSmartCompressor,
    homogeneous_1000: str,
) -> None:
    """should_apply must be O(1) — benchmark confirms it's fast."""
    result = benchmark(compressor.should_apply, homogeneous_1000, "")
    assert result is True
