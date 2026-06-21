"""Unit tests for JsonSmartCompressor and its helper functions."""

from __future__ import annotations

import json

import pytest

from contextly.compressors.json_smart import (
    JsonSmartCompressor,
    _aggressiveness_from_query,
    _analyze_fields,
    _compute_percentile_buckets,
    _find_anomalies,
    _get_bucket,
    _is_homogeneous,
    _target_sample_size,
)

# ── Fixtures and helpers ───────────────────────────────────────────────────────


def _make_records(
    n: int = 50,
    statuses: list[str] | None = None,
    countries: list[str] | None = None,
) -> list[dict]:
    statuses = statuses or ["active", "inactive"]
    countries = countries or ["US", "CA", "UK"]
    return [
        {
            "id": i,
            "username": f"user{i:04d}",
            "status": statuses[i % len(statuses)],
            "country": countries[i % len(countries)],
            "age": 25 + (i % 40),
            "score": 60.0 + (i % 30),
        }
        for i in range(n)
    ]


@pytest.fixture
def compressor() -> JsonSmartCompressor:
    return JsonSmartCompressor()


# ── should_apply ───────────────────────────────────────────────────────────────


def test_should_apply_json_array_of_objects(compressor: JsonSmartCompressor) -> None:
    assert compressor.should_apply(json.dumps([{"a": 1, "b": 2, "c": 3}])) is True


def test_should_apply_rejects_plain_prose(compressor: JsonSmartCompressor) -> None:
    assert compressor.should_apply("The quick brown fox jumps over the lazy dog.") is False


def test_should_apply_rejects_json_object(compressor: JsonSmartCompressor) -> None:
    assert compressor.should_apply('{"key": "value"}') is False


def test_should_apply_rejects_plain_array(compressor: JsonSmartCompressor) -> None:
    assert compressor.should_apply("[1, 2, 3]") is False


def test_should_apply_rejects_empty_string(compressor: JsonSmartCompressor) -> None:
    assert compressor.should_apply("") is False


def test_should_apply_handles_leading_whitespace(compressor: JsonSmartCompressor) -> None:
    assert compressor.should_apply('  \n[{"a": 1, "b": 2, "c": 3}]') is True


# ── compress — passthrough cases ───────────────────────────────────────────────


def test_compress_name(compressor: JsonSmartCompressor) -> None:
    assert compressor.name == "json_smart"


def test_compress_passthrough_for_non_json(compressor: JsonSmartCompressor) -> None:
    result = compressor.compress("not json at all")
    assert result.compression_ratio == pytest.approx(1.0)
    assert result.compressor_name == "json_smart"


def test_compress_passthrough_for_json_object(compressor: JsonSmartCompressor) -> None:
    result = compressor.compress('{"key": "value", "other": 1}')
    assert result.compression_ratio == pytest.approx(1.0)


def test_compress_passthrough_for_small_array(compressor: JsonSmartCompressor) -> None:
    small = json.dumps([{"id": i, "name": f"u{i}", "x": i} for i in range(5)])
    result = compressor.compress(small)
    assert result.compression_ratio == pytest.approx(1.0)


def test_compress_passthrough_for_empty_array(compressor: JsonSmartCompressor) -> None:
    result = compressor.compress("[]")
    assert result.compression_ratio == pytest.approx(1.0)


def test_compress_passthrough_for_non_dict_elements(compressor: JsonSmartCompressor) -> None:
    mixed = json.dumps([1, 2, 3, "str", [4, 5]])
    result = compressor.compress(mixed)
    assert result.compression_ratio == pytest.approx(1.0)


def test_compress_passthrough_when_no_shared_keys(compressor: JsonSmartCompressor) -> None:
    # Records share zero keys
    records = [{"a": i} if i % 2 == 0 else {"b": i} for i in range(20)]
    result = compressor.compress(json.dumps(records))
    assert result.compression_ratio == pytest.approx(1.0)


# ── compress — successful compression ─────────────────────────────────────────


def test_compress_reduces_size(compressor: JsonSmartCompressor) -> None:
    records = _make_records(100)
    result = compressor.compress(json.dumps(records))
    assert result.compression_ratio < 1.0


def test_compress_adds_sample_meta(compressor: JsonSmartCompressor) -> None:
    records = _make_records(50)
    result = compressor.compress(json.dumps(records))
    output = json.loads(result.content)
    meta_entries = [r for r in output if "_sample_meta" in r]
    assert len(meta_entries) == 1
    meta = meta_entries[0]["_sample_meta"]
    assert meta["total"] == 50
    assert "shown" in meta
    assert meta["strategy"] == "stratified+outliers"


def test_compress_shown_less_than_total(compressor: JsonSmartCompressor) -> None:
    records = _make_records(200)
    result = compressor.compress(json.dumps(records))
    output = json.loads(result.content)
    meta = next(r["_sample_meta"] for r in output if "_sample_meta" in r)
    assert meta["shown"] < meta["total"]


def test_compress_result_records_are_valid_dicts(compressor: JsonSmartCompressor) -> None:
    records = _make_records(80)
    result = compressor.compress(json.dumps(records))
    output = json.loads(result.content)
    real_records = [r for r in output if "_sample_meta" not in r]
    assert all(isinstance(r, dict) for r in real_records)


def test_compress_metadata_keys_present(compressor: JsonSmartCompressor) -> None:
    records = _make_records(60)
    result = compressor.compress(json.dumps(records))
    expected_keys = (
        "total_records",
        "selected_records",
        "anomaly_count",
        "cluster_count",
        "aggressiveness",
    )
    for key in expected_keys:
        assert key in result.metadata, f"missing metadata key: {key}"


def test_compress_metadata_totals_match(compressor: JsonSmartCompressor) -> None:
    records = _make_records(80)
    result = compressor.compress(json.dumps(records))
    assert result.metadata["total_records"] == 80
    output = json.loads(result.content)
    real_records = [r for r in output if "_sample_meta" not in r]
    assert result.metadata["selected_records"] == len(real_records)


# ── Anomaly preservation ───────────────────────────────────────────────────────


def test_compress_preserves_numeric_anomaly() -> None:
    """A record with extreme z-score must appear in the compressed output."""
    compressor = JsonSmartCompressor()
    # 99 normal records, age centred on 30
    records: list[dict] = [
        {"id": i, "status": "active", "country": "US", "age": 30, "score": 80.0} for i in range(99)
    ]
    # One extreme outlier: age 200 (z >> 2)
    records.append({"id": 9999, "status": "active", "country": "US", "age": 200, "score": 80.0})

    result = compressor.compress(json.dumps(records))
    output = json.loads(result.content)
    ids = {r["id"] for r in output if "_sample_meta" not in r}
    assert 9999 in ids, "Numeric outlier must be preserved in compressed output"


def test_compress_preserves_rare_category_anomaly() -> None:
    """A record with a rare categorical value must not be dropped."""
    compressor = JsonSmartCompressor()
    records: list[dict] = [
        {"id": i, "status": "active", "country": "US", "age": 30, "score": 80.0} for i in range(98)
    ]
    # Rare country appearing in < 5% of records (1/100 = 1%)
    records.append({"id": 8888, "status": "active", "country": "ZZ", "age": 30, "score": 80.0})
    records.append({"id": 7777, "status": "active", "country": "US", "age": 30, "score": 80.0})

    result = compressor.compress(json.dumps(records))
    output = json.loads(result.content)
    ids = {r["id"] for r in output if "_sample_meta" not in r}
    assert 8888 in ids, "Rare categorical value must be preserved"


# ── Query-aware aggressiveness ─────────────────────────────────────────────────


def test_compress_aggressive_query_keeps_fewer_records(compressor: JsonSmartCompressor) -> None:
    records = _make_records(200)
    content = json.dumps(records)

    result_agg = compressor.compress(content, query="count how many users are active")
    result_cons = compressor.compress(content, query="find the user with unusual activity")

    def _shown(r: object) -> int:
        output = json.loads(r.content)  # type: ignore[union-attr]
        return next(x["_sample_meta"]["shown"] for x in output if "_sample_meta" in x)

    assert _shown(result_agg) <= _shown(result_cons)


def test_compress_default_aggressiveness_is_mid(compressor: JsonSmartCompressor) -> None:
    records = _make_records(100)
    result = compressor.compress(json.dumps(records))
    assert result.metadata["aggressiveness"] == pytest.approx(0.7)


# ── Helper: _aggressiveness_from_query ─────────────────────────────────────────


def test_agg_query_aggressive() -> None:
    assert _aggressiveness_from_query("count how many users") > 0.7


def test_agg_query_conservative() -> None:
    assert _aggressiveness_from_query("find the unusual user") < 0.5


def test_agg_query_empty_is_default() -> None:
    assert _aggressiveness_from_query("") == pytest.approx(0.7)


def test_agg_query_generic_is_default() -> None:
    assert _aggressiveness_from_query("show me some records") == pytest.approx(0.7)


# ── Helper: _target_sample_size ───────────────────────────────────────────────


def test_target_sample_size_scales_inversely_with_aggressiveness() -> None:
    low = _target_sample_size(1000, 0.2)
    high = _target_sample_size(1000, 0.9)
    assert low > high


def test_target_sample_size_minimum_floor() -> None:
    assert _target_sample_size(1000, 1.0) >= 3


def test_target_sample_size_small_array() -> None:
    assert _target_sample_size(10, 0.5) >= 3


# ── Helper: _compute_percentile_buckets / _get_bucket ─────────────────────────


def test_percentile_buckets_length() -> None:
    vals = list(map(float, range(100)))
    buckets = _compute_percentile_buckets(vals, 5)
    assert len(buckets) == 4


def test_percentile_buckets_empty() -> None:
    assert _compute_percentile_buckets([], 5) == []


def test_get_bucket_low_value() -> None:
    thresholds = [10.0, 20.0, 30.0, 40.0]
    assert _get_bucket(5.0, thresholds) == 0


def test_get_bucket_high_value() -> None:
    thresholds = [10.0, 20.0, 30.0, 40.0]
    assert _get_bucket(99.0, thresholds) == 4


def test_get_bucket_boundary() -> None:
    thresholds = [10.0, 20.0]
    assert _get_bucket(10.0, thresholds) == 0
    assert _get_bucket(10.1, thresholds) == 1


# ── Helper: _is_homogeneous ───────────────────────────────────────────────────


def test_is_homogeneous_true() -> None:
    records = [{"a": 1, "b": 2, "c": 3}, {"a": 4, "b": 5, "c": 6}]
    assert _is_homogeneous(records) is True


def test_is_homogeneous_false_no_shared_keys() -> None:
    records = [{"a": 1, "b": 2}, {"c": 3, "d": 4}]
    assert _is_homogeneous(records) is False


def test_is_homogeneous_empty() -> None:
    assert _is_homogeneous([]) is False


def test_is_homogeneous_requires_min_3_keys() -> None:
    # Only 2 shared keys → not considered homogeneous
    records = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    assert _is_homogeneous(records) is False


# ── Helper: _find_anomalies ────────────────────────────────────────────────────


def test_find_anomalies_zscore() -> None:
    records: list[dict] = [{"age": 30, "score": 80} for _ in range(98)]
    records.append({"age": 500, "score": 80})  # z >> 2
    records.append({"age": -200, "score": 80})
    stats = _analyze_fields(records)
    anomalies = _find_anomalies(records, stats)
    assert 98 in anomalies
    assert 99 in anomalies


def test_find_anomalies_none_for_uniform_data() -> None:
    records = [{"x": 5.0, "y": 10.0} for _ in range(50)]
    stats = _analyze_fields(records)
    anomalies = _find_anomalies(records, stats)
    assert len(anomalies) == 0


def test_find_anomalies_rare_category() -> None:
    records: list[dict] = [{"cat": "A", "num": 1.0} for _ in range(99)]
    records.append({"cat": "RARE_ONE", "num": 1.0})  # freq = 1%
    stats = _analyze_fields(records)
    anomalies = _find_anomalies(records, stats)
    assert 99 in anomalies


# ── Helper: _analyze_fields ───────────────────────────────────────────────────


def test_analyze_fields_numeric_type() -> None:
    records = [{"x": float(i)} for i in range(50)]
    stats = _analyze_fields(records)
    assert stats["x"]["type"] == "numeric"
    assert "mean" in stats["x"]
    assert "stdev" in stats["x"]
    assert "buckets" in stats["x"]


def test_analyze_fields_categorical_type() -> None:
    records = [{"cat": "A" if i % 2 == 0 else "B"} for i in range(50)]
    stats = _analyze_fields(records)
    assert stats["cat"]["type"] == "categorical"
    assert "freqs" in stats["cat"]


def test_analyze_fields_high_cardinality() -> None:
    # Every record has a unique string value → high cardinality
    records = [{"uid": f"id_{i}", "val": 1.0} for i in range(50)]
    stats = _analyze_fields(records)
    assert stats["uid"]["type"] == "high_cardinality"


def test_analyze_fields_boolean_type() -> None:
    records = [{"active": True}, {"active": False}] * 25
    stats = _analyze_fields(records)
    assert stats["active"]["type"] == "boolean"
