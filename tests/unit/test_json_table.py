"""Unit tests for the lossless JsonTableCompressor."""

from __future__ import annotations

import json

from contextly.compressors.json_table import (
    JsonTableCompressor,
    decode_table,
    encode_table,
)

_RECORDS = [
    {"id": 1000 + i, "name": f"user_{i:03d}", "city": "Tbilisi", "mrr": i * 7} for i in range(40)
]


def _compress(obj: object) -> object:
    return JsonTableCompressor().compress(json.dumps(obj))


# ── should_apply ────────────────────────────────────────────────────────────────


def test_should_apply_json_array() -> None:
    assert JsonTableCompressor().should_apply('[{"a": 1}]') is True


def test_should_apply_rejects_prose() -> None:
    assert JsonTableCompressor().should_apply("just some text") is False


# ── Losslessness ────────────────────────────────────────────────────────────────


def test_roundtrip_is_lossless() -> None:
    result = _compress(_RECORDS)
    table = json.loads(result.content)
    assert decode_table(table) == _RECORDS


def test_encode_decode_inverse() -> None:
    assert decode_table(encode_table(_RECORDS)) == _RECORDS


def test_preserves_every_record() -> None:
    result = _compress(_RECORDS)
    table = json.loads(result.content)
    assert len(table["rows"]) == len(_RECORDS)
    assert table["fields"] == list(_RECORDS[0].keys())


def test_preserves_value_types() -> None:
    # Check the transform directly: it must round-trip every JSON scalar type.
    records = [{"n": 1, "f": 1.5, "b": True, "s": "x", "z": None} for _ in range(3)]
    assert decode_table(encode_table(records)) == records


# ── Token / size reduction ──────────────────────────────────────────────────────


def test_reduces_size() -> None:
    result = _compress(_RECORDS)
    assert result.compressor_name == "json_table"
    assert result.compressed_length < result.original_length
    assert result.metadata["lossless"] is True


# ── Fallbacks ───────────────────────────────────────────────────────────────────


def test_nested_records_array_is_compressed() -> None:
    # Paginated-API style: records wrapped in an object {"list": [...], "pageInfo": {...}}.
    payload = {"list": _RECORDS, "pageInfo": {"totalRows": len(_RECORDS), "page": 1}}
    result = _compress(payload)
    assert result.compressor_name == "json_table"
    assert result.compressed_length < result.original_length
    out = json.loads(result.content)
    # The surrounding object is preserved; the list became a table.
    assert out["pageInfo"] == {"totalRows": len(_RECORDS), "page": 1}
    assert decode_table(out["list"]) == _RECORDS


def test_should_apply_on_json_object() -> None:
    assert JsonTableCompressor().should_apply('{"list": [{"a": 1}, {"a": 2}]}') is True


def test_object_without_record_arrays_passes_through() -> None:
    result = _compress({"a": 1, "b": {"c": 2}})
    assert result.content == json.dumps({"a": 1, "b": {"c": 2}})


def test_heterogeneous_schema_passes_through() -> None:
    records = [{"a": 1, "b": 2}, {"a": 1, "c": 3}]  # differing key sets
    result = _compress(records)
    assert result.compressor_name == "json_table"
    assert result.content == json.dumps(records)  # unchanged


def test_non_list_passes_through() -> None:
    result = _compress({"a": 1})
    assert result.content == json.dumps({"a": 1})


def test_array_of_scalars_passes_through() -> None:
    result = _compress([1, 2, 3])
    assert result.content == json.dumps([1, 2, 3])


def test_invalid_json_passes_through() -> None:
    result = JsonTableCompressor().compress("{not json")
    assert result.content == "{not json"


def test_single_record_passes_through() -> None:
    # Below the 2-record minimum: the header would not pay for itself.
    result = _compress([{"a": 1, "b": 2}])
    assert result.content == json.dumps([{"a": 1, "b": 2}])
