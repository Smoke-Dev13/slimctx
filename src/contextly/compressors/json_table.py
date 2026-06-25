"""Lossless columnar compressor for homogeneous JSON arrays.

A JSON array of objects with a shared schema spends most of its tokens on field
names and punctuation repeated in every record:

    [{"id":1000,"name":"user_000","city":"Reykjavik"}, {"id":1001, ...}, ...]

This compressor rewrites such an array into a columnar table where the field
names appear exactly once:

    {"_format":"records-table",
     "_help":"Columnar JSON. Each list in 'rows' is one record; its items align
              by position with 'fields'. So row[i] is the value of fields[i].
              All records are present.",
     "fields":["id","name","city"],
     "rows":[[1000,"user_000","Reykjavik"],[1001, ...], ...]}

The ``_help`` string is a self-describing hint so the model reads the columns
correctly; it carries no data and is ignored on decode. Every record and every
value is preserved — the transform is **lossless** and round-trips exactly — so
an LLM can still answer record-level lookups, unlike the sampling compressor in
``json_smart``. It only applies when all elements are objects sharing an
identical key set (the guarantee that makes it reversible) and when the table
form is actually smaller.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from contextly.compressors.base import Compressor, CompressResult

logger = structlog.get_logger(__name__)

_MIN_RECORDS: int = 2
_FORMAT_TAG: str = "records-table"
_HELP: str = (
    "Columnar JSON. Each list in 'rows' is one record; its items align by "
    "position with 'fields', so row[i] is the value of fields[i]. All records "
    "are present (lossless)."
)


def _make_passthrough(content: str, name: str) -> CompressResult:
    length = len(content)
    return CompressResult(
        content=content,
        original_length=length,
        compressed_length=length,
        compressor_name=name,
    )


def encode_table(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Encode homogeneous *records* into the columnar table structure."""
    fields = list(records[0].keys())
    return {
        "_format": _FORMAT_TAG,
        "_help": _HELP,
        "fields": fields,
        "rows": [[r[f] for f in fields] for r in records],
    }


def decode_table(table: dict[str, Any]) -> list[dict[str, Any]]:
    """Inverse of :func:`encode_table` — restore the original record list."""
    fields = table["fields"]
    return [dict(zip(fields, row, strict=True)) for row in table["rows"]]


def _is_tabular(value: Any) -> bool:
    """True if *value* is a list of >= _MIN_RECORDS dicts with one shared schema."""
    if not isinstance(value, list) or len(value) < _MIN_RECORDS:
        return False
    if not all(isinstance(r, dict) for r in value):
        return False
    key_set = set(value[0].keys())
    return bool(key_set) and all(set(r.keys()) == key_set for r in value[1:])


def _transform(value: Any) -> tuple[Any, int]:
    """Recursively rewrite homogeneous record arrays (even nested in objects) into
    columnar tables. Returns (new_value, number_of_records_tabled)."""
    if _is_tabular(value):
        return encode_table(value), len(value)
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        total = 0
        for k, v in value.items():
            nv, n = _transform(v)
            out[k] = nv
            total += n
        return out, total
    if isinstance(value, list):
        new_list: list[Any] = []
        total = 0
        for item in value:
            ni, n = _transform(item)
            new_list.append(ni)
            total += n
        return new_list, total
    return value, 0


class JsonTableCompressor(Compressor):
    """Lossless columnar encoding of homogeneous JSON object arrays.

    Detection (should_apply): O(1) heuristic — JSON that may contain object arrays.

    Compression pipeline:
      1. Parse JSON; bail to passthrough on parse error.
      2. Recursively rewrite every homogeneous array of dicts — top-level *or
         nested inside an object* (e.g. a paginated API's
         {"list": [...], "pageInfo": ...}) — into a {"_format","fields","rows"}
         table, keeping the surrounding structure intact.
      3. Bail if nothing was tabled or the result is not actually shorter.
    """

    @property
    def name(self) -> str:
        return "json_table"

    def should_apply(self, content: str, query: str = "") -> bool:
        """Return True if content is JSON that may hold an array of objects."""
        stripped = content.lstrip()
        return (stripped.startswith("[") or stripped.startswith("{")) and "{" in stripped[:2000]

    def compress(self, content: str, query: str = "") -> CompressResult:
        """Rewrite homogeneous JSON record arrays into lossless columnar tables."""
        original_length = len(content)

        try:
            data: Any = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            return _make_passthrough(content, self.name)

        transformed, n_records = _transform(data)
        if n_records == 0:
            return _make_passthrough(content, self.name)

        compressed_content = json.dumps(transformed, separators=(",", ":"), ensure_ascii=False)
        compressed_length = len(compressed_content)
        if compressed_length >= original_length:
            return _make_passthrough(content, self.name)

        logger.info(
            "json_table_compressed",
            records=n_records,
            ratio=round(compressed_length / original_length, 3),
        )

        return CompressResult(
            content=compressed_content,
            original_length=original_length,
            compressed_length=compressed_length,
            compressor_name=self.name,
            metadata={
                "records": n_records,
                "lossless": True,
            },
        )
