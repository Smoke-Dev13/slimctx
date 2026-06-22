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


class JsonTableCompressor(Compressor):
    """Lossless columnar encoding of homogeneous JSON object arrays.

    Detection (should_apply): O(1) heuristic — a JSON array containing objects.

    Compression pipeline:
      1. Parse JSON; bail to passthrough unless it is a list of >= 2 dicts.
      2. Require an identical key set across all records (keeps it reversible).
      3. Emit {"_format","fields","rows"} with compact separators.
      4. Use the result only if it is actually shorter than the input.
    """

    @property
    def name(self) -> str:
        return "json_table"

    def should_apply(self, content: str, query: str = "") -> bool:
        """Return True if content looks like a JSON array of objects."""
        stripped = content.lstrip()
        return stripped.startswith("[") and "{" in stripped[:500]

    def compress(self, content: str, query: str = "") -> CompressResult:
        """Rewrite a homogeneous JSON array into a lossless columnar table."""
        original_length = len(content)

        try:
            data: Any = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            return _make_passthrough(content, self.name)

        if not isinstance(data, list) or len(data) < _MIN_RECORDS:
            return _make_passthrough(content, self.name)
        if not all(isinstance(r, dict) for r in data):
            return _make_passthrough(content, self.name)

        records: list[dict[str, Any]] = data
        key_set = set(records[0].keys())
        if not key_set or any(set(r.keys()) != key_set for r in records[1:]):
            # Differing schemas would not round-trip cleanly — leave it alone.
            return _make_passthrough(content, self.name)

        encoded = encode_table(records)
        compressed_content = json.dumps(encoded, separators=(",", ":"), ensure_ascii=False)
        compressed_length = len(compressed_content)

        if compressed_length >= original_length:
            return _make_passthrough(content, self.name)

        logger.info(
            "json_table_compressed",
            records=len(records),
            fields=len(key_set),
            ratio=round(compressed_length / original_length, 3),
        )

        return CompressResult(
            content=compressed_content,
            original_length=original_length,
            compressed_length=compressed_length,
            compressor_name=self.name,
            metadata={
                "records": len(records),
                "fields": len(key_set),
                "lossless": True,
            },
        )
