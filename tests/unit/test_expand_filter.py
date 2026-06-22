"""Unit tests for granular expansion (filter_original)."""

from __future__ import annotations

import json

from contextly.compressors.json_table import encode_table
from contextly.expand import filter_original

_RECORDS = [
    {"id": 8421, "city": "Tbilisi", "status": "active"},
    {"id": 8422, "city": "Reykjavik", "status": "churned"},
    {"id": 8423, "city": "Tbilisi", "status": "trial"},
]
_LOG = "INFO request 1 ok\nERROR timeout on host 5\nINFO request 2 ok\nERROR timeout on host 9"


def test_empty_filter_returns_everything() -> None:
    content, n = filter_original("anything", "")
    assert content == "anything"
    assert n == -1


def test_filter_json_array_by_record() -> None:
    content, n = filter_original(json.dumps(_RECORDS), "8421")
    assert n == 1
    assert json.loads(content) == [_RECORDS[0]]


def test_filter_json_array_multiple_matches() -> None:
    content, n = filter_original(json.dumps(_RECORDS), "Tbilisi")
    assert n == 2
    assert {r["id"] for r in json.loads(content)} == {8421, 8423}


def test_filter_table_form_records() -> None:
    table = json.dumps(encode_table(_RECORDS))
    content, n = filter_original(table, "churned")
    assert n == 1
    assert json.loads(content) == [_RECORDS[1]]


def test_filter_logs_by_line() -> None:
    content, n = filter_original(_LOG, "ERROR")
    assert n == 2
    assert content.splitlines() == [
        "ERROR timeout on host 5",
        "ERROR timeout on host 9",
    ]


def test_filter_is_case_insensitive() -> None:
    _, n = filter_original(_LOG, "error")
    assert n == 2


def test_filter_no_match_returns_empty() -> None:
    content, n = filter_original(_LOG, "nonexistent")
    assert n == 0
    assert content == ""
