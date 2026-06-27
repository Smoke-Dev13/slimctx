"""Unit tests for AuditWriter and replay()."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contextly.audit import AuditWriter, replay


def test_new_request_returns_string(tmp_path: Path) -> None:
    w = AuditWriter(str(tmp_path / "audit.jsonl"))
    rid = w.new_request()
    assert isinstance(rid, str) and len(rid) == 8


def test_record_writes_jsonl(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    w = AuditWriter(str(log))
    w.new_request()
    w.record(
        model="gpt-4o",
        msg_index="msg:0",
        ccr_key="abc123",
        compressor="prose",
        original_chars=200,
        compressed_chars=100,
    )
    w.close()
    lines = log.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["model"] == "gpt-4o"
    assert rec["compressor"] == "prose"
    assert rec["original_chars"] == 200
    assert rec["compressed_chars"] == 100
    assert rec["ratio"] == pytest.approx(0.5)
    assert rec["deduped"] is False
    assert rec["ccr_key"] == "abc123"


def test_record_deduped_flag(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    w = AuditWriter(str(log))
    w.new_request()
    w.record(
        model="gpt-4o",
        msg_index="msg:1",
        ccr_key=None,
        compressor="passthrough",
        original_chars=50,
        compressed_chars=50,
        deduped=True,
    )
    w.close()
    rec = json.loads(log.read_text().strip())
    assert rec["deduped"] is True
    assert rec["ccr_key"] is None


def test_record_zero_original_chars(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    w = AuditWriter(str(log))
    w.new_request()
    w.record(
        model="gpt-4o",
        msg_index="msg:0",
        ccr_key=None,
        compressor="passthrough",
        original_chars=0,
        compressed_chars=0,
    )
    w.close()
    rec = json.loads(log.read_text().strip())
    assert rec["ratio"] == 1.0


def test_record_creates_parent_dirs(tmp_path: Path) -> None:
    log = tmp_path / "nested" / "dir" / "audit.jsonl"
    w = AuditWriter(str(log))
    w.new_request()
    w.record(
        model="m",
        msg_index="msg:0",
        ccr_key=None,
        compressor="c",
        original_chars=10,
        compressed_chars=5,
    )
    w.close()
    assert log.exists()


def test_multiple_records(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    w = AuditWriter(str(log))
    w.new_request()
    for i in range(3):
        w.record(
            model="gpt-4o",
            msg_index=f"msg:{i}",
            ccr_key=f"key{i}",
            compressor="prose",
            original_chars=100,
            compressed_chars=60,
        )
    w.close()
    lines = log.read_text().strip().splitlines()
    assert len(lines) == 3


def test_replay_reads_records(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    w = AuditWriter(str(log))
    w.new_request()
    w.record(
        model="gpt-4o",
        msg_index="msg:0",
        ccr_key="k1",
        compressor="prose",
        original_chars=200,
        compressed_chars=80,
    )
    w.record(
        model="gpt-4o",
        msg_index="msg:1",
        ccr_key="k2",
        compressor="json_table",
        original_chars=500,
        compressed_chars=300,
    )
    w.close()
    records = replay(str(log))
    assert len(records) == 2
    assert records[0]["ccr_key"] == "k1"
    assert records[1]["ccr_key"] == "k2"


def test_replay_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        replay(str(tmp_path / "nonexistent.jsonl"))


def test_close_idempotent(tmp_path: Path) -> None:
    w = AuditWriter(str(tmp_path / "audit.jsonl"))
    w.new_request()
    w.record(
        model="m",
        msg_index="msg:0",
        ccr_key=None,
        compressor="c",
        original_chars=10,
        compressed_chars=5,
    )
    w.close()
    w.close()  # should not raise
