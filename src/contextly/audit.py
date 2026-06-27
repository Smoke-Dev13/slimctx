"""Append-only JSONL compression audit log.

When ``audit_log_path`` is set in config, every compression event is recorded
as one JSON line: original hash, compressor, savings, model, and timestamp.

This gives teams a compliance artifact: an immutable record of what was sent
to which LLM, in what compressed form, and how to recover the original via
the CCR key. Useful for HIPAA/SOC2/GDPR-adjacent workloads.

CLI replay::

    contextly audit replay <log_file>

reads the JSONL and fetches originals from the CCR store to reconstruct each
conversation as it was before compression.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any


class AuditWriter:
    """Thread-safe append-only JSONL writer for compression audit events.

    Each record written to the log has the form::

        {
          "ts": 1234567890.123,
          "request_id": "...",
          "model": "gpt-4o",
          "msg_index": "msg:0",
          "ccr_key": "a1b2c3d4e5f6a7b8",
          "compressor": "json_table",
          "original_chars": 4096,
          "compressed_chars": 1800,
          "ratio": 0.44,
          "deduped": false
        }

    Args:
        path: File path for the audit log. Parent directories are created on
            first write. The file is opened in append mode; existing records
            are preserved across restarts.
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._fh: Any = None
        self._request_id: str = ""

    def new_request(self) -> str:
        """Generate and store a fresh request ID for one request lifecycle.

        Returns:
            The new request ID string.
        """
        self._request_id = str(uuid.uuid4())[:8]
        return self._request_id

    def _ensure_open(self) -> None:
        if self._fh is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self._path.open("a", encoding="utf-8")

    def record(
        self,
        *,
        model: str,
        msg_index: str,
        ccr_key: str | None,
        compressor: str,
        original_chars: int,
        compressed_chars: int,
        deduped: bool = False,
    ) -> None:
        """Append one compression event to the log.

        Args:
            model: LLM model identifier for this request.
            msg_index: Message slot identifier (e.g. ``"msg:0"``).
            ccr_key: CCR key where the original is stored, or None if lossless.
            compressor: Name of the compressor used.
            original_chars: Original content length in characters.
            compressed_chars: Compressed content length in characters.
            deduped: True when this block was a cross-message duplicate.
        """
        ratio = round(compressed_chars / original_chars, 4) if original_chars > 0 else 1.0
        record = {
            "ts": round(time.time(), 3),
            "request_id": self._request_id,
            "model": model,
            "msg_index": msg_index,
            "ccr_key": ccr_key,
            "compressor": compressor,
            "original_chars": original_chars,
            "compressed_chars": compressed_chars,
            "ratio": ratio,
            "deduped": deduped,
        }
        line = json.dumps(record, separators=(",", ":")) + "\n"
        with self._lock:
            self._ensure_open()
            self._fh.write(line)
            self._fh.flush()

    def close(self) -> None:
        """Flush and close the underlying file handle."""
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


def replay(log_path: str) -> list[dict[str, Any]]:
    """Read an audit log and return all records as a list of dicts.

    Args:
        log_path: Path to the JSONL audit log file.

    Returns:
        List of audit record dicts, in order of recording.

    Raises:
        FileNotFoundError: If the log file does not exist.
    """
    path = Path(log_path)
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
