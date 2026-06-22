"""Granular expansion of CCR-stored originals.

``expand`` recovers the full original behind a compression reference, but an
agent usually needs just one record or one log pattern — not the whole payload.
:func:`filter_original` narrows a stored original to the parts that match a
substring, so the agent pulls back only what it asked for:

  * JSON array  → the records whose serialization contains the substring
  * table form  → decoded to records first, then filtered the same way
  * plain text / logs → the lines that contain the substring

An empty filter returns the original unchanged.
"""

from __future__ import annotations

import json
from typing import Any

from contextly.compressors.json_table import decode_table


def _records_of(parsed: Any) -> list[dict[str, Any]] | None:
    """Return a record list from a JSON array or a json_table payload, else None."""
    if isinstance(parsed, list) and all(isinstance(r, dict) for r in parsed):
        return parsed
    if isinstance(parsed, dict) and isinstance(parsed.get("rows"), list):
        try:
            return decode_table(parsed)
        except (KeyError, TypeError, ValueError):
            return None
    return None


def filter_original(content: str, contains: str) -> tuple[str, int]:
    """Filter *content* to the parts matching *contains* (case-insensitive).

    Args:
        content: The full original recovered from the CCR store.
        contains: Substring to match. Empty means "return everything".

    Returns:
        ``(filtered_content, n_matches)``. ``n_matches`` is -1 when no filter was
        applied (empty substring), otherwise the number of matching records/lines.
    """
    if not contains:
        return content, -1

    needle = contains.lower()

    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        parsed = None

    if parsed is not None:
        records = _records_of(parsed)
        if records is not None:
            matches = [r for r in records if needle in json.dumps(r).lower()]
            return json.dumps(matches), len(matches)

    lines = [ln for ln in content.splitlines() if needle in ln.lower()]
    return "\n".join(lines), len(lines)
