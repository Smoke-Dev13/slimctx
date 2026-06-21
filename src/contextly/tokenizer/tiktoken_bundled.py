"""Offline tiktoken tokenizer backed by encoding files bundled in package_data.

Uses importlib.resources to read .tiktoken BPE files from the
contextly/tokenizer/data/ directory (included in the wheel via hatchling's
default package-data inclusion). Constructs tiktoken.Encoding directly from
the pre-loaded BPE ranks — zero network access on first request.

Supported encodings (bundled at wheel build time via scripts/download_encodings.py):
  - cl100k_base  (GPT-4, GPT-3.5-turbo, text-embedding-ada-002, …)
  - o200k_base   (GPT-4o, o1, o3, …)
"""

from __future__ import annotations

import base64
import importlib.resources
from typing import Any

import tiktoken

from contextly.tokenizer.base import Tokenizer

# Encoding parameters sourced from tiktoken's own registry module.
# These are stable; update only when OpenAI releases a new encoding family.
_ENCODING_PARAMS: dict[str, dict[str, Any]] = {
    "cl100k_base": {
        "pat_str": (
            r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)"""
            r"""|[^\r\n\p{L}\p{N}]?\p{L}+"""
            r"""|\p{N}{1,3}"""
            r"""| ?[^\s\p{L}\p{N}]+[\r\n]*"""
            r"""|\s*[\r\n]+"""
            r"""|\s+(?!\S)"""
            r"""|\s+"""
        ),
        "special_tokens": {
            "<|endoftext|>": 100257,
            "<|fim_prefix|>": 100258,
            "<|fim_middle|>": 100259,
            "<|fim_suffix|>": 100260,
            "<|endofprompt|>": 100276,
        },
    },
    "o200k_base": {
        "pat_str": (
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{Mn}*]*[\p{Ll}\p{Lm}\p{Lo}\p{Mn}*]+"""
            r"""|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{Mn}*]+[\p{Ll}\p{Lm}\p{Lo}\p{Mn}*]*"""
            r"""|\p{N}{1,3}"""
            r"""| ?[^\s\p{L}\p{N}]+[\r\n/]*"""
            r"""|\s*[\r\n]+"""
            r"""|\s+(?!\S)"""
            r"""|\s+"""
        ),
        "special_tokens": {
            "<|endoftext|>": 199999,
            "<|endofprompt|>": 200018,
        },
    },
}


def _parse_tiktoken_bpe(data: bytes) -> dict[bytes, int]:
    """Parse the .tiktoken BPE file format into a rank dictionary.

    File format: each non-empty line is ``<base64-token> <integer-rank>``.

    Args:
        data: Raw bytes of a .tiktoken encoding file.

    Returns:
        Mapping from token bytes to integer rank.
    """
    ranks: dict[bytes, int] = {}
    for line in data.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        token_b64, rank_str = stripped.split()
        ranks[base64.b64decode(token_b64)] = int(rank_str)
    return ranks


def is_bundled(encoding_name: str) -> bool:
    """Return True if the encoding file exists in the package data directory.

    Args:
        encoding_name: One of the supported encoding names.

    Returns:
        True if the .tiktoken file is available without network access.
    """
    try:
        pkg = importlib.resources.files("contextly.tokenizer") / "data"
        return (pkg / f"{encoding_name}.tiktoken").is_file()
    except (FileNotFoundError, TypeError):
        return False


class BundledTiktokenTokenizer(Tokenizer):
    """Tokenizer that loads BPE ranks from bundled package_data files.

    Construction reads and parses the .tiktoken file exactly once; subsequent
    calls to count() and encode() are purely in-memory operations.

    Args:
        encoding_name: Either 'cl100k_base' or 'o200k_base'.

    Raises:
        KeyError: If encoding_name is not in the supported set.
        FileNotFoundError: If the bundled .tiktoken file is missing (run
            ``python scripts/download_encodings.py`` to populate it).
    """

    def __init__(self, encoding_name: str) -> None:
        if encoding_name not in _ENCODING_PARAMS:
            raise KeyError(
                f"Unsupported encoding: {encoding_name!r}. Supported: {list(_ENCODING_PARAMS)}"
            )
        params = _ENCODING_PARAMS[encoding_name]
        data_file = (
            importlib.resources.files("contextly.tokenizer") / "data" / f"{encoding_name}.tiktoken"
        )
        raw: bytes = data_file.read_bytes()
        ranks = _parse_tiktoken_bpe(raw)
        self._enc = tiktoken.Encoding(
            name=encoding_name,
            pat_str=params["pat_str"],
            mergeable_ranks=ranks,
            special_tokens=params["special_tokens"],
        )
        self._encoding_name = encoding_name

    @property
    def encoding_name(self) -> str:
        return self._encoding_name

    def count(self, text: str) -> int:
        """Count tokens using the bundled BPE encoding.

        Args:
            text: Input text to count.

        Returns:
            Number of tokens.
        """
        return len(self._enc.encode(text, disallowed_special=()))

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Input text.

        Returns:
            List of integer token IDs.
        """
        return self._enc.encode(text, disallowed_special=())
