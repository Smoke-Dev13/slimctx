"""WordTokenizer — zero-dependency fallback for unknown models.

Splits text on whitespace and punctuation boundaries. Deliberately simple:
no regex-Unicode extensions, no external libraries, no network access ever.
Used when the requested model has no bundled tiktoken encoding.

Accuracy: under-counts relative to real BPE tokenizers (no subword splitting)
but the count is conservative — callers that use it to enforce a token budget
will be slightly permissive rather than aggressively truncating.
"""

from __future__ import annotations

import re

from contextly.tokenizer.base import Tokenizer

# Matches words (sequence of word chars) OR punctuation groups (non-word,
# non-space). Whitespace is consumed but not counted as a token.
_SPLIT_RE = re.compile(r"\w+|[^\w\s]+")


class WordTokenizer(Tokenizer):
    """Whitespace-and-punctuation tokenizer, usable without any LLM dependency.

    Token boundaries: word characters form a single token; each run of
    punctuation/symbols forms a single token; whitespace is discarded.

    Examples:
        "Hello, world!"  → ["Hello", ",", "world", "!"]  → 4 tokens
        "x=1+2"          → ["x", "=", "1", "+", "2"]     → 5 tokens
    """

    @property
    def encoding_name(self) -> str:
        return "word"

    def count(self, text: str) -> int:
        """Return number of whitespace/punctuation-split tokens.

        Args:
            text: Input text.

        Returns:
            Non-negative integer.
        """
        return len(_SPLIT_RE.findall(text))

    def encode(self, text: str) -> list[int]:
        """Return per-token hash values (stable but not interoperable with BPE).

        Args:
            text: Input text.

        Returns:
            List of integer hashes, one per token.
        """
        return [hash(tok) & 0xFFFF_FFFF for tok in _SPLIT_RE.findall(text)]
