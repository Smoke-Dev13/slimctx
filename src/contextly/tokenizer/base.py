"""Abstract base for all tokenizer implementations.

Tokenizers are used by compressors to measure payload size before and after
compression, enabling token-budget enforcement and accurate reporting in
CompressResult.

Both BundledTiktokenTokenizer and WordTokenizer implement this interface so
compressors can remain tokenizer-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Tokenizer(ABC):
    """Abstract tokenizer interface.

    Implementations must be thread-safe and stateless after construction.
    """

    @abstractmethod
    def count(self, text: str) -> int:
        """Return the number of tokens in the given text.

        Args:
            text: The text to tokenize and count.

        Returns:
            Non-negative integer token count.
        """
        ...

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Return the token IDs for the given text.

        Args:
            text: The text to tokenize.

        Returns:
            List of integer token IDs.
        """
        ...

    @property
    @abstractmethod
    def encoding_name(self) -> str:
        """Human-readable encoding name, e.g. 'cl100k_base' or 'word'."""
        ...
