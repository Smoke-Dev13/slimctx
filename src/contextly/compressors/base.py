"""Abstract base for all content compressors.

Every compressor implements two methods:

  should_apply(content, query) -> bool
      Cheap content-sniff with no I/O. Used by ContentRouter to select the
      best compressor without committing to expensive work.

  compress(content, query) -> CompressResult
      The actual compression. May be CPU-intensive; the router runs this via
      asyncio.to_thread if the compressor declares slow=True (added in M3).

Compressors must be stateless and thread-safe — a single instance is shared
across all concurrent requests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CompressResult:
    """Immutable result of a compression operation."""

    content: str
    original_length: int
    compressed_length: int
    compressor_name: str
    metadata: dict[str, Any] = field(default_factory=dict)
    # Token counts populated by compressors that use a Tokenizer (M3+).
    original_tokens: int | None = None
    compressed_tokens: int | None = None

    @property
    def compression_ratio(self) -> float:
        """Fraction of original size retained (1.0 = no compression)."""
        if self.original_length == 0:
            return 1.0
        return self.compressed_length / self.original_length

    @property
    def tokens_saved_estimate(self) -> int:
        """Saved tokens: exact if token counts are available, else char heuristic."""
        if self.original_tokens is not None and self.compressed_tokens is not None:
            return max(0, self.original_tokens - self.compressed_tokens)
        return max(0, (self.original_length - self.compressed_length) // 4)


class Compressor(ABC):
    """Abstract compressor interface.

    Implementations must be stateless and thread-safe — no instance-level
    mutable state, no writes to shared data structures.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier used in metrics, logs, and CompressResult."""
        ...

    @abstractmethod
    def should_apply(self, content: str, query: str = "") -> bool:
        """Return True if this compressor is appropriate for the given content.

        Must be fast — no I/O, regex at most. Called for every message before
        commit to compression.

        Args:
            content: Text to potentially compress.
            query: The user's last message, used as an aggressiveness hint.

        Returns:
            True if this compressor should handle the content.
        """
        ...

    @abstractmethod
    def compress(self, content: str, query: str = "") -> CompressResult:
        """Compress content, optionally guided by the user's query.

        Args:
            content: Text to compress.
            query: Hint from the user's last message. Compressors may use this
                   to decide aggressiveness — "count users" → aggressive,
                   "find unusual activity" → preserve outliers.

        Returns:
            CompressResult with the (possibly unchanged) content and metadata.
        """
        ...
