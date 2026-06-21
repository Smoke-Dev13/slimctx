"""Passthrough compressor — returns content completely unchanged.

Used as the M1 default and as the always-available final fallback in
ContentRouter. It is always safe to apply and has zero overhead.
"""

from __future__ import annotations

from contextly.compressors.base import Compressor, CompressResult


class PassthroughCompressor(Compressor):
    """Identity compressor — content passes through unmodified."""

    @property
    def name(self) -> str:
        return "passthrough"

    def should_apply(self, content: str, query: str = "") -> bool:
        return True

    def compress(self, content: str, query: str = "") -> CompressResult:
        length = len(content)
        return CompressResult(
            content=content,
            original_length=length,
            compressed_length=length,
            compressor_name=self.name,
        )
