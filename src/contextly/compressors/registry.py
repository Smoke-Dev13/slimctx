"""Content router — selects the most appropriate compressor for a given input.

The router calls each registered compressor's should_apply() (cheap) in priority
order and returns the first match. PassthroughCompressor is always the final
fallback and cannot be overridden.

For M1, only PassthroughCompressor is active. Additional compressors are
registered at startup in M3 (JSON), M4 (prose, code).
"""

from __future__ import annotations

import structlog

from contextly.compressors.base import Compressor
from contextly.compressors.passthrough import PassthroughCompressor

logger = structlog.get_logger(__name__)

_PASSTHROUGH = PassthroughCompressor()


class ContentRouter:
    """Selects the best compressor for a given content string.

    Compressors are checked in registration order; the first whose
    should_apply() returns True is used. PassthroughCompressor is always
    appended as the final fallback.
    """

    def __init__(self) -> None:
        self._compressors: list[Compressor] = []

    def register(self, compressor: Compressor) -> None:
        """Register a compressor into the routing chain.

        Must be called at startup, not per-request. Compressors are evaluated
        in the order they are registered.

        Args:
            compressor: A Compressor implementation to add.
        """
        self._compressors.append(compressor)
        logger.debug("compressor_registered", name=compressor.name)

    def select(self, content: str, query: str = "") -> Compressor:
        """Return the first matching compressor for the given content.

        Args:
            content: Text content to route.
            query: The user's last query, forwarded to should_apply.

        Returns:
            A Compressor instance. Never None — falls back to passthrough.
        """
        for compressor in self._compressors:
            try:
                if compressor.should_apply(content, query):
                    logger.debug("compressor_selected", name=compressor.name)
                    return compressor
            except Exception:
                logger.warning(
                    "compressor_should_apply_error",
                    name=compressor.name,
                    exc_info=True,
                )
        return _PASSTHROUGH
