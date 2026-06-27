"""ML-based prompt compression via LLMLingua-2.

LLMLingua-2 is a token-classification model trained to drop low-information
tokens while preserving meaning — typically achieving higher compression at
equal quality than rule-based extractive methods on prose.

The dependency is heavy (torch + transformers) and therefore **optional**: the
model is imported lazily and cached. When ``llmlingua`` is not installed,
``should_apply`` returns False so the router simply never selects this
compressor and Contextly's zero-dependency default is preserved. Mirrors the
optional-``yake`` pattern in ``compressors/prose.py``.
"""

from __future__ import annotations

import re
import threading
from typing import Any

import structlog

from contextly.compressors.base import Compressor, CompressResult

logger = structlog.get_logger(__name__)

_MIN_CONTENT_LENGTH = 200
_MIN_SENTENCES = 3
_CODE_LINE_RE = re.compile(
    r"^(def |class |import |from \S+ import |function |const |let |var |func |package )",
    re.MULTILINE,
)

_AGGRESSIVE_KEYWORDS: frozenset[str] = frozenset(
    ["summarize", "summary", "brief", "overview", "tldr", "main points", "key points", "shorten"]
)
_CONSERVATIVE_KEYWORDS: frozenset[str] = frozenset(
    ["specific", "exact", "detail", "particular", "find", "identify", "which", "who"]
)

# Compression rate = fraction of tokens to KEEP (LLMLingua convention).
_RATE_AGGRESSIVE = 0.4
_RATE_DEFAULT = 0.6
_RATE_CONSERVATIVE = 0.8


def _rate_from_query(query: str) -> float:
    """Return the LLMLingua keep-rate in [0, 1]; lower keeps fewer tokens."""
    if not query:
        return _RATE_DEFAULT
    q = query.lower()
    if any(kw in q for kw in _AGGRESSIVE_KEYWORDS):
        return _RATE_AGGRESSIVE
    if any(kw in q for kw in _CONSERVATIVE_KEYWORDS):
        return _RATE_CONSERVATIVE
    return _RATE_DEFAULT


class LLMLinguaCompressor(Compressor):
    """Neural extractive compressor backed by LLMLingua-2 (optional dependency).

    Stateless and thread-safe: the underlying model is loaded once behind a lock
    and shared across requests.
    """

    def __init__(
        self,
        model_name: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    ) -> None:
        self._model_name = model_name
        self._model: Any | None = None
        self._load_attempted = False
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "llmlingua"

    def _get_model(self) -> Any | None:
        """Lazily load and cache the LLMLingua-2 model; None if unavailable."""
        if self._load_attempted:
            return self._model
        with self._lock:
            if self._load_attempted:
                return self._model
            self._load_attempted = True
            try:
                from llmlingua import PromptCompressor  # type: ignore[import-not-found]

                self._model = PromptCompressor(
                    model_name=self._model_name,
                    use_llmlingua2=True,
                )
                logger.info("llmlingua_model_loaded", model=self._model_name)
            except Exception:
                logger.info("llmlingua_unavailable", model=self._model_name)
                self._model = None
        return self._model

    def should_apply(self, content: str, query: str = "") -> bool:
        """Apply to long prose when the LLMLingua model is available."""
        if len(content) < _MIN_CONTENT_LENGTH:
            return False
        stripped = content.lstrip()
        if stripped.startswith("[") or stripped.startswith("{"):
            return False
        if len(_CODE_LINE_RE.findall(content)) >= 2:
            return False
        if len(re.findall(r"[.!?]", content)) < _MIN_SENTENCES:
            return False
        return self._get_model() is not None

    def compress(self, content: str, query: str = "") -> CompressResult:
        """Compress prose via LLMLingua-2, falling back to passthrough on error."""
        original_length = len(content)
        model = self._get_model()
        if model is None:
            return CompressResult(
                content=content,
                original_length=original_length,
                compressed_length=original_length,
                compressor_name=self.name,
            )

        rate = _rate_from_query(query)
        try:
            result = model.compress_prompt(content, rate=rate)
            compressed = result["compressed_prompt"] if isinstance(result, dict) else str(result)
        except Exception:
            logger.warning("llmlingua_compress_failed", exc_info=True)
            return CompressResult(
                content=content,
                original_length=original_length,
                compressed_length=original_length,
                compressor_name=self.name,
            )

        compressed_length = len(compressed)
        # Guard against a model returning something larger than the input.
        if compressed_length >= original_length:
            return CompressResult(
                content=content,
                original_length=original_length,
                compressed_length=original_length,
                compressor_name=self.name,
            )

        return CompressResult(
            content=compressed,
            original_length=original_length,
            compressed_length=compressed_length,
            compressor_name=self.name,
            metadata={"rate": rate, "model": self._model_name},
        )
