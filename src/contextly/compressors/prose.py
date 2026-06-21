"""Extractive prose compressor using YAKE keyword-based sentence scoring.

Algorithm:
  1. Extract keywords with YAKE (falls back to word-frequency if yake absent).
  2. Score each sentence by the sum of keyword importances it contains,
     normalised by sqrt(word count) to avoid bias toward long sentences.
  3. Greedily select top-scored sentences until the target character budget
     is filled, then re-order selections back to their original positions.

Query-aware aggressiveness mirrors the json_smart convention:
  "summarize / brief / overview" → 0.9 (few sentences)
  "specific / exact / detail"    → 0.3 (keep more)
  (default)                      → 0.7
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import structlog

from contextly.compressors.base import Compressor, CompressResult

logger = structlog.get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_MIN_CONTENT_LENGTH: int = 200
_MIN_SENTENCES: int = 3
_WORD_RE = re.compile(r"\b[a-zA-Z]{3,}\b")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Code markers that, if present at ≥2 line starts, indicate code not prose.
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


# ── Helpers ───────────────────────────────────────────────────────────────────


def _split_sentences(text: str) -> list[str]:
    """Split *text* into non-trivial sentences (length > 15 chars)."""
    parts = _SENT_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 15]


def _word_freq_keywords(text: str, top_n: int) -> dict[str, float]:
    """Fallback keyword weights: term frequency normalised to [0, 1]."""
    words = _WORD_RE.findall(text.lower())
    if not words:
        return {}
    freq: Counter[str] = Counter(words)
    max_f = freq.most_common(1)[0][1]
    return {w: c / max_f for w, c in freq.most_common(top_n)}


def _extract_keywords(text: str, top_n: int) -> dict[str, float]:
    """Return {keyword: importance_weight} where 1.0 = most important.

    Uses YAKE if installed; falls back to word-frequency scoring.
    YAKE's raw scores are inverted (lower score = more important in YAKE).
    """
    try:
        import yake

        extractor: Any = yake.KeywordExtractor(lan="en", n=1, top=top_n, dedupLim=0.7)
        raw: list[tuple[str, float]] = extractor.extract_keywords(text)
        if not raw:
            return _word_freq_keywords(text, top_n)
        scores = [s for _, s in raw]
        min_s, max_s = min(scores), max(scores)
        rng = max_s - min_s if max_s != min_s else 1.0
        return {kw.lower(): 1.0 - (s - min_s) / rng for kw, s in raw}
    except ImportError:
        return _word_freq_keywords(text, top_n)


def _score_sentence(sentence: str, keyword_weights: dict[str, float]) -> float:
    """Score a sentence by keyword coverage, normalised for length."""
    words = _WORD_RE.findall(sentence.lower())
    if not words:
        return 0.0
    total = sum(keyword_weights.get(w, 0.0) for w in words)
    return total / math.sqrt(len(words))


def _aggressiveness_from_query(query: str) -> float:
    """Return aggressiveness in [0, 1]. Higher → fewer sentences kept."""
    if not query:
        return 0.7
    q = query.lower()
    if any(kw in q for kw in _AGGRESSIVE_KEYWORDS):
        return 0.9
    if any(kw in q for kw in _CONSERVATIVE_KEYWORDS):
        return 0.3
    return 0.7


def _make_passthrough(content: str, name: str) -> CompressResult:
    length = len(content)
    return CompressResult(
        content=content,
        original_length=length,
        compressed_length=length,
        compressor_name=name,
    )


# ── Compressor ────────────────────────────────────────────────────────────────


class ProseCompressor(Compressor):
    """Extractive summarisation of natural-language prose via keyword scoring.

    Detection (should_apply): accepts long text with sentence structure;
    rejects JSON arrays/objects and content that looks like code.

    Compression pipeline:
      1. Split into sentences.
      2. Extract keywords (YAKE or word-frequency fallback).
      3. Score each sentence against the keyword weights.
      4. Greedily select top-scored sentences up to the character budget.
      5. Re-emit sentences in original order.
    """

    @property
    def name(self) -> str:
        return "prose"

    def should_apply(self, content: str, query: str = "") -> bool:
        """Return True if content is long, sentence-like prose (not JSON or code)."""
        if len(content) < _MIN_CONTENT_LENGTH:
            return False
        stripped = content.lstrip()
        if stripped.startswith("[") or stripped.startswith("{"):
            return False
        if len(_CODE_LINE_RE.findall(content)) >= 2:
            return False
        return len(re.findall(r"[.!?]", content)) >= _MIN_SENTENCES

    def compress(self, content: str, query: str = "") -> CompressResult:
        """Apply keyword-guided extractive summarisation.

        Falls back to passthrough if the text is too short to summarise or
        if no useful keywords can be extracted.
        """
        original_length = len(content)
        sentences = _split_sentences(content)

        if len(sentences) < _MIN_SENTENCES:
            return _make_passthrough(content, self.name)

        aggressiveness = _aggressiveness_from_query(query)
        target_chars = max(200, int(original_length * (1.0 - aggressiveness)))
        top_n = max(10, len(content.split()) // 15)

        keyword_weights = _extract_keywords(content, top_n)
        if not keyword_weights:
            return _make_passthrough(content, self.name)

        scores = [_score_sentence(s, keyword_weights) for s in sentences]
        ranked = sorted(range(len(sentences)), key=lambda i: -scores[i])

        selected: set[int] = set()
        char_count = 0
        for idx in ranked:
            if char_count >= target_chars:
                break
            selected.add(idx)
            char_count += len(sentences[idx]) + 1

        if len(selected) < 2 and len(ranked) >= 2:
            selected = {ranked[0], ranked[1]}

        selected_sents = [sentences[i] for i in sorted(selected)]
        compressed = " ".join(selected_sents)
        compressed_length = len(compressed)

        logger.info(
            "prose_compressed",
            total_sentences=len(sentences),
            selected_sentences=len(selected_sents),
            ratio=round(compressed_length / original_length, 3),
        )

        return CompressResult(
            content=compressed,
            original_length=original_length,
            compressed_length=compressed_length,
            compressor_name=self.name,
            metadata={
                "total_sentences": len(sentences),
                "selected_sentences": len(selected_sents),
                "keyword_count": len(keyword_weights),
                "aggressiveness": round(aggressiveness, 2),
            },
        )
