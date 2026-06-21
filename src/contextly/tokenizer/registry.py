"""Tokenizer factory — resolves a model name to the best available tokenizer.

Resolution order:
  1. Exact model name lookup in _MODEL_TO_ENCODING
  2. Prefix matching (handles versioned names like "gpt-4o-2024-08-06")
  3. WordTokenizer fallback for unknown models

BundledTiktokenTokenizer is returned only when the bundled .tiktoken file
is present (verified by is_bundled()). If the file is missing — e.g. the
package was installed without running scripts/download_encodings.py — the
factory silently returns WordTokenizer rather than raising.
"""

from __future__ import annotations

import structlog

from contextly.tokenizer.base import Tokenizer
from contextly.tokenizer.tiktoken_bundled import BundledTiktokenTokenizer, is_bundled
from contextly.tokenizer.word_fallback import WordTokenizer

logger = structlog.get_logger(__name__)

# Maps exact model names to encoding names.
# Source: tiktoken/model.py in the tiktoken repository.
_MODEL_TO_ENCODING: dict[str, str] = {
    # GPT-4o family (o200k_base)
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-4o-2024-05-13": "o200k_base",
    "gpt-4o-2024-08-06": "o200k_base",
    "gpt-4o-mini-2024-07-18": "o200k_base",
    "o1": "o200k_base",
    "o1-mini": "o200k_base",
    "o1-preview": "o200k_base",
    "o3": "o200k_base",
    "o3-mini": "o200k_base",
    # GPT-4 family (cl100k_base)
    "gpt-4": "cl100k_base",
    "gpt-4-32k": "cl100k_base",
    "gpt-4-0314": "cl100k_base",
    "gpt-4-32k-0314": "cl100k_base",
    "gpt-4-0613": "cl100k_base",
    "gpt-4-32k-0613": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4-turbo-preview": "cl100k_base",
    "gpt-4-1106-preview": "cl100k_base",
    "gpt-4-0125-preview": "cl100k_base",
    # GPT-3.5 family (cl100k_base)
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-3.5-turbo-0301": "cl100k_base",
    "gpt-3.5-turbo-0613": "cl100k_base",
    "gpt-3.5-turbo-16k": "cl100k_base",
    "gpt-3.5-turbo-16k-0613": "cl100k_base",
    "gpt-3.5-turbo-1106": "cl100k_base",
    "gpt-3.5-turbo-0125": "cl100k_base",
    "gpt-3.5-turbo-instruct": "cl100k_base",
    # Embeddings (cl100k_base)
    "text-embedding-ada-002": "cl100k_base",
    "text-embedding-3-small": "cl100k_base",
    "text-embedding-3-large": "cl100k_base",
    # Anthropic: cl100k_base is the best public approximation
    "claude-3-5-sonnet-20241022": "cl100k_base",
    "claude-3-5-haiku-20241022": "cl100k_base",
    "claude-3-opus-20240229": "cl100k_base",
    "claude-3-sonnet-20240229": "cl100k_base",
    "claude-3-haiku-20240307": "cl100k_base",
}

# Prefix rules applied when no exact match is found (first match wins).
_PREFIX_RULES: list[tuple[str, str]] = [
    ("gpt-4o", "o200k_base"),
    ("o1-", "o200k_base"),
    ("o1", "o200k_base"),
    ("o3-", "o200k_base"),
    ("o3", "o200k_base"),
    ("gpt-4", "cl100k_base"),
    ("gpt-3.5", "cl100k_base"),
    ("claude-", "cl100k_base"),
]


def get_tokenizer(model: str) -> Tokenizer:
    """Return the best available tokenizer for the given model name.

    Resolution is fully local — no network calls, no subprocess spawning.
    Falls back to WordTokenizer when the model is unknown or the bundled
    encoding file is absent.

    Args:
        model: LLM model name as it appears in the API request (e.g. 'gpt-4o').

    Returns:
        A ready-to-use Tokenizer instance.
    """
    encoding_name = _MODEL_TO_ENCODING.get(model)
    if encoding_name is None:
        for prefix, enc in _PREFIX_RULES:
            if model.startswith(prefix):
                encoding_name = enc
                break

    if encoding_name is not None and is_bundled(encoding_name):
        logger.debug("tokenizer_resolved", model=model, encoding=encoding_name)
        return BundledTiktokenTokenizer(encoding_name)

    logger.info(
        "tokenizer_fallback_word",
        model=model,
        reason="encoding_not_bundled" if encoding_name else "unknown_model",
    )
    return WordTokenizer()
