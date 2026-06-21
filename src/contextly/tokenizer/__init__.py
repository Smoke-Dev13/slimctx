"""Offline-first tokenization for Contextly.

Public surface:
  get_tokenizer(model)     → Tokenizer   (factory — never raises, falls back to WordTokenizer)
  Tokenizer                             (abstract base)
  BundledTiktokenTokenizer              (reads cl100k_base / o200k_base from package_data)
  WordTokenizer                         (whitespace+punctuation split, zero deps)
"""

from contextly.tokenizer.base import Tokenizer
from contextly.tokenizer.registry import get_tokenizer
from contextly.tokenizer.tiktoken_bundled import BundledTiktokenTokenizer
from contextly.tokenizer.word_fallback import WordTokenizer

__all__ = ["BundledTiktokenTokenizer", "Tokenizer", "WordTokenizer", "get_tokenizer"]
