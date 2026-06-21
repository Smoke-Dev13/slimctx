"""Unit tests for CodeCompressor."""

from __future__ import annotations

import pytest

from contextly.compressors.code import (
    CodeCompressor,
    _detect_language,
    _extract_go,
    _extract_javascript,
    _extract_python,
)

# ── Sample code fixtures ───────────────────────────────────────────────────────


PYTHON_MODULE = """\
import os
import sys
from typing import List, Optional

class DataProcessor:
    def process(self, data: List[str], limit: int = 100) -> Optional[str]:
        result = []
        for item in data:
            if len(item) > limit:
                result.append(item[:limit])
            else:
                result.append(item)
        return "\\n".join(result) if result else None

    def validate(self, data: List[str]) -> bool:
        return all(isinstance(item, str) for item in data)

def load_file(path: str) -> List[str]:
    with open(path) as f:
        return f.readlines()

def save_file(path: str, content: str) -> None:
    with open(path, "w") as f:
        f.write(content)
"""

JAVASCRIPT_MODULE = """\
class UserService {
  getUser(id) { return fetch("/users/" + id).then(r => r.json()); }
  createUser(data) { return fetch("/users", { method: "POST", body: JSON.stringify(data) }); }
}
function processData(items) {
  return items.filter(x => x.active).map(x => x.name);
}
class AdminService {
  deleteUser(id) { return fetch("/users/" + id, { method: "DELETE" }); }
}
"""

GO_MODULE = """\
package main

func ProcessItems(items []string, limit int) []string {
    result := make([]string, 0)
    for _, item := range items {
        if len(item) <= limit {
            result = append(result, item)
        }
    }
    return result
}

func ValidateInput(input string) bool {
    return len(input) > 0 && len(input) < 1000
}

type Config struct {
    Host string
    Port int
}
"""

PLAIN_PROSE = (
    "The system uses machine learning to optimize performance. "
    "Algorithms analyze the data stream continuously. "
    "Results are cached for efficiency. " * 10
)


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def compressor() -> CodeCompressor:
    return CodeCompressor()


# ── _detect_language ───────────────────────────────────────────────────────────


def test_detect_python() -> None:
    assert _detect_language(PYTHON_MODULE) == "python"


def test_detect_javascript() -> None:
    assert _detect_language(JAVASCRIPT_MODULE) == "javascript"


def test_detect_go() -> None:
    assert _detect_language(GO_MODULE) == "go"


def test_detect_none_for_prose() -> None:
    assert _detect_language(PLAIN_PROSE) is None


def test_detect_none_for_empty() -> None:
    assert _detect_language("") is None


def test_detect_none_for_too_few_markers() -> None:
    one_marker = "def foo(): pass"
    assert _detect_language(one_marker) is None


# ── should_apply ───────────────────────────────────────────────────────────────


def test_should_apply_python(compressor: CodeCompressor) -> None:
    assert compressor.should_apply(PYTHON_MODULE) is True


def test_should_apply_javascript(compressor: CodeCompressor) -> None:
    assert compressor.should_apply(JAVASCRIPT_MODULE) is True


def test_should_apply_go(compressor: CodeCompressor) -> None:
    assert compressor.should_apply(GO_MODULE) is True


def test_should_apply_rejects_prose(compressor: CodeCompressor) -> None:
    assert compressor.should_apply(PLAIN_PROSE) is False


def test_name(compressor: CodeCompressor) -> None:
    assert compressor.name == "code"


# ── compress — Python ──────────────────────────────────────────────────────────


def test_compress_python_reduces_size(compressor: CodeCompressor) -> None:
    result = compressor.compress(PYTHON_MODULE)
    assert result.compression_ratio < 1.0


def test_compress_python_preserves_function_names(compressor: CodeCompressor) -> None:
    result = compressor.compress(PYTHON_MODULE)
    assert "load_file" in result.content
    assert "save_file" in result.content


def test_compress_python_preserves_class_name(compressor: CodeCompressor) -> None:
    result = compressor.compress(PYTHON_MODULE)
    assert "DataProcessor" in result.content


def test_compress_python_preserves_method_signatures(compressor: CodeCompressor) -> None:
    result = compressor.compress(PYTHON_MODULE)
    assert "process" in result.content
    assert "validate" in result.content


def test_compress_python_preserves_imports(compressor: CodeCompressor) -> None:
    result = compressor.compress(PYTHON_MODULE)
    assert "import os" in result.content
    assert "import sys" in result.content


def test_compress_python_omits_body(compressor: CodeCompressor) -> None:
    result = compressor.compress(PYTHON_MODULE)
    assert "for item in data" not in result.content
    assert "result.append" not in result.content


def test_compress_python_includes_ellipsis(compressor: CodeCompressor) -> None:
    result = compressor.compress(PYTHON_MODULE)
    assert "..." in result.content


def test_compress_python_preserves_type_hints(compressor: CodeCompressor) -> None:
    result = compressor.compress(PYTHON_MODULE)
    assert "List[str]" in result.content or "List" in result.content


def test_compress_python_60_percent_reduction(compressor: CodeCompressor) -> None:
    result = compressor.compress(PYTHON_MODULE)
    savings = (1.0 - result.compression_ratio) * 100.0
    assert savings > 40.0, f"Expected >40% savings, got {savings:.1f}%"


# ── compress — JavaScript ──────────────────────────────────────────────────────


def test_compress_javascript_reduces_size(compressor: CodeCompressor) -> None:
    result = compressor.compress(JAVASCRIPT_MODULE)
    assert result.compression_ratio < 1.0


def test_compress_javascript_preserves_class_names(compressor: CodeCompressor) -> None:
    result = compressor.compress(JAVASCRIPT_MODULE)
    assert "UserService" in result.content
    assert "AdminService" in result.content


def test_compress_javascript_omits_fetch_body(compressor: CodeCompressor) -> None:
    result = compressor.compress(JAVASCRIPT_MODULE)
    assert "fetch" not in result.content


def test_compress_javascript_metadata(compressor: CodeCompressor) -> None:
    result = compressor.compress(JAVASCRIPT_MODULE)
    assert result.metadata.get("language") == "javascript"


# ── compress — Go ─────────────────────────────────────────────────────────────


def test_compress_go_reduces_size(compressor: CodeCompressor) -> None:
    result = compressor.compress(GO_MODULE)
    assert result.compression_ratio < 1.0


def test_compress_go_preserves_function_names(compressor: CodeCompressor) -> None:
    result = compressor.compress(GO_MODULE)
    assert "ProcessItems" in result.content
    assert "ValidateInput" in result.content


def test_compress_go_preserves_type(compressor: CodeCompressor) -> None:
    result = compressor.compress(GO_MODULE)
    assert "Config" in result.content


def test_compress_go_omits_body(compressor: CodeCompressor) -> None:
    result = compressor.compress(GO_MODULE)
    assert "append(result" not in result.content


def test_compress_go_metadata(compressor: CodeCompressor) -> None:
    result = compressor.compress(GO_MODULE)
    assert result.metadata.get("language") == "go"


# ── compress — general ─────────────────────────────────────────────────────────


def test_compress_passthrough_for_unrecognised_content(compressor: CodeCompressor) -> None:
    result = compressor.compress(PLAIN_PROSE)
    assert result.compression_ratio == pytest.approx(1.0)


def test_compress_metadata_has_language(compressor: CodeCompressor) -> None:
    result = compressor.compress(PYTHON_MODULE)
    assert "language" in result.metadata
    assert result.metadata["language"] == "python"


def test_compress_metadata_has_line_counts(compressor: CodeCompressor) -> None:
    result = compressor.compress(PYTHON_MODULE)
    assert "original_lines" in result.metadata
    assert "compressed_lines" in result.metadata
    assert result.metadata["original_lines"] > result.metadata["compressed_lines"]


# ── _extract_python directly ───────────────────────────────────────────────────


def test_extract_python_returns_string() -> None:
    out = _extract_python(PYTHON_MODULE)
    assert isinstance(out, str)
    assert len(out) > 0


def test_extract_python_no_implementations() -> None:
    out = _extract_python(PYTHON_MODULE)
    assert "for item in data" not in out
    assert "result.append" not in out


# ── _extract_javascript directly ──────────────────────────────────────────────


def test_extract_javascript_returns_string() -> None:
    out = _extract_javascript(JAVASCRIPT_MODULE)
    assert isinstance(out, str)
    assert len(out) > 0


# ── _extract_go directly ──────────────────────────────────────────────────────


def test_extract_go_returns_string() -> None:
    out = _extract_go(GO_MODULE)
    assert isinstance(out, str)
    assert "ProcessItems" in out
