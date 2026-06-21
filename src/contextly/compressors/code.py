"""Code structure compressor backed by tree-sitter AST parsing.

Supported languages: Python, JavaScript, Go.
Falls back to passthrough if tree-sitter is not installed or parsing fails.

Output: condensed code that preserves:
  - All import statements
  - Class signatures + first-line docstrings
  - Function/method signatures + first-line docstrings
  - Type declarations (Go)

Function bodies are replaced with ``...`` (Python) or ``{...}`` (JS/Go),
reducing a typical module by 60-80 % while keeping the structure an LLM
needs to reason about the code.
"""

from __future__ import annotations

import re
from typing import Any

import structlog

from contextly.compressors.base import Compressor, CompressResult

logger = structlog.get_logger(__name__)

# ── Language detection ────────────────────────────────────────────────────────

_PY_MARKERS = re.compile(r"^(def |class |import |from \S+ import |async def |@\w)", re.MULTILINE)
_JS_MARKERS = re.compile(
    r"^(function |class |const |let |export (default )?function |export (default )?class )",
    re.MULTILINE,
)
_GO_MARKERS = re.compile(r"^(func |package |type )", re.MULTILINE)

_MIN_MARKER_COUNT = 2


def _detect_language(content: str) -> str | None:
    """Return 'python', 'javascript', 'go', or None."""
    py = len(_PY_MARKERS.findall(content))
    js = len(_JS_MARKERS.findall(content))
    go = len(_GO_MARKERS.findall(content))
    if py >= _MIN_MARKER_COUNT and py >= js and py >= go:
        return "python"
    if js >= _MIN_MARKER_COUNT and js >= go:
        return "javascript"
    if go >= _MIN_MARKER_COUNT:
        return "go"
    return None


# ── Tree-sitter Python extraction ─────────────────────────────────────────────


def _ts_text(node: Any) -> str:
    return node.text.decode("utf-8", errors="replace") if node else ""


def _py_docstring(body_node: Any) -> str:
    """Return the first-line docstring from a block node, or empty string."""
    if not body_node:
        return ""
    for child in body_node.children:
        if child.type == "expression_statement" and child.children:
            expr = child.children[0]
            if expr.type == "string":
                raw = _ts_text(expr).strip("\"'").strip()
                return raw.splitlines()[0].strip()[:120]
    return ""


def _py_function_sig(node: Any, indent: str) -> list[str]:
    name = _ts_text(node.child_by_field_name("name"))
    params = _ts_text(node.child_by_field_name("parameters"))
    ret_node = node.child_by_field_name("return_type")
    ret = f" -> {_ts_text(ret_node)}" if ret_node else ""
    sig = f"{indent}def {name}{params}{ret}:"
    body = node.child_by_field_name("body")
    doc = _py_docstring(body)
    if doc:
        return [sig, f'{indent}    "{doc}"', f"{indent}    ..."]
    return [f"{sig} ..."]


def _py_class_sig(node: Any, indent: str) -> list[str]:
    name = _ts_text(node.child_by_field_name("name"))
    bases_node = node.child_by_field_name("superclasses")
    bases = f"({_ts_text(bases_node)})" if bases_node else ""
    header = f"{indent}class {name}{bases}:"
    lines = [header]

    body = node.child_by_field_name("body")
    doc = _py_docstring(body)
    if doc:
        lines.append(f'{indent}    "{doc}"')

    if body:
        for child in body.children:
            if child.type == "function_definition":
                lines.extend(_py_function_sig(child, indent + "    "))
            elif child.type == "decorated_definition":
                deco_lines, defn = _py_decorated(child, indent + "    ")
                lines.extend(deco_lines)
                if defn is not None and defn.type == "function_definition":
                    lines.extend(_py_function_sig(defn, indent + "    "))
                elif defn is not None and defn.type == "class_definition":
                    lines.extend(_py_class_sig(defn, indent + "    "))
    if len(lines) == 1:
        lines.append(f"{indent}    ...")
    return lines


def _py_decorated(node: Any, indent: str) -> tuple[list[str], Any]:
    """Return (decorator_lines, inner_definition_node)."""
    deco_lines: list[str] = []
    defn = None
    for child in node.children:
        if child.type == "decorator":
            deco_lines.append(indent + _ts_text(child))
        elif child.type in ("function_definition", "class_definition"):
            defn = child
    return deco_lines, defn


def _extract_python(content: str) -> str:
    try:
        import tree_sitter as ts
        import tree_sitter_python as tsp
    except ImportError:
        return content

    src = content.encode("utf-8", errors="replace")
    lang: Any = ts.Language(tsp.language())
    parser: Any = ts.Parser(lang)
    tree: Any = parser.parse(src)

    if tree.root_node.has_error:
        return content

    lines: list[str] = []
    for node in tree.root_node.children:
        ntype = node.type
        if ntype in ("import_statement", "import_from_statement"):
            lines.append(_ts_text(node))
        elif ntype == "function_definition":
            lines.extend(_py_function_sig(node, ""))
        elif ntype == "class_definition":
            lines.extend(_py_class_sig(node, ""))
        elif ntype == "decorated_definition":
            deco_lines, defn = _py_decorated(node, "")
            lines.extend(deco_lines)
            if defn is not None and defn.type == "function_definition":
                lines.extend(_py_function_sig(defn, ""))
            elif defn is not None and defn.type == "class_definition":
                lines.extend(_py_class_sig(defn, ""))
        # Skip: assignments, expression statements, comments, etc.

    return "\n".join(lines)


# ── Tree-sitter JavaScript extraction ─────────────────────────────────────────


def _js_function_sig(node: Any, indent: str, keyword: str = "function") -> str:
    name_node = node.child_by_field_name("name")
    name = _ts_text(name_node) if name_node else "<anonymous>"
    params_node = node.child_by_field_name("parameters")
    params = _ts_text(params_node) if params_node else "()"
    return f"{indent}{keyword} {name}{params} {{...}}"


def _js_method_sig(node: Any, indent: str) -> str:
    name_node = node.child_by_field_name("name")
    name = _ts_text(name_node) if name_node else "<method>"
    params_node = node.child_by_field_name("parameters")
    params = _ts_text(params_node) if params_node else "()"
    is_static = any(c.type == "static" for c in node.children)
    prefix = "static " if is_static else ""
    return f"{indent}{prefix}{name}{params} {{...}}"


def _extract_javascript(content: str) -> str:
    try:
        import tree_sitter as ts
        import tree_sitter_javascript as tsjs
    except ImportError:
        return content

    src = content.encode("utf-8", errors="replace")
    lang: Any = ts.Language(tsjs.language())
    parser: Any = ts.Parser(lang)
    tree: Any = parser.parse(src)

    if tree.root_node.has_error:
        return content

    lines: list[str] = []
    for node in tree.root_node.children:
        ntype = node.type
        if ntype == "import_statement":
            lines.append(_ts_text(node))
        elif ntype == "function_declaration":
            lines.append(_js_function_sig(node, ""))
        elif ntype == "class_declaration":
            name_node = node.child_by_field_name("name")
            name = _ts_text(name_node) if name_node else "Anonymous"
            heritage = node.child_by_field_name("heritage")
            ext = f" extends {_ts_text(heritage)}" if heritage else ""
            lines.append(f"class {name}{ext} {{")
            body = node.child_by_field_name("body")
            if body:
                for child in body.children:
                    if child.type == "method_definition":
                        lines.append(_js_method_sig(child, "  "))
            lines.append("}")
        elif ntype == "export_statement":
            decl = node.child_by_field_name("declaration")
            if decl is None:
                lines.append(_ts_text(node).split("\n")[0] + " ...")
            elif decl.type == "function_declaration":
                lines.append("export " + _js_function_sig(decl, ""))
            elif decl.type == "class_declaration":
                name_node = decl.child_by_field_name("name")
                name = _ts_text(name_node) if name_node else "Anonymous"
                lines.append(f"export class {name} {{...}}")
            else:
                lines.append(_ts_text(node).split("\n")[0] + " ...")
        # Skip: variable declarations without function values, comments, etc.

    return "\n".join(lines)


# ── Tree-sitter Go extraction ─────────────────────────────────────────────────


def _extract_go(content: str) -> str:
    try:
        import tree_sitter as ts
        import tree_sitter_go as tsgo
    except ImportError:
        return content

    src = content.encode("utf-8", errors="replace")
    lang: Any = ts.Language(tsgo.language())
    parser: Any = ts.Parser(lang)
    tree: Any = parser.parse(src)

    if tree.root_node.has_error:
        return content

    lines: list[str] = []
    for node in tree.root_node.children:
        ntype = node.type
        if ntype == "package_clause":
            lines.append(_ts_text(node))
        elif ntype == "import_declaration":
            lines.append(_ts_text(node))
        elif ntype == "function_declaration":
            name = _ts_text(node.child_by_field_name("name"))
            params = _ts_text(node.child_by_field_name("parameters"))
            result = node.child_by_field_name("result")
            ret = f" {_ts_text(result)}" if result else ""
            lines.append(f"func {name}{params}{ret} {{...}}")
        elif ntype == "method_declaration":
            recv = _ts_text(node.child_by_field_name("receiver"))
            name = _ts_text(node.child_by_field_name("name"))
            params = _ts_text(node.child_by_field_name("parameters"))
            result = node.child_by_field_name("result")
            ret = f" {_ts_text(result)}" if result else ""
            lines.append(f"func {recv} {name}{params}{ret} {{...}}")
        elif ntype == "type_declaration":
            lines.append(_ts_text(node))
        # Skip: var blocks, const blocks, comments

    return "\n".join(lines)


# ── Language dispatch ─────────────────────────────────────────────────────────


_EXTRACTORS: dict[str, Any] = {
    "python": _extract_python,
    "javascript": _extract_javascript,
    "go": _extract_go,
}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_passthrough(content: str, name: str) -> CompressResult:
    length = len(content)
    return CompressResult(
        content=content,
        original_length=length,
        compressed_length=length,
        compressor_name=name,
    )


# ── Compressor ────────────────────────────────────────────────────────────────


class CodeCompressor(Compressor):
    """Structure extractor for Python, JavaScript, and Go source code.

    Detection (should_apply): checks for ≥2 code-specific keywords
    (def/class/import/function/func/package) at line starts.

    Compression pipeline:
      1. Detect language from marker patterns.
      2. Parse with the appropriate tree-sitter grammar.
      3. Walk top-level nodes; emit imports, signatures, docstrings; skip bodies.
      4. If parsing fails or produces errors, fall back to passthrough.
    """

    @property
    def name(self) -> str:
        return "code"

    def should_apply(self, content: str, query: str = "") -> bool:
        """Return True if content looks like Python, JavaScript, or Go source."""
        return _detect_language(content) is not None

    def compress(self, content: str, query: str = "") -> CompressResult:
        """Extract code signatures; replace function/method bodies with ellipsis."""
        original_length = len(content)
        language = _detect_language(content)

        if language is None:
            return _make_passthrough(content, self.name)

        extractor = _EXTRACTORS[language]
        try:
            compressed_content = extractor(content)
        except Exception:
            logger.warning("code_extraction_failed", language=language, exc_info=True)
            return _make_passthrough(content, self.name)

        if not compressed_content.strip() or len(compressed_content) >= original_length:
            return _make_passthrough(content, self.name)

        compressed_length = len(compressed_content)
        original_lines = content.count("\n") + 1
        compressed_lines = compressed_content.count("\n") + 1

        logger.info(
            "code_compressed",
            language=language,
            original_lines=original_lines,
            compressed_lines=compressed_lines,
            ratio=round(compressed_length / original_length, 3),
        )

        return CompressResult(
            content=compressed_content,
            original_length=original_length,
            compressed_length=compressed_length,
            compressor_name=self.name,
            metadata={
                "language": language,
                "original_lines": original_lines,
                "compressed_lines": compressed_lines,
            },
        )
