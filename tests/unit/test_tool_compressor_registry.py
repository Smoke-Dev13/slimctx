"""Unit tests for ToolCompressorRegistry in mcp_gateway."""

from __future__ import annotations

from contextly.compressors.json_table import JsonTableCompressor
from contextly.compressors.prose import ProseCompressor
from contextly.mcp_gateway import ToolCompressorRegistry


def _registry(learning_calls: int = 3) -> ToolCompressorRegistry:
    return ToolCompressorRegistry(
        compressors=[JsonTableCompressor(), ProseCompressor()],
        learning_calls=learning_calls,
    )


def test_select_returns_compressor_during_learning() -> None:
    reg = _registry(learning_calls=5)
    comp = reg.select("my_tool", "some text")
    assert comp is not None
    assert hasattr(comp, "compress")


def test_select_locks_in_after_learning() -> None:
    reg = _registry(learning_calls=2)
    text = "some text content"
    reg.select("my_tool", text)
    reg.select("my_tool", text)  # hits threshold
    # After learning, the tool should appear in routing table
    table = reg.routing_table()
    assert "my_tool" in table["learned"]


def test_override_skips_learning() -> None:
    reg = ToolCompressorRegistry(
        compressors=[JsonTableCompressor(), ProseCompressor()],
        learning_calls=5,
        overrides={"my_tool": "json_table"},
    )
    table = reg.routing_table()
    assert table["learned"].get("my_tool") == "json_table"
    # First call should use the override directly
    comp = reg.select("my_tool", "some data")
    assert comp.name == "json_table"


def test_routing_table_structure() -> None:
    reg = _registry()
    reg.select("tool_a", "text")
    table = reg.routing_table()
    assert "learned" in table
    assert "call_counts" in table
    assert "learning_calls_required" in table
    assert table["call_counts"]["tool_a"] == 1


def test_multiple_tools_independent() -> None:
    reg = _registry(learning_calls=3)
    reg.select("tool_a", "text a")
    reg.select("tool_b", "text b")
    table = reg.routing_table()
    assert table["call_counts"].get("tool_a") == 1
    assert table["call_counts"].get("tool_b") == 1
