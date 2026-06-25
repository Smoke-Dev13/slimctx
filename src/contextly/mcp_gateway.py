"""MCP compression gateway.

Sits between an MCP client (e.g. Claude Desktop) and a downstream MCP server,
forwarding the protocol transparently while compressing the *tool outputs* that
flow back to the client — where agentic token cost concentrates (DB rows, API
responses, logs, file dumps).

    Claude Desktop  ──stdio──▶  contextly mcp-gateway  ──stdio──▶  real MCP server

The gateway:
  * forwards the downstream server's tools unchanged (the client sees them as-is),
  * compresses the text content of each tool result (lossless JSON tables, log
    folding, etc.),
  * injects an ``expand`` tool so the client can recover the full original of any
    lossily-compressed output (optionally filtered to matching records/lines).

Lossless compression (JSON → columnar table) is applied with no expand marker —
the data is all there. Lossy compression (logs, prose) stores the original in a
CCR store and appends an ``expand("<ref>")`` hint.

Run it:
    contextly mcp-gateway -- npx -y @modelcontextprotocol/server-filesystem /data
"""

from __future__ import annotations

import structlog

from contextly.ccr import CCRStore
from contextly.compressors.code import CodeCompressor
from contextly.compressors.json_table import JsonTableCompressor
from contextly.compressors.logs import LogCompressor
from contextly.compressors.prose import ProseCompressor
from contextly.compressors.registry import ContentRouter
from contextly.expand import filter_original

try:
    import mcp.types as types
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
except ImportError as _err:  # pragma: no cover
    raise ImportError(
        "The 'mcp' package is required for the MCP gateway. "
        "Install it with: pip install 'contextly[mcp-server]'"
    ) from _err

logger = structlog.get_logger(__name__)


def build_router() -> ContentRouter:
    """Default compressor chain for tool outputs (lossless JSON first)."""
    router = ContentRouter()
    router.register(JsonTableCompressor())
    router.register(CodeCompressor())
    router.register(LogCompressor())
    router.register(ProseCompressor())
    return router


def compress_payload(text: str, router: ContentRouter, store: CCRStore) -> tuple[str, str | None]:
    """Compress a tool-output string, returning (new_text, expand_ref | None).

    * No benefit → original text, no ref.
    * Lossless (JSON table) → compressed text, no ref (everything is preserved).
    * Lossy (logs/prose) → compressed text + an expand hint, with the original
      stored in the CCR store under the returned ref.
    """
    if not text.strip():
        return text, None
    result = router.select(text).compress(text)
    if result.compression_ratio >= 1.0:
        return text, None
    if result.metadata.get("lossless"):
        return result.content, None
    ref = store.store(text)
    saved = round(100 * (1 - result.compression_ratio))
    note = (
        f"\n\n[contextly] Output compressed ~{saved}% to save tokens. "
        f'Call expand("{ref}") for the full original, or '
        f'expand("{ref}", contains="...") for matching records/lines.'
    )
    return result.content + note, ref


_EXPAND_TOOL_DESCRIPTION = (
    "Expand a compressed tool output back to its original. Pass the ref shown in "
    "a '[contextly] ... expand(\"<ref>\")' hint. Optionally pass 'contains' to get "
    "only the matching records (JSON) or lines (logs/text)."
)


def build_gateway_server(session: ClientSession, store: CCRStore, router: ContentRouter) -> Server:
    """Build the proxy MCP server that wraps *session* (the downstream server)."""
    server: Server = Server("contextly-gateway")
    expand_tool = types.Tool(
        name="expand",
        description=_EXPAND_TOOL_DESCRIPTION,
        inputSchema={
            "type": "object",
            "properties": {
                "ref": {"type": "string", "description": "The expand reference."},
                "contains": {"type": "string", "description": "Optional substring filter."},
            },
            "required": ["ref"],
        },
    )

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        downstream = await session.list_tools()
        return [*downstream.tools, expand_tool]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, object]) -> types.CallToolResult:
        if name == "expand":
            ref = str(arguments.get("ref", ""))
            contains = str(arguments.get("contains", ""))
            original = store.retrieve(ref)
            if original is None:
                return types.CallToolResult(
                    content=[
                        types.TextContent(type="text", text=f"expand: ref '{ref}' not found.")
                    ],
                    isError=True,
                )
            content, matches = filter_original(original, contains)
            logger.info("gateway_expand", ref=ref, matches=matches)
            return types.CallToolResult(content=[types.TextContent(type="text", text=content)])

        result = await session.call_tool(name, arguments)
        new_content: list[types.ContentBlock] = []
        for block in result.content:
            if isinstance(block, types.TextContent):
                compressed, block_ref = compress_payload(block.text, router, store)
                if compressed != block.text:
                    logger.info(
                        "gateway_compressed",
                        tool=name,
                        ref=block_ref,
                        saved=len(block.text) - len(compressed),
                    )
                new_content.append(types.TextContent(type="text", text=compressed))
            else:
                new_content.append(block)
        return types.CallToolResult(
            content=new_content,
            structuredContent=result.structuredContent,
            isError=bool(result.isError),
        )

    return server


async def run_gateway(
    command: str,
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    store: CCRStore | None = None,
) -> None:
    """Launch the downstream MCP server and serve the gateway over stdio."""
    store = store or CCRStore()
    router = build_router()
    params = StdioServerParameters(command=command, args=args, env=env)
    logger.info("gateway_starting", downstream=command, args=args)
    async with stdio_client(params) as (down_read, down_write):
        async with ClientSession(down_read, down_write) as session:
            await session.initialize()
            server = build_gateway_server(session, store, router)
            async with stdio_server() as (read, write):
                await server.run(read, write, server.create_initialization_options())
