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
  * injects a ``contextly_expand`` tool so the client can recover the full
    original of any lossily-compressed output (optionally filtered),
  * forwards resources and prompts unchanged when the downstream exposes them, so
    wrapping a server never hides its features.

Lossless compression (JSON → columnar table) is applied with no expand marker —
the data is all there. Lossy compression (logs, prose) stores the original in a
CCR store and appends a ``contextly_expand("<ref>")`` hint.

Run it:
    contextly mcp-gateway -- npx -y @modelcontextprotocol/server-filesystem /data
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from urllib.parse import urlparse

import structlog
from pydantic import AnyUrl

from contextly.ccr import CCRStore
from contextly.compressors.code import CodeCompressor
from contextly.compressors.json_table import JsonTableCompressor
from contextly.compressors.logs import LogCompressor
from contextly.compressors.prose import ProseCompressor
from contextly.compressors.registry import ContentRouter
from contextly.expand import filter_original
from contextly.gateway_stats import SQLiteStatsStore, StatsRecorder, default_stats_path

try:
    import mcp.types as types
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.server import Server
    from mcp.server.lowlevel.helper_types import ReadResourceContents
    from mcp.server.stdio import stdio_server
except ImportError as _err:  # pragma: no cover
    raise ImportError(
        "The 'mcp' package is required for the MCP gateway. "
        "Install it with: pip install 'contextly[mcp-server]'"
    ) from _err

logger = structlog.get_logger(__name__)

_EXPAND_TOOL_NAME = "contextly_expand"


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
        f'Call {_EXPAND_TOOL_NAME}("{ref}") for the full original, or '
        f'{_EXPAND_TOOL_NAME}("{ref}", contains="...") for matching records/lines.'
    )
    return result.content + note, ref


_EXPAND_TOOL_DESCRIPTION = (
    "Expand a compressed tool output back to its original. Pass the ref shown in "
    f"a '[contextly] ... {_EXPAND_TOOL_NAME}(\"<ref>\")' hint. Optionally pass "
    "'contains' to get only the matching records (JSON) or lines (logs/text)."
)


def build_gateway_server(
    session: ClientSession,
    store: CCRStore,
    router: ContentRouter,
    *,
    forward_resources: bool = False,
    forward_prompts: bool = False,
    stats: StatsRecorder | None = None,
) -> Server:
    """Build the proxy MCP server that wraps *session* (the downstream server).

    Tools are always forwarded (and their outputs compressed). Resources and
    prompts are forwarded only when the downstream advertises those capabilities,
    so the gateway never advertises something it cannot serve. When *stats* is
    given, each tool call's before/after size is recorded for the live dashboard.
    """
    server: Server = Server("contextly-gateway")
    expand_tool = types.Tool(
        name=_EXPAND_TOOL_NAME,
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
        if name == _EXPAND_TOOL_NAME:
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
                # Compression is best-effort: a compressor fault must never turn a
                # working tool into a failing one, so fall back to the raw text.
                try:
                    compressed, _ref = compress_payload(block.text, router, store)
                except Exception:
                    logger.warning("gateway_compress_failed", tool=name, exc_info=True)
                    compressed = block.text
                new_content.append(types.TextContent(type="text", text=compressed))
            else:
                new_content.append(block)

        # One clear savings line per tool call, visible in the client's MCP log.
        before = sum(len(b.text) for b in result.content if isinstance(b, types.TextContent))
        after = sum(len(b.text) for b in new_content if isinstance(b, types.TextContent))
        saved_pct = round(100 * (1 - after / before)) if before else 0
        if stats is not None:
            stats.record(name, before, after)
        logger.info(
            "gateway_tool_result",
            tool=name,
            chars_before=before,
            chars_after=after,
            saved_pct=saved_pct,
        )
        # Diagnostic: when a sizeable output did not compress, log its head so the
        # output's shape can be inspected (and a compressor added if worthwhile).
        if before > 500 and after >= before:
            head = next(
                (b.text[:200] for b in result.content if isinstance(b, types.TextContent)), ""
            )
            logger.info("gateway_uncompressed_sample", tool=name, chars=before, head=head)
        return types.CallToolResult(
            content=new_content,
            structuredContent=result.structuredContent,
            isError=bool(result.isError),
        )

    if forward_resources:

        @server.list_resources()
        async def _list_resources() -> list[types.Resource]:
            return list((await session.list_resources()).resources)

        @server.read_resource()
        async def _read_resource(uri: AnyUrl) -> list[ReadResourceContents]:
            result = await session.read_resource(uri)
            out: list[ReadResourceContents] = []
            for c in result.contents:
                if isinstance(c, types.TextResourceContents):
                    out.append(ReadResourceContents(content=c.text, mime_type=c.mimeType))
                elif isinstance(c, types.BlobResourceContents):
                    out.append(
                        ReadResourceContents(content=base64.b64decode(c.blob), mime_type=c.mimeType)
                    )
            return out

    if forward_prompts:

        @server.list_prompts()
        async def _list_prompts() -> list[types.Prompt]:
            return list((await session.list_prompts()).prompts)

        @server.get_prompt()
        async def _get_prompt(name: str, arguments: dict[str, str] | None) -> types.GetPromptResult:
            return await session.get_prompt(name, arguments)

    return server


def derive_server_name(command: str, args: list[str]) -> str:
    """Best-effort short label for the wrapped server, for the dashboard.

    Prefers the first hostname's leading label among the downstream tokens (so
    ``mcp-remote https://nocodb.example/...`` → ``nocodb``); otherwise falls back
    to the command's basename without extension.
    """
    for token in [command, *args]:
        if "://" in token:
            host = urlparse(token).hostname or ""
            if host:
                return host.split(".")[0]
    base = Path(command).name
    return base.rsplit(".", 1)[0] if "." in base else base


async def run_gateway(
    command: str,
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    store: CCRStore | None = None,
    dashboard_host: str = "127.0.0.1",
    dashboard_port: int | None = None,
    server_name: str = "",
    stats_path: str | None = None,
    stats: StatsRecorder | None = None,
) -> None:
    """Launch the downstream MCP server and serve the gateway over stdio.

    Savings are always recorded into a shared SQLite file (*stats_path*, default
    ``~/.contextly/gateway_stats.db``) under *server_name*. The single dashboard is
    the proxy's ``/dashboard`` (default ``http://127.0.0.1:4000``), which reads that
    file — so there is one pane for every wrapped server *and* the proxy's own
    stats, and the gateway processes never compete for a port.

    *dashboard_port* is opt-in (off by default) for the case where no proxy runs:
    set it (CLI: ``--dashboard-port <n>``) to also serve the standalone gateway
    dashboard on a background thread. Avoid it when wrapping multiple servers —
    only one process can bind the port.
    """
    # MCP stdio requires stdout to carry ONLY JSON-RPC; send all logs to stderr.
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(file=sys.stderr))
    store = store or CCRStore()
    router = build_router()
    server_name = server_name or derive_server_name(command, args)
    stats = stats or SQLiteStatsStore(stats_path or default_stats_path(), server_name)
    if dashboard_port:
        from contextly.gateway_dashboard import start_dashboard

        start_dashboard(stats, dashboard_host, dashboard_port)
    params = StdioServerParameters(command=command, args=args, env=env)
    logger.info("gateway_starting", downstream=command, args=args, server=server_name)
    async with stdio_client(params) as (down_read, down_write):
        async with ClientSession(down_read, down_write) as session:
            init = await session.initialize()
            caps = init.capabilities
            logger.info(
                "gateway_downstream_ready",
                resources=caps.resources is not None,
                prompts=caps.prompts is not None,
            )
            server = build_gateway_server(
                session,
                store,
                router,
                forward_resources=caps.resources is not None,
                forward_prompts=caps.prompts is not None,
                stats=stats,
            )
            async with stdio_server() as (read, write):
                await server.run(read, write, server.create_initialization_options())
