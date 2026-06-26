"""Click-based CLI for Contextly.

Commands:
  contextly proxy   — start the proxy server
  contextly bench   — run compression benchmarks (M3)
  contextly stats   — print live stats from a running proxy
  contextly mcp     — run as an MCP server (stdio transport for Claude Desktop)
"""

from __future__ import annotations

import sys

import click

from contextly.config import Config, UpstreamProvider


@click.group()
@click.version_option(package_name="contextly")
def main() -> None:
    """Contextly — smart context optimization proxy for LLM APIs."""


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind host")
@click.option("--port", default=4000, show_default=True, type=int, help="Listen port")
@click.option(
    "--upstream",
    default="openai",
    show_default=True,
    type=click.Choice([p.value for p in UpstreamProvider], case_sensitive=False),
    help="Upstream LLM provider",
)
@click.option(
    "--upstream-url",
    default=None,
    envvar="CONTEXTLY_UPSTREAM_BASE_URL",
    help="Override upstream base URL (for local or custom targets)",
)
@click.option(
    "--upstream-api-key",
    default="",
    envvar=["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "CONTEXTLY_UPSTREAM_API_KEY"],
    help="API key forwarded to upstream (defaults to OPENAI_API_KEY / ANTHROPIC_API_KEY)",
    show_default=False,
)
@click.option(
    "--ab-sample-rate",
    default=0.0,
    show_default=True,
    type=float,
    help="Fraction of requests to run through A/B quality monitoring (0-1)",
)
@click.option("--workers", default=1, show_default=True, type=int, help="Uvicorn worker count")
@click.option(
    "--log-level",
    default="info",
    show_default=True,
    type=click.Choice(["debug", "info", "warning", "error"], case_sensitive=False),
)
@click.option("--no-compress", is_flag=True, default=False, help="Disable compression pipeline")
@click.option(
    "--safe-mode",
    is_flag=True,
    default=False,
    help="Never drop JSON records or prose sentences (only strip code comments/whitespace)",
)
@click.option(
    "--ccr-backend",
    type=click.Choice(["memory", "sqlite"], case_sensitive=False),
    default="memory",
    show_default=True,
    help="Reversible store backend; use 'sqlite' to persist and share across --workers",
)
@click.option(
    "--ccr-path",
    default=".contextly/ccr.db",
    show_default=True,
    help="SQLite database path (when --ccr-backend sqlite)",
)
def proxy(
    host: str,
    port: int,
    upstream: str,
    upstream_url: str | None,
    upstream_api_key: str,
    ab_sample_rate: float,
    workers: int,
    log_level: str,
    no_compress: bool,
    safe_mode: bool,
    ccr_backend: str,
    ccr_path: str,
) -> None:
    """Start the Contextly proxy server.

    Example:

    \b
        export OPENAI_API_KEY=sk-...
        contextly proxy --upstream openai --port 4000
    """
    from contextly.server import run

    config = Config(
        host=host,
        port=port,
        upstream=UpstreamProvider(upstream),
        upstream_base_url=upstream_url,  # type: ignore[arg-type]
        upstream_api_key=upstream_api_key,
        ab_sample_rate=ab_sample_rate,
        workers=workers,
        log_level=log_level,
        compression_enabled=not no_compress,
        safe_mode=safe_mode,
        ccr_backend=ccr_backend.lower(),  # type: ignore[arg-type]
        ccr_path=ccr_path,
    )
    run(config)


@main.command()
@click.argument("payload_file", type=click.Path(exists=True))
@click.option(
    "--model",
    default="gpt-4o",
    show_default=True,
    help="Model name displayed in output (affects token estimates for known models)",
)
def bench(payload_file: str, model: str) -> None:
    """Benchmark compression on a JSON payload file.

    PAYLOAD_FILE should be a JSON file containing either an OpenAI messages array
    or a full chat payload object with a "messages" key.

    \b
    Example:
        contextly bench messages.json --model gpt-4o
    """
    import json
    import time as _time

    from contextly.compressors.code import CodeCompressor
    from contextly.compressors.json_table import JsonTableCompressor
    from contextly.compressors.logs import LogCompressor
    from contextly.compressors.prose import ProseCompressor
    from contextly.compressors.registry import ContentRouter

    try:
        with open(payload_file, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        click.echo(f"Error reading {payload_file}: {exc}", err=True)
        sys.exit(1)

    if isinstance(data, dict):
        messages: list[dict[str, object]] = list(data.get("messages", []))
        model = str(data.get("model", model))
    elif isinstance(data, list):
        messages = list(data)
    else:
        click.echo("Payload must be a JSON object with 'messages' or a messages array.", err=True)
        sys.exit(1)

    if not messages:
        click.echo("No messages found in payload.", err=True)
        sys.exit(1)

    router = ContentRouter()
    router.register(JsonTableCompressor())
    router.register(CodeCompressor())
    router.register(LogCompressor())
    router.register(ProseCompressor())

    row_fmt = " {:>3}  {:<10}  {:>9,}  {:>10,}  {:>8,}  {:<14}  {:>7.1f}"
    header = " {:>3}  {:<10}  {:>9}  {:>10}  {:>8}  {:<14}  {:>7}".format(
        "#", "Role", "Original", "Compressed", "Saved", "Compressor", "ms"
    )
    sep = "-" * len(header)

    click.echo(f"\nBenchmarking {len(messages)} messages  (model: {model})\n")
    click.echo(header)
    click.echo(sep)

    total_orig = 0
    total_comp = 0
    total_tokens = 0
    total_ms = 0.0

    for idx, msg in enumerate(messages):
        role = str(msg.get("role", "?"))
        content = msg.get("content", "")
        if not isinstance(content, str) or not content:
            click.echo(row_fmt.format(idx, role, 0, 0, 0, "passthrough", 0.0))
            continue
        t0 = _time.monotonic()
        compressor = router.select(content, "")
        result = compressor.compress(content, "")
        elapsed_ms = (_time.monotonic() - t0) * 1000.0
        total_orig += result.original_length
        total_comp += result.compressed_length
        total_tokens += result.tokens_saved_estimate
        total_ms += elapsed_ms
        click.echo(
            row_fmt.format(
                idx,
                role,
                result.original_length,
                result.compressed_length,
                result.tokens_saved_estimate,
                result.compressor_name,
                elapsed_ms,
            )
        )

    click.echo(sep)
    saved_pct = round((total_orig - total_comp) / total_orig * 100, 1) if total_orig else 0.0
    click.echo(
        row_fmt.format("", "TOTAL", total_orig, total_comp, total_tokens, "", total_ms)
        + f"   ({saved_pct}% saved)"
    )
    click.echo()


@main.command(name="mcp")
def mcp_server() -> None:
    """Run Contextly as an MCP server over stdio (for Claude Desktop / MCP clients).

    Exposes the compress_text, retrieve_original, and compression_stats tools.

    Example entry in claude_desktop_config.json:

    \b
        {
          "contextly": {
            "command": "contextly",
            "args": ["mcp"]
          }
        }
    """
    import asyncio

    try:
        from contextly.mcp_server import mcp
    except ImportError as exc:
        click.echo(f"MCP server requires the 'mcp' package: {exc}", err=True)
        sys.exit(1)

    asyncio.run(mcp.run_stdio_async())


@main.command(
    name="mcp-gateway",
    context_settings={"ignore_unknown_options": True},
)
@click.option(
    "--dashboard-port",
    default=0,
    show_default=True,
    type=int,
    help="Also serve a standalone gateway dashboard on this port (0 = off). "
    "Normally unnecessary — the proxy's :4000 dashboard already shows gateway "
    "savings. Only one process can bind it, so avoid when wrapping several servers.",
)
@click.option(
    "--dashboard-host",
    default="127.0.0.1",
    show_default=True,
    help="Bind host for the standalone gateway dashboard (when --dashboard-port is set)",
)
@click.option(
    "--name",
    "server_name",
    default="",
    help="Label for this server in the dashboard (default: derived from the downstream URL)",
)
@click.option(
    "--stats-path",
    default="",
    help="Shared savings DB path (default: ~/.contextly/gateway_stats.db). "
    "Keep identical across wrapped servers to aggregate them in one dashboard.",
)
@click.argument("downstream", nargs=-1, type=click.UNPROCESSED)
def mcp_gateway(
    dashboard_port: int,
    dashboard_host: str,
    server_name: str,
    stats_path: str,
    downstream: tuple[str, ...],
) -> None:
    """Proxy a downstream MCP server, compressing its tool outputs on the way back.

    Put Contextly between an MCP client (Claude Desktop) and another MCP server:
    the client sees the downstream tools unchanged, but their outputs are
    compressed, with an injected ``expand`` tool to recover the full original.

    Savings are recorded into a shared stats file (one per machine), so the
    proxy's single dashboard at http://127.0.0.1:4000/dashboard shows this
    server's tool-output savings next to the proxy's own stats — run
    ``contextly proxy`` once and open that one page. Wrapping several servers? They
    all write to the same file and appear together there (label each with --name;
    relocate with --stats-path). The gateways never bind a port, so they can't
    knock each other off the dashboard.

    Pass the downstream server command after ``--``.

    \b
    Example:
        contextly mcp-gateway -- npx -y @modelcontextprotocol/server-filesystem /data

    \b
    claude_desktop_config.json:
        {
          "fs-compressed": {
            "command": "contextly",
            "args": ["mcp-gateway", "--",
                     "npx", "-y", "@modelcontextprotocol/server-filesystem", "/data"]
          }
        }
    """
    import asyncio

    cmd = list(downstream)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        click.echo("Usage: contextly mcp-gateway -- <command> [args...]", err=True)
        sys.exit(1)

    try:
        from contextly.mcp_gateway import run_gateway
    except ImportError as exc:
        click.echo(f"MCP gateway requires the 'mcp' package: {exc}", err=True)
        sys.exit(1)

    asyncio.run(
        run_gateway(
            cmd[0],
            cmd[1:],
            dashboard_host=dashboard_host,
            dashboard_port=dashboard_port or None,
            server_name=server_name,
            stats_path=stats_path or None,
        )
    )


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Proxy host")
@click.option("--port", default=4000, show_default=True, type=int, help="Proxy port")
def stats(host: str, port: int) -> None:
    """Print live statistics from a running Contextly proxy."""
    import json

    import httpx as _httpx

    try:
        resp = _httpx.get(f"http://{host}:{port}/stats", timeout=5.0)
        click.echo(json.dumps(resp.json(), indent=2))
    except _httpx.ConnectError:
        click.echo(f"Cannot connect to proxy at {host}:{port}", err=True)
        sys.exit(1)
