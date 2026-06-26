# Changelog

All notable changes to Contextly are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **One dashboard for the proxy and every wrapped MCP server.** Each
  `mcp-gateway` instance records its savings into a shared SQLite file
  (`~/.contextly/gateway_stats.db`, override `--stats-path`) tagged by a
  per-server label (override `--name`, default derived from the downstream URL),
  and the proxy's `/dashboard` (plus a new `/gateway-stats` endpoint,
  `CONTEXTLY_GATEWAY_STATS_PATH`) reads it. A single page at
  <http://127.0.0.1:4000/dashboard> now shows the gateway's tool-output savings
  (each tool as `<server> · <tool>`, combined across all wrapped servers) next to
  the proxy's own per-compressor quality.
- **Standalone gateway dashboard, opt-in.** For the case where no proxy runs, the
  gateway can still serve its own dashboard via `--dashboard-port <n>`. It is now
  **off by default**: gateways only write to the shared file and never bind a
  port, so multiple wrapped servers can't knock each other off the dashboard.

### Fixed
- A compressor fault on a tool output no longer breaks the call: the gateway
  falls back to the raw text and logs `gateway_compress_failed`, instead of the
  MCP client reporting "Failed to call tool".
- **Shared multi-server stats.** Gateways now record savings into a shared
  SQLite file (`~/.contextly/gateway_stats.db`, override `--stats-path`) tagged
  by a per-server label (derived from the downstream URL, override `--name`).
  When several servers are wrapped at once — each its own gateway process
  contending for the same dashboard port — the one dashboard that binds the port
  reports the **combined** savings of all of them instead of just its own.

### Fixed
- Gateway compression is now strictly best-effort: a compressor fault on a tool
  output falls back to the raw text and logs `gateway_compress_failed`, instead
  of propagating out and making the MCP client report "Failed to call tool".

## [0.1.0] - 2026-06-22

First public release — a reversible compression gateway for LLM APIs.

### Compression
- **Lossless JSON** (`json_table`): homogeneous record arrays rewritten to a
  columnar table (field names once), every record preserved — ~55–63% fewer
  tokens with no accuracy loss (verified on Llama 3.3 70B: table == full).
- **Log / tool-output folding** (`logs`): repeated lines folded by template
  (timestamps/ids/numbers masked) with `(xN)` counts — up to ~99% on noisy logs.
- **Code** (`code`): tree-sitter comment/whitespace stripping, structure kept.
- **Prose** (`prose`): YAKE keyword-based extractive summarization (lossy).
- **Opt-in record sampling** (`json_smart`): MinHash stratified sampling for
  gist/aggregate workloads (not in the default chain).

### Reversibility
- **CCR store** with `expand`-on-demand: every lossy compression leaves an
  `expand_ref`; full originals recover via `GET /v1/expand/{ref}` or the MCP
  `expand` tool.
- **Granular expand**: `?contains=` / `expand(ref, contains=...)` returns only
  the matching records or log lines.
- **SQLite backend** (`--ccr-backend sqlite`): persistent and shared across
  workers (fixes `--workers > 1`).

### Proxy & integrations
- OpenAI-compatible `/v1/chat/completions` and Anthropic `/v1/messages`, both
  with compression and `X-Contextly-Mode: off|safe|default` per-request control.
- MCP server mode (`compress_text`, `expand`, `retrieve_original`,
  `compression_stats`).
- MCP gateway (`contextly mcp-gateway -- <server>`): proxies another MCP server
  and compresses its tool outputs, injecting an `expand` tool for recovery —
  the way to use Contextly with Claude Desktop's MCP servers.
- Live `/dashboard`, Prometheus `/metrics`, `/stats`, and shadow A/B `/quality`
  with ROUGE-1 + numeric-consistency scoring.

### Tooling
- Offline retention benchmark and a real accuracy benchmark (any
  OpenAI-compatible endpoint) with a CI self-test.
- Docker image published to GHCR on release; PyPI publishing via Trusted
  Publishing (opt-in).

[0.1.0]: https://github.com/Smoke-Dev13/slimctx/releases/tag/v0.1.0
