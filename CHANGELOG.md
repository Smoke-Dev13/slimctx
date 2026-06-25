# Changelog

All notable changes to Contextly are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

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
