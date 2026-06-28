# Contextly

[![CI](https://github.com/Smoke-Dev13/slimctx/actions/workflows/ci.yml/badge.svg)](https://github.com/Smoke-Dev13/slimctx/actions/workflows/ci.yml)
[![Demo & Accuracy](https://github.com/Smoke-Dev13/slimctx/actions/workflows/demo.yml/badge.svg)](https://github.com/Smoke-Dev13/slimctx/actions/workflows/demo.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Smart context optimization proxy for LLM APIs. Drop it in front of any OpenAI-compatible endpoint to compress large prompts, save tokens, and measure quality impact without changing a line of application code.

**A reversible compression gateway for agents and tools.** JSON is compressed losslessly; logs, tool outputs, and long prose are folded aggressively but stay recoverable — every compression leaves an `expand_ref`, so an agent (over HTTP or MCP) can pull the full original back on demand. See it run with no API key: `python scripts/demo_agent_expand.py`.

```
Your app -> Contextly (localhost:4000) -> OpenAI / Anthropic / any LLM
```

**What it does:**

- Compresses prompt messages on the fly — **JSON losslessly by default** (columnar rewrite, every record kept), prose by extractive summarization, code by comment/whitespace stripping
- Stores originals in a reversible CCR store so compressed context can be retrieved verbatim
- Shadows a configurable fraction of requests to the original (uncompressed) upstream and scores quality with ROUGE-1 F1 **and a numeric-consistency check**; run `contextly learn` to mine the log for regressions
- **Bidirectional security firewall** — inbound prompt-injection detection and secret redaction on every request; opt-in outbound scanning catches secrets/PII the model echoes back and flags system-prompt disclosure in responses
- **Cross-agent shared memory** — agents share a persistent key-value store accessible over HTTP (`/v1/memory`) and MCP; entries are semantically deduplicated on write
- Exposes Prometheus metrics at `/metrics` and a JSON stats endpoint at `/stats`
- Optionally runs as an MCP server (Claude Desktop / any MCP client)

---

## How much you lose (and when)

Not all of Contextly's compression is equal — some is lossless, some isn't. Measured on the bundled fixtures (`python scripts/benchmark_quality.py`, model `gpt-4o`):

| Content | Mode | Tokens saved | Information retained |
|---|---|---:|---|
| JSON (200 records) | **default (`json_table`, lossless)** | **56%** | **100%** of records (200/200) |
| JSON (200 records) | opt-in (`json_smart`, sampling) | 98% | 2% of records (5/200) |
| Prose | default (`prose`) | 65% | 32% of numeric facts (9/28) |
| Code | default (`code`) | 63% | 100% of function/class signatures |

**JSON is lossless by default**: homogeneous record arrays are rewritten into a columnar table (field names stated once instead of per record), so every record survives and a model can still answer exact lookups — at ~half the tokens. The aggressive record-*sampling* compressor (98% savings, but 2% of records) is **opt-in**, for gist/aggregate workloads where a representative sample is enough.

Prose compression *is* lossy (it drops low-salience sentences), and `--safe-mode` disables it when answers must be complete. Mitigations across the board:

- **`--safe-mode`** — keeps lossless JSON compression but disables prose sentence-dropping.
- **CCR retrieval** — every original is stored and retrievable by key, so agents can fetch full content on demand.
- **A/B quality + numeric-consistency monitoring** — measure the actual degradation on *your* traffic before trusting it.

### Is this for me?

| Use case | Fit |
|---|---|
| JSON / structured payloads (lookups, analytics, audits) | ✅ Strong — lossless columnar compression keeps every record |
| Summarization, sentiment, topic, "gist" over long prose | ✅ Good |
| Agents / RAG with a retrieval step (MCP `retrieve_original`, CCR keys) | ✅ Strong — compress up front, fetch full fidelity on demand |
| Exact lookups over *prose*, or maximal JSON savings via sampling | 🟡 Use `--safe-mode` for prose; validate sampling with A/B first |

---

## Quick Start

```bash
pip install "contextly[nlp,ast]"

export OPENAI_API_KEY=sk-...
contextly proxy --upstream openai --port 4000
```

Point your OpenAI client at `http://localhost:4000`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:4000/v1", api_key="unused")
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Summarise this document: ..."}],
)
```

Compression is transparent. The upstream response passes through unchanged.

---

## Installation

```bash
# Core only (passthrough compressor, word-tokenizer fallback)
pip install contextly

# Recommended: adds prose (YAKE keyword extraction) and AST-aware code compression
pip install "contextly[nlp,ast]"

# Full: adds sentence-transformer ML ranking and MCP server support
pip install "contextly[all]"
```

**Extras:**

| Extra | Adds |
|---|---|
| `nlp` | YAKE keyword extractor for prose compressor |
| `ast` | tree-sitter parsers for Python, JS, Go code compressor |
| `ml` | sentence-transformers for semantic sentence ranking |
| `mcp-server` | MCP server mode (`contextly mcp`) |
| `all` | Everything above |

---

## CLI Reference

```
contextly proxy      Start the proxy server
contextly bench      Benchmark compression on a JSON payload file
contextly stats      Print live stats from a running proxy
contextly learn      Mine A/B quality log for compression regressions
contextly mcp        Run as an MCP server (stdio transport)
contextly mcp-gateway  Wrap another MCP server, compressing its tool outputs
contextly audit replay  Replay a compression audit log
```

### `contextly proxy`

```
contextly proxy [OPTIONS]

Options:
  --host TEXT                  Bind host  [default: 127.0.0.1]
  --port INTEGER               Listen port  [default: 4000]
  --upstream TEXT              Provider preset (openai|anthropic|openrouter|custom)  [default: openai]
  --upstream-url TEXT          Override upstream base URL
  --upstream-api-key TEXT      API key (defaults to OPENAI_API_KEY / ANTHROPIC_API_KEY)
  --ab-sample-rate FLOAT       Fraction of requests for A/B quality monitoring (0-1)  [default: 0.0]
  --ab-log-path TEXT           Persist A/B samples as JSONL for 'contextly learn'
  --workers INTEGER            Uvicorn worker count  [default: 1]
  --log-level TEXT             [default: info]
  --no-compress                Disable compression pipeline
  --safe-mode                  Never drop JSON records or prose sentences
  --ccr-backend TEXT           Reversible store: memory or sqlite  [default: memory]
  --ccr-path TEXT              SQLite path (when --ccr-backend sqlite)
```

### `contextly learn`

Mine the A/B quality log (produced when `--ab-log-path` is set) for compressor/model combinations whose quality regressed and emit ranked, actionable recommendations:

```bash
contextly learn .contextly/ab.jsonl
contextly learn .contextly/ab.jsonl --min-quality 0.75 --json
```

```
Options:
  --min-quality FLOAT   Mean ROUGE-1 below which a combo is a failure  [default: 0.7]
  --min-numeric FLOAT   Mean numeric-consistency below which factual loss is flagged  [default: 0.9]
  --json                Emit the report as JSON
```

Groups with fewer than 5 samples are reported as `low` confidence so you don't act on noise. A group with good ROUGE-1 but low numeric consistency is still flagged as `medium` severity — a fluent answer with a wrong figure is the silent failure mode lossy compression is most likely to produce.

### Safe mode

`--safe-mode` (or `CONTEXTLY_SAFE_MODE=true`) guarantees the model still sees **every** prose sentence. It removes the only lossy default compressor — `prose` sentence-dropping — from the routing chain. JSON still gets the **lossless** `json_table` treatment (so you keep ~half the token savings with zero data loss), and `code` (comment/whitespace stripping) stays enabled. Use it when wrong answers are worse than expensive ones.

---

## Configuration

All settings can be set via environment variables with the `CONTEXTLY_` prefix or in a `.env` file.

| Variable | Default | Description |
|---|---|---|
| `CONTEXTLY_HOST` | `127.0.0.1` | Bind address |
| `CONTEXTLY_PORT` | `4000` | Listen port |
| `CONTEXTLY_WORKERS` | `1` | Uvicorn worker processes |
| `CONTEXTLY_LOG_LEVEL` | `info` | Log level (`debug`, `info`, `warning`, `error`) |
| `CONTEXTLY_UPSTREAM` | `openai` | Provider preset (`openai`, `anthropic`, `openrouter`, `custom`) |
| `CONTEXTLY_UPSTREAM_BASE_URL` | -- | Explicit upstream base URL (overrides preset) |
| `CONTEXTLY_UPSTREAM_API_KEY` | -- | Upstream API key (also reads `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`) |
| `CONTEXTLY_COMPRESSION_ENABLED` | `true` | Enable/disable the compression pipeline |
| `CONTEXTLY_SAFE_MODE` | `false` | Disable lossy prose compression (JSON stays lossless either way) |
| `CONTEXTLY_CCR_BACKEND` | `memory` | Reversible store: `memory` or `sqlite` (persisted + shared across workers) |
| `CONTEXTLY_CCR_PATH` | `.contextly/ccr.db` | SQLite file path when `CCR_BACKEND=sqlite` |
| `CONTEXTLY_TARGET_TOKEN_BUDGET` | -- | Optional token budget hint for budget-aware compressors |
| `CONTEXTLY_AB_SAMPLE_RATE` | `0.0` | Fraction of requests to shadow for A/B quality measurement |
| `CONTEXTLY_AB_LOG_PATH` | -- | Append A/B samples as JSONL here; `contextly learn` reads this file |
| `CONTEXTLY_FIREWALL_ENABLED` | `false` | Enable inbound prompt-injection detection and secret/PII redaction |
| `CONTEXTLY_FIREWALL_SCAN_RESPONSES` | `false` | Also scan upstream responses for echoed secrets and injection-leak markers (requires `FIREWALL_ENABLED`) |
| `CONTEXTLY_INJECTION_BLOCK_THRESHOLD` | `0.0` | Auto-reject requests whose injection risk score exceeds this (0 = flag-only) |

---

## API Reference

### OpenAI-compatible endpoints

#### `POST /v1/chat/completions`

Drop-in replacement for the OpenAI chat completions endpoint. Compresses each message before forwarding, passes the response through unchanged.

**Per-request control** — send `X-Contextly-Mode` to override the global setting for a single call:

- `off` — skip compression for this request
- `safe` — use the lossless chain only (no prose/log dropping)
- *(absent / anything else)* — the configured default

**Additional response headers:**

- `X-Contextly-Compressed: true|false` -- whether compression ran
- `X-Contextly-CCR-Keys: {"msg:0": "<key>", ...}` -- CCR keys per compressed message index (text content blocks use `msg:{i}:{j}`)

#### `POST /v1/messages`

Anthropic Messages API proxy **with compression** (same pipeline and `X-Contextly-Mode` control as chat/completions). Use with `--upstream anthropic`.

#### `POST /v1/compress`

Explicit compression without proxying to an LLM.

```json
// Request
{"content": "Your long document...", "query": "optional user query"}

// Response
{
  "content": "...compressed...",
  "original_length": 4200,
  "compressed_length": 1800,
  "ratio": 0.4286,
  "compressor": "prose",
  "metadata": {"keywords": ["..."]},
  "ccr_key": "a3f1c9e2b8d74501"
}
```

#### `GET /v1/retrieve/{key}`

Retrieve the original content stored under a CCR key.

```json
// 200
{"key": "a3f1c9e2b8d74501", "content": "Your long document..."}

// 404 (evicted or invalid)
{"error": "Key 'a3f1c9e2b8d74501' not found. It may have been evicted."}
```

#### `GET /v1/expand/{ref}`

Expand a compressed result back to its full original — the recovery path for lossy compression. `ref` is the `ccr_key` / `expand_ref` from the compression response or the `X-Contextly-CCR-Keys` header.

Add `?contains=<substr>` for **granular** recovery — only the matching records (JSON) or lines (logs/text), so the agent spends tokens on just the detail it needs.

```json
// 200  GET /v1/expand/a3f1c9e2b8d74501
{"ref": "a3f1c9e2b8d74501", "found": true, "content": "...", "matches": -1}

// 200  GET /v1/expand/a3f1c9e2b8d74501?contains=order-8421
{"ref": "a3f1c9e2b8d74501", "found": true, "content": "[{...}]", "matches": 1}

// 404
{"ref": "a3f1c9e2b8d74501", "found": false, "error": "Reference '...' not found or evicted."}
```

### Observability endpoints

#### `GET /dashboard`

A self-contained live dashboard (no CDN, no build step). Open it in a browser while the proxy runs — it polls `/stats` and `/quality` and shows tokens saved, estimated cost saved, average compression, and per-compressor quality, refreshing every 2 seconds.

#### `GET /health`

Liveness probe. Returns 200 with server status.

#### `GET /stats`

Aggregate compression statistics since server start.

```json
{
  "requests_total": 1024,
  "requests_compressed": 987,
  "chars_saved_total": 8431200,
  "compression_ratio_mean": 0.42,
  "ab_samples_total": 51
}
```

#### `GET /quality`

A/B quality regression report (populated when `ab_sample_rate > 0`).

```json
{
  "samples_total": 51,
  "quality": {"mean": 0.8721, "p10": 0.72, "p50": 0.88, "p90": 0.96},
  "numeric_consistency": {"mean": 0.91, "p10": 0.5},
  "chars_saved": {"mean": 2140.3, "total": 109155},
  "by_compressor": {
    "prose": {"samples": 38, "mean_quality": 0.89, "mean_numeric_consistency": 0.88},
    "json_smart": {"samples": 13, "mean_quality": 0.94, "mean_numeric_consistency": 0.97}
  }
}
```

`numeric_consistency` is the fraction of numbers in the original-context response that survive in the compressed-context response. It catches a failure mode ROUGE-1 misses: two responses can share most words yet quote a different figure ("312 tickets" vs "47 tickets"). Watch the `p10` — a low tail means compression is occasionally corrupting facts even when mean quality looks healthy.

#### `GET /metrics`

Prometheus text format exposition. Scrape with Prometheus or any compatible agent.

---

## Compression Pipeline

The content router selects a compressor per message in registration-priority order. The first compressor whose `should_apply()` returns `True` is used; `passthrough` is the guaranteed fallback.

| Compressor | Triggers on | Algorithm | Lossy? |
|---|---|---|---|
| `json_table` | Homogeneous JSON object arrays | **Lossless** columnar rewrite — field names stated once, every record/value preserved (round-trips exactly) | No |
| `prose` | Natural-language text above a length threshold | YAKE keyword extraction; keeps top-K sentences by keyword density | Yes |
| `code` | Source code (detected by AST parse) | tree-sitter parse; removes comments and blank lines, preserves structure | Minimal |
| `logs` | Multi-line logs / tool output | Folds repeated lines by template (timestamps/ids/numbers masked); each unique pattern kept once with an `(xN)` count | Yes (expand-recoverable) |
| `passthrough` | Everything else | Returns content unchanged | No |
| `json_smart` | *(opt-in, not in default chain)* | MinHash clustering + stratified record **sampling**; keeps outliers, drops the rest | Yes |

### CCR Store + expand-on-demand

When a compressor reduces content, the original is stored in an in-memory LRU cache keyed by a 16-character hex SHA-256 prefix. The key is returned in the response so callers can retrieve the original via `GET /v1/retrieve/{key}` or the MCP `retrieve_original` tool.

**Expand-on-demand** turns this into a safety net for *lossy* compression. Whenever compression drops information, the result is marked `expandable` with an `expand_ref`, and the full original can be pulled back via `GET /v1/expand/{ref}` or the MCP `expand` tool. This is what lets an agent compress aggressively up front yet recover any specific record, line, or figure it later needs — so token savings never become permanent data loss.

The store holds up to 10 000 entries; oldest entries are evicted on overflow (LRU order).

By default the store is in-memory (per process). For multi-worker deployments (`--workers > 1`) or to keep references across restarts, switch to the SQLite backend with `--ccr-backend sqlite` (or `CONTEXTLY_CCR_BACKEND=sqlite`): all workers share one database file, so a reference stored by one worker expands correctly from any other.

---

## A/B Quality Monitoring

When `CONTEXTLY_AB_SAMPLE_RATE > 0`, a shadow request fires for a random fraction of non-streaming requests:

1. The main request proceeds with compressed context -- response is returned immediately.
2. In a background task, the **original** (uncompressed) context is sent to the same upstream.
3. Both responses are scored two ways: word-level ROUGE-1 F1 (precision/recall harmonic mean) and **numeric consistency** (fraction of figures preserved).
4. The scores are stored in a ring buffer (1 000 samples max) and emitted to Prometheus.

A ROUGE-1 score of `1.0` means the compressed-context response is word-for-word identical to the original; a numeric-consistency of `1.0` means no figure was changed. The two are complementary — ROUGE-1 measures overall wording, numeric consistency guards the specific facts (counts, prices, dates) that lossy compression is most likely to corrupt silently. Streaming requests are excluded because buffering the response would negate the point of streaming.

> **Caveat:** ROUGE-1 compares the compressed-context answer to the *full-context* answer, not to ground truth — it tells you how much the answer *changed*, not whether it was right to begin with. For high-stakes use, pair it with an LLM-judge or task-specific exact-match eval on a held-out set.

View results at `GET /quality` or via the `contextly_ab_quality_score` histogram in Prometheus.

---

## Security Firewall

Contextly ships a zero-dependency, regex-based security layer that operates on both sides of the proxy. It requires no external service and adds sub-millisecond overhead.

### Inbound (request) scanning

Enable with `CONTEXTLY_FIREWALL_ENABLED=true`:

- **Prompt-injection detection** (`InjectionScanner`) — scans every incoming message for patterns characteristic of override attempts, jailbreak triggers, role escalation, data-exfiltration requests, and delimiter injection. Returns a risk score in `[0, 1]`; requests above `CONTEXTLY_INJECTION_BLOCK_THRESHOLD` are rejected with `400`.
- **Secret/PII redaction** (`SecretRedactor`) — replaces API keys (OpenAI, Anthropic, AWS, GCP, Stripe, …), SSNs, credit-card numbers, and other PII in the prompt before it reaches the upstream model.

### Outbound (response) scanning

Enable additionally with `CONTEXTLY_FIREWALL_SCAN_RESPONSES=true`:

- **Response secret detection** — runs the same secret catalogue on the model's reply. If the model echoed a key or SSN from context back to the caller, `X-Contextly-Response-Secrets-Redacted: <n>` is added to the response headers and the counter increments in `/stats`.
- **Injection-leak detection** — scans the response for symptoms of a *successful* injection: system-prompt disclosure ("my instructions are …", "here is my system prompt"), verbatim chat-template delimiters (`<|im_start|>`, `[INST]`, `<<SYS>>`). Detected leaks set `X-Contextly-Response-Injection-Leak: <score>`.

Both outbound detections are **flag-only** (non-destructive) — the response body is not rewritten, so structured JSON responses are never corrupted. Body-rewrite is a planned follow-up.

### Security stats

```bash
curl http://localhost:4000/stats | jq .firewall
```

```json
{
  "secrets_redacted_total": 12,
  "requests_with_secrets_total": 7,
  "response_secrets_redacted_total": 2,
  "responses_with_secrets_total": 1,
  "injections_detected_total": 3,
  "injections_blocked_total": 1
}
```

---

## Cross-Agent Shared Memory

Contextly includes a persistent key-value memory store accessible over HTTP and MCP, designed for multi-agent pipelines where agents need to share state across calls.

### HTTP API

| Method | Path | Description |
|---|---|---|
| `PUT` | `/v1/memory/{key}` | Store a value (body: `{"value": "..."}`) |
| `GET` | `/v1/memory/{key}` | Retrieve a stored value |
| `DELETE` | `/v1/memory/{key}` | Delete a key |
| `GET` | `/v1/memory` | List all keys |

### MCP tools

When running as an MCP server (`contextly mcp`), the same store is exposed as `memory_write`, `memory_read`, `memory_delete`, and `memory_list` tools.

### Semantic deduplication

On write, Contextly checks for near-duplicate entries using a fast content hash. Entries whose content is semantically equivalent (within the configured threshold) are silently merged rather than stored twice, keeping the memory store compact even in long-running agent loops.

---

## Prometheus Metrics

| Metric | Type | Labels | Description |
|---|---|---|---|
| `contextly_requests_total` | Counter | `model`, `compressor` | Proxied non-streaming requests |
| `contextly_chars_saved_total` | Counter | `compressor` | Characters saved by compression |
| `contextly_compression_ratio` | Histogram | `compressor` | Ratio of compressed to original chars (1.0 = no compression) |
| `contextly_request_latency_seconds` | Histogram | `model` | End-to-end latency to upstream response |
| `contextly_ab_quality_score` | Histogram | `compressor` | ROUGE-1 F1 quality score per A/B sample |

### Prometheus scrape config

```yaml
scrape_configs:
  - job_name: contextly
    static_configs:
      - targets: ["contextly:4000"]
```

---

## MCP Server Mode

Contextly can run as an MCP server over stdio, exposing three tools to Claude Desktop or any MCP-compatible client.

```bash
pip install "contextly[mcp-server]"
contextly mcp
```

**Claude Desktop config** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "contextly": {
      "command": "contextly",
      "args": ["mcp"]
    }
  }
}
```

**Available MCP tools:**

| Tool | Description |
|---|---|
| `compress_text` | Compress text; returns an `expand_ref` and `expandable` flag when compression was lossy |
| `expand` | Expand a compressed result back to the full original by its `expand_ref` |
| `retrieve_original` | Retrieve original content by CCR key |
| `compression_stats` | Return CCR store hit/miss statistics |

---

## MCP Gateway (compress another server's tool outputs)

`contextly mcp` exposes Contextly's *own* tools. The **gateway** instead sits between an MCP client (e.g. Claude Desktop) and any **other** MCP server, forwarding its tools unchanged while compressing the tool *outputs* — where agentic token cost actually piles up (DB rows, API responses, logs, file dumps):

```
Claude Desktop  ──▶  contextly mcp-gateway  ──▶  real MCP server (filesystem, postgres, …)
```

Lossless JSON is compressed to a columnar table; logs are folded; an `expand` tool is injected so the model can recover the full original (or just matching records/lines via `contains`) of anything compressed with loss.

```bash
contextly mcp-gateway -- npx -y @modelcontextprotocol/server-filesystem /data
```

**Claude Desktop config** (`claude_desktop_config.json`) — wrap an existing server:

```json
{
  "mcpServers": {
    "fs-compressed": {
      "command": "contextly",
      "args": ["mcp-gateway", "--",
               "npx", "-y", "@modelcontextprotocol/server-filesystem", "/data"]
    }
  }
}
```

Requires the `mcp` extra (`pip install "contextly[mcp-server]"`); the published binaries already bundle it.

**One dashboard for everything.** Each gateway records its savings into a shared file (`~/.contextly/gateway_stats.db`; `--stats-path` to change, `--name` to label a server), and the **proxy's** dashboard reads it — so a single page shows the gateway's tool-output savings (each tool as `<server> · <tool>`) next to the proxy's own per-compressor quality. Run the proxy once and open that one URL:

```bash
contextly proxy            # serves the dashboard
# → open http://127.0.0.1:4000/dashboard
```

Wrapping several servers spawns several gateway processes, but they only *write* to the shared file — they never bind a port, so they can't knock each other off the dashboard; all of them appear together on `:4000`.

If you don't run the proxy at all, the gateway can serve its own standalone dashboard instead — opt in with `--dashboard-port <n>` (off by default). Use it only when wrapping a single server, since just one process can bind the port.

**Wrapping several servers → one dashboard.** Claude Desktop launches one gateway process per wrapped server, and they all default to the same dashboard port — only one can bind it. So every gateway records into a *shared* SQLite file (`~/.contextly/gateway_stats.db`, override with `--stats-path`), and the single dashboard that wins the port shows the **combined** savings of all of them, tagged by a `server` label derived from each downstream URL (override with `--name`). Just wrap each server the same way; no extra config needed to see them together.

> Note: this compresses the **tool outputs** flowing through MCP. It cannot compress the Claude Desktop chat itself — that conversation goes straight to Anthropic and has no proxy hook.

---

## Docker

Pull the published image from GHCR (built and pushed by the release workflow on each version tag):

```bash
docker run -p 4000:4000 -e OPENAI_API_KEY=sk-... ghcr.io/smoke-dev13/slimctx:latest
```

Or build it locally:

```bash
docker build -t contextly:latest .
docker run -p 4000:4000 -e OPENAI_API_KEY=sk-... contextly:latest
```

### Docker Compose

```bash
# Proxy only
docker compose up contextly

# Proxy + Prometheus
docker compose --profile observability up
```

**Environment variables** (via `.env` or `docker compose` environment section):

```env
OPENAI_API_KEY=sk-...
CONTEXTLY_UPSTREAM=openai
CONTEXTLY_COMPRESSION_ENABLED=true
CONTEXTLY_AB_SAMPLE_RATE=0.05
CONTEXTLY_LOG_LEVEL=info
```

The container runs as a non-root user (`uid=1001`), listens on port `4000`, and includes a health check against `/health`.

---

## Development

```bash
git clone <repo>
cd contextly
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[all,dev]"
```

**Run tests:**

```bash
pytest                        # all tests
pytest tests/unit/            # unit tests only
pytest tests/integration/     # integration tests only
pytest --cov=contextly --cov-report=term-missing
```

**Lint + type check:**

```bash
ruff check src/ tests/
mypy src/
```

Coverage threshold is 85% (`fail_under = 85` in `pyproject.toml`).

---

## Architecture

```
contextly proxy (FastAPI)
+-- POST /v1/chat/completions
|   +-- ContentRouter          -> selects compressor per message
|   |   +-- JsonSmartCompressor
|   |   +-- ProseCompressor
|   |   +-- CodeCompressor
|   |   +-- PassthroughCompressor (fallback)
|   +-- CCRStore               -> stores originals, returns 16-char keys
|   +-- ABMonitor              -> records every request; optional shadow A/B
|   +-- metrics.observe_request -> Prometheus counters/histograms
+-- GET /v1/retrieve/{key}     -> CCR retrieval
+-- GET /health                -> liveness
+-- GET /stats                 -> JSON aggregate stats
+-- GET /quality               -> A/B quality report
+-- GET /metrics               -> Prometheus exposition

contextly mcp (FastMCP / stdio)
+-- tool: compress_text
+-- tool: retrieve_original
+-- tool: compression_stats
```

---

## Benchmarks

Reproduce the savings-vs-retention numbers locally — no API key or network required:

```bash
python scripts/benchmark_quality.py --model gpt-4o
# narrow the aggressiveness with a query:
python scripts/benchmark_quality.py --model gpt-4o --query "find the unusual transaction"
```

The script runs each compressor on a representative fixture and prints token savings next to an information-retention metric (records retained for JSON, numeric facts for prose, signatures for code), so you can judge the trade-off rather than just the headline savings. Wire your own corpus in by editing the fixtures at the top of the script.

### Accuracy benchmark — does compression change the answers?

Retention is a proxy; the real question is whether a model gives the *right* answer with less context. `scripts/accuracy_benchmark.py` asks record-lookup questions over a synthetic JSON set under three strategies — `full`, `table` (the lossless default), and `sampled` (the opt-in lossy compressor) — calls a real LLM, and grades answers against gold values:

```bash
# Any OpenAI-compatible endpoint (OpenAI, OpenRouter, local Ollama, or the proxy itself)
export LLM_API_KEY=...
python scripts/accuracy_benchmark.py \
    --base-url https://api.groq.com/openai/v1 \
    --model llama-3.3-70b-versatile \
    --api-key-env LLM_API_KEY

# Validate the harness offline, no API key (deterministic oracle model):
python scripts/accuracy_benchmark.py --self-test
```

The harness is verified offline on every push (the **Demo & Accuracy** workflow). To reproduce live numbers, add your provider API key as the `LLM_API_KEY` repository secret (Settings → Secrets → Actions) and run that workflow manually (Actions → *Demo & Accuracy Benchmark* → *Run workflow*); results are written to the run summary. Defaults target Groq's free tier.

**Why the default is lossless.** Measured on a real model — Llama 3.3 70B (Groq), record-lookup questions over a 120-record JSON set:

| Strategy | Accuracy | Mean context tokens | Tokens vs full |
|---|---:|---:|---:|
| full (raw JSON) | **100%** (4/4) | 4096 | — |
| **`table` (default, lossless)** | **100%** (4/4) | 1523 | **−63%** |
| `sampled` (opt-in, lossy) | **0%** (0/4) | 57 | −99% |

The lossless `table` form matches full-context accuracy exactly while cutting tokens by ~63%, because the model sees every record — just with the field names factored out and a one-line `_help` hint on how to read the columns. The `sampled` compressor keeps ~1% of records and scores 0% on lookups, which is why it is opt-in. (The live run above is small — Groq's free tier rate-limited it at 4 questions — but it confirms the model reads the columnar format correctly; the deterministic round-trip test and offline self-test independently verify all 200/200 records are preserved. Re-run with your own key/model via the workflow for a larger sample.) Reserve `--arms sampled` / `json_smart` for gist/aggregate workloads where a representative sample is enough.

---

## How Contextly compares to LLMLingua

[LLMLingua / LLMLingua-2](https://github.com/microsoft/LLMLingua) (Microsoft) is the best-known prompt-compression project. It uses a small language model to score and drop low-information *tokens*, and is excellent at squeezing verbose natural-language prompts.

Contextly is a different shape:

| | Contextly | LLMLingua |
|---|---|---|
| Unit of compression | Records / sentences / code structure | Individual tokens |
| Deployment | **Transparent OpenAI-compatible proxy** — zero app changes | Library you call in-process |
| Structured data (JSON) | First-class (MinHash clustering + stratified sampling) | Not the focus |
| Reversibility | **CCR store** — fetch any original back by key | None |
| Runtime cost | No model inference on the hot path | Runs a compressor LM per request |
| Quality monitoring | **Built-in shadow A/B + numeric consistency** | Bring your own |
| Best at | Agents/RAG, JSON-heavy payloads, drop-in cost control | Dense token-level reduction of prose prompts |

They are complementary: LLMLingua minimizes tokens within text you've decided to keep; Contextly decides *what to keep* at the record/sentence level and gives you a retrieval escape hatch. If your bottleneck is huge JSON payloads or you want a proxy you can drop in front of an existing app without code changes, Contextly fits. If you need maximal token reduction of prose and can call a library, LLMLingua is strong.

---

## License

MIT
