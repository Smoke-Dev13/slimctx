# Contextly

Smart context optimization proxy for LLM APIs. Drop it in front of any OpenAI-compatible endpoint to compress large prompts, save tokens, and measure quality impact without changing a line of application code.

```
Your app -> Contextly (localhost:4000) -> OpenAI / Anthropic / any LLM
```

**What it does:**

- Compresses prompt messages on the fly (prose, JSON, code -- each with a specialized algorithm)
- Stores originals in a reversible CCR store so compressed context can be retrieved verbatim
- Shadows a configurable fraction of requests to the original (uncompressed) upstream and scores quality with ROUGE-1 F1
- Exposes Prometheus metrics at `/metrics` and a JSON stats endpoint at `/stats`
- Optionally runs as an MCP server (Claude Desktop / any MCP client)

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
contextly mcp        Run as an MCP server (stdio transport)
contextly stats      Print live stats from a running proxy
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
  --workers INTEGER            Uvicorn worker count  [default: 1]
  --log-level TEXT             [default: info]
  --no-compress                Disable compression pipeline
```

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
| `CONTEXTLY_TARGET_TOKEN_BUDGET` | -- | Optional token budget hint for budget-aware compressors |
| `CONTEXTLY_AB_SAMPLE_RATE` | `0.0` | Fraction of requests to shadow for A/B quality measurement |

---

## API Reference

### OpenAI-compatible endpoints

#### `POST /v1/chat/completions`

Drop-in replacement for the OpenAI chat completions endpoint. Compresses each message before forwarding, passes the response through unchanged.

**Additional response headers:**

- `X-Contextly-Compressed: true|false` -- whether compression ran
- `X-Contextly-CCR-Keys: {"msg:0": "<key>", ...}` -- CCR keys per compressed message index

#### `POST /v1/messages`

Anthropic Messages API pass-through (no compression). Use with `--upstream anthropic`.

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

### Observability endpoints

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
  "chars_saved": {"mean": 2140.3, "total": 109155},
  "by_compressor": {
    "prose": {"samples": 38, "mean_quality": 0.89},
    "json_smart": {"samples": 13, "mean_quality": 0.94}
  }
}
```

#### `GET /metrics`

Prometheus text format exposition. Scrape with Prometheus or any compatible agent.

---

## Compression Pipeline

The content router selects a compressor per message in registration-priority order. The first compressor whose `should_apply()` returns `True` is used; `passthrough` is the guaranteed fallback.

| Compressor | Triggers on | Algorithm |
|---|---|---|
| `json_smart` | Valid JSON objects/arrays | Elides null/empty fields, rounds floats, truncates long string values |
| `prose` | Natural-language text above a length threshold | YAKE keyword extraction; keeps top-K sentences by keyword density |
| `code` | Source code (detected by AST parse) | tree-sitter parse; removes comments and blank lines, preserves structure |
| `passthrough` | Everything else | Returns content unchanged |

### CCR Store (Contextly Compression & Retrieval)

When a compressor reduces content, the original is stored in an in-memory LRU cache keyed by a 16-character hex SHA-256 prefix. The key is returned in the response so callers can retrieve the original via `GET /v1/retrieve/{key}` or the MCP `retrieve_original` tool.

The store holds up to 10 000 entries; oldest entries are evicted on overflow (LRU order).

---

## A/B Quality Monitoring

When `CONTEXTLY_AB_SAMPLE_RATE > 0`, a shadow request fires for a random fraction of non-streaming requests:

1. The main request proceeds with compressed context -- response is returned immediately.
2. In a background task, the **original** (uncompressed) context is sent to the same upstream.
3. Both responses are scored with word-level ROUGE-1 F1 (precision/recall harmonic mean).
4. The score is stored in a ring buffer (1 000 samples max) and emitted to Prometheus.

A score of `1.0` means the compressed-context response is word-for-word identical to the original. Streaming requests are excluded because buffering the response would negate the point of streaming.

View results at `GET /quality` or via the `contextly_ab_quality_score` histogram in Prometheus.

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
| `compress_text` | Compress text and store original in CCR store |
| `retrieve_original` | Retrieve original content by CCR key |
| `compression_stats` | Return CCR store hit/miss statistics |

---

## Docker

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

## License

MIT
