# Contextly

[![CI](https://github.com/Smoke-Dev13/slimctx/actions/workflows/ci.yml/badge.svg)](https://github.com/Smoke-Dev13/slimctx/actions/workflows/ci.yml)
[![Demo & Accuracy](https://github.com/Smoke-Dev13/slimctx/actions/workflows/demo.yml/badge.svg)](https://github.com/Smoke-Dev13/slimctx/actions/workflows/demo.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Smart context optimization proxy for LLM APIs. Drop it in front of any OpenAI-compatible endpoint to compress large prompts, save tokens, and measure quality impact without changing a line of application code.

```
Your app -> Contextly (localhost:4000) -> OpenAI / Anthropic / any LLM
```

**What it does:**

- Compresses prompt messages on the fly (prose, JSON, code -- each with a specialized algorithm)
- Stores originals in a reversible CCR store so compressed context can be retrieved verbatim
- Shadows a configurable fraction of requests to the original (uncompressed) upstream and scores quality with ROUGE-1 F1 **and a numeric-consistency check**
- Exposes Prometheus metrics at `/metrics` and a JSON stats endpoint at `/stats`
- Optionally runs as an MCP server (Claude Desktop / any MCP client)

---

## ⚠️ Compression is lossy by design

Contextly does **not** losslessly pack your prompt. To save tokens it *drops information*: the JSON compressor samples a small subset of records, and the prose compressor keeps only the highest-scoring sentences. The model sees a **representative fraction** of your content, not all of it.

Measured on the bundled fixtures (`python scripts/benchmark_quality.py`, model `gpt-4o`):

| Content | Tokens saved | Information retained |
|---|---:|---|
| JSON (200 records) | **98%** | **2%** of records (5/200) |
| Prose | 65% | 32% of numeric facts (9/28) |
| Code | 63% | 100% of function/class signatures |

**This is great for aggregate questions** ("what's the overall sentiment?", "roughly how many users churned?") **and dangerous for lookups** ("what is order #8421's total?", "list every failed transaction"). The dropped record might be the one that mattered.

Mitigations Contextly ships with:

- **`--safe-mode`** — never drops JSON records or prose sentences; only structure-preserving code compression runs (see below). Use this when answers must be complete.
- **CCR retrieval** — every original is stored and retrievable by key, so agents can fetch the full content back on demand.
- **A/B quality + numeric-consistency monitoring** — measure the actual degradation on *your* traffic before trusting it.

### Is this for me?

| Use case | Fit |
|---|---|
| Agents / RAG with a retrieval step (MCP `retrieve_original`, CCR keys) | ✅ Strong — lossy summary up front, full fidelity on demand |
| Summarization, sentiment, topic, "gist" over long context | ✅ Good |
| Cost control on exploratory analytics over large JSON | 🟡 With `--safe-mode` or careful A/B validation |
| Exact lookups, audits, anything requiring every record/figure | ❌ Use `--safe-mode` or don't compress |

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
  --safe-mode                  Never drop JSON records or prose sentences
```

### Safe mode

`--safe-mode` (or `CONTEXTLY_SAFE_MODE=true`) guarantees the model still sees **every** JSON record and **every** prose sentence. The lossy compressors (`json_smart` record sampling, `prose` sentence dropping) are removed from the routing chain, leaving only `code` (which strips comments and blank lines while preserving all logic) and `passthrough`. You trade most of the token savings for full fidelity — the right default when wrong answers are worse than expensive ones.

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
| `CONTEXTLY_SAFE_MODE` | `false` | Disable lossy compressors (keep every JSON record / prose sentence) |
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

| Compressor | Triggers on | Algorithm |
|---|---|---|
| `json_smart` | Valid JSON objects/arrays | **Samples a representative subset of records** (MinHash clustering + stratified sampling, keeps outliers), then elides null/empty fields, rounds floats, truncates long strings |
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
3. Both responses are scored two ways: word-level ROUGE-1 F1 (precision/recall harmonic mean) and **numeric consistency** (fraction of figures preserved).
4. The scores are stored in a ring buffer (1 000 samples max) and emitted to Prometheus.

A ROUGE-1 score of `1.0` means the compressed-context response is word-for-word identical to the original; a numeric-consistency of `1.0` means no figure was changed. The two are complementary — ROUGE-1 measures overall wording, numeric consistency guards the specific facts (counts, prices, dates) that lossy compression is most likely to corrupt silently. Streaming requests are excluded because buffering the response would negate the point of streaming.

> **Caveat:** ROUGE-1 compares the compressed-context answer to the *full-context* answer, not to ground truth — it tells you how much the answer *changed*, not whether it was right to begin with. For high-stakes use, pair it with an LLM-judge or task-specific exact-match eval on a held-out set.

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

## Benchmarks

Reproduce the savings-vs-retention numbers locally — no API key or network required:

```bash
python scripts/benchmark_quality.py --model gpt-4o
# narrow the aggressiveness with a query:
python scripts/benchmark_quality.py --model gpt-4o --query "find the unusual transaction"
```

The script runs each compressor on a representative fixture and prints token savings next to an information-retention metric (records retained for JSON, numeric facts for prose, signatures for code), so you can judge the trade-off rather than just the headline savings. Wire your own corpus in by editing the fixtures at the top of the script.

### Accuracy benchmark — does compression change the answers?

Retention is a proxy; the real question is whether a model gives the *right* answer with less context. `scripts/accuracy_benchmark.py` asks the same questions over a synthetic record set under three strategies — `full`, `compressed` (lossy), and `safe` — calls a real LLM, and grades answers against gold values:

```bash
# Any OpenAI-compatible endpoint (OpenAI, OpenRouter, local Ollama, or the proxy itself)
export OPENROUTER_API_KEY=sk-or-...
python scripts/accuracy_benchmark.py \
    --base-url https://openrouter.ai/api/v1 \
    --model google/gemma-4-31b-it:free

# Validate the harness offline, no API key (deterministic oracle model):
python scripts/accuracy_benchmark.py --self-test
```

The harness is verified offline on every push (the **Demo & Accuracy** workflow). To produce **real LLM numbers**, add an `OPENROUTER_API_KEY` repository secret and run that workflow manually (Actions → *Demo & Accuracy Benchmark* → *Run workflow*); results are written to the run summary.

Harness self-test output (deterministic *oracle* model — illustrates the mechanism, **not** real-LLM accuracy; run the workflow for live figures):

| Strategy | Accuracy | Mean context tokens |
|---|---:|---:|
| full | 100% | 5169 |
| compressed | 31% | 70 |
| safe | 100% | 5169 |

The pattern is the whole point: dropping records to shrink context to ~1% collapses lookup accuracy, while `safe` mode keeps every record and matches `full`. On *your* model and data the exact figures will differ — measure them before trusting compression in production.

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
