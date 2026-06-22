# Launch post drafts

Fill the `<…>` placeholders with the real figures from the **Demo & Accuracy**
workflow (Actions → *Demo & Accuracy Benchmark* → *Run workflow*) before posting.

---

## Show HN (news.ycombinator.com/submit)

**Title:** Show HN: Contextly – a transparent proxy that compresses LLM prompts (and tells you when it hurts)

**URL:** https://github.com/Smoke-Dev13/slimctx

**Text:**

I kept paying for the same giant JSON and document context on every LLM call, so
I built Contextly: an OpenAI-compatible proxy you drop in front of any endpoint.
It compresses each message (JSON via MinHash record sampling, prose via
extractive summarization, code via comment/whitespace stripping) and passes the
response straight through — no app changes.

The honest part: this compression is **lossy**. It drops records and sentences,
so it's great for "what's the gist / rough totals" and dangerous for exact
lookups. Instead of hiding that, Contextly ships the tools to measure it:

- an offline retention benchmark (tokens saved vs. info kept),
- a real accuracy benchmark that grades answers under full / compressed / safe
  context against gold values,
- built-in shadow A/B in the proxy (ROUGE-1 + a numeric-consistency check), and
- a `--safe-mode` that never drops a record when you need full fidelity.

On <MODEL> over <N> lookup questions I measured: full context <ACC_FULL>%
accuracy, lossy compressed <ACC_COMPRESSED>% at <TOKENS_SAVED>% fewer tokens,
safe mode back to <ACC_SAFE>%. So the pitch isn't "free tokens" — it's "here's
exactly what the trade costs, decide per workload."

Stack: FastAPI + httpx, reversible in-memory store with retrieval-by-key, MCP
server mode, Prometheus metrics, Docker. MIT. Feedback very welcome —
especially on where lossy context is/ isn't acceptable for you.

---

## Reddit r/LocalLLaMA / r/MachineLearning

**Title:** I built a transparent LLM proxy that compresses prompts — and benchmarks how much accuracy you lose

Body: same as above, lead with the accuracy table (full vs compressed vs safe)
as an image or code block, since this audience wants the numbers first.

---

## One-line summaries

- X/Twitter: "Contextly: drop-in proxy that cuts LLM input tokens by <TOKENS_SAVED>% — and a benchmark that shows the accuracy you trade for it. Lossy by design, honest by default. MIT."
- LinkedIn: "Most prompt-compression tools quote token savings and stay quiet about quality. Contextly ships the accuracy benchmark too."
