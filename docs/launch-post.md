# Launch post drafts

Figures below are from a real run — Llama 3.3 70B (Groq), 13 record-lookup
questions over a 120-record JSON set. Re-run with your own model/data via the
**Demo & Accuracy** workflow (Actions → *Demo & Accuracy Benchmark* → *Run
workflow*) and update the numbers if you change the setup.

---

## Show HN (news.ycombinator.com/submit)

**Title:** Show HN: Contextly – a proxy that losslessly halves JSON prompt tokens for LLMs

**URL:** https://github.com/Smoke-Dev13/slimctx

**Text:**

I kept paying for the same giant JSON context on every LLM call, so I built
Contextly: an OpenAI-compatible proxy you drop in front of any endpoint. The
insight is that a JSON array of records spends most of its tokens repeating the
same field names in every row. Contextly rewrites it into a columnar table
(field names once, then rows) — **losslessly**. Every record survives, the model
still answers exact lookups, and you spend ~half the tokens. No app changes.

I went down the lossy path first (sample a representative subset of records) and
benchmarked it honestly: on Llama 3.3 70B over 13 record-lookups, record
sampling scored **0% accuracy** at 99% fewer tokens — full context was 92%. If
you drop the record someone asks about, you can't answer. So sampling is now
opt-in, and the **default is the lossless table**: same answers as full context
(by construction) at −58% tokens. Lossy summarization stays available for prose
and gist workloads, with a `--safe-mode` switch and shadow A/B (ROUGE-1 + a
numeric-consistency check) to measure quality on your own traffic.

The pitch isn't "free tokens, trust us" — it's "lossless where it counts,
measured where it isn't."

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

- X/Twitter: "Contextly: a drop-in LLM proxy that rewrites JSON arrays into a columnar table — losslessly. −58% tokens, every record kept, lookups still work. (Lossy record-sampling? I benchmarked it at 0% on lookups, so it's opt-in.) MIT."
- LinkedIn: "Most prompt-compression tools quote token savings and stay quiet about quality. Contextly's default is lossless, and it ships the accuracy benchmark that proves it."
