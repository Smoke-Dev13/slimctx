#!/usr/bin/env python3
"""Real accuracy benchmark: does compression change the LLM's answers?

Unlike ``benchmark_quality.py`` (which measures *information retention* offline),
this script calls a real LLM and grades answer correctness against gold values,
comparing three context strategies on the same questions:

    full        — every record sent to the model (baseline accuracy)
    compressed  — records routed through Contextly's lossy JSON compressor
    safe        — Contextly safe mode (records preserved, no dropping)

It reports accuracy *and* mean context tokens per strategy, so the
token-savings-vs-correctness trade-off is measured rather than asserted.

The model is reached over any OpenAI-compatible endpoint (OpenAI, OpenRouter,
a local Ollama/llama.cpp server, or the Contextly proxy itself). The API key is
read from an environment variable and is never written to disk.

Examples:
    # OpenRouter (free model)
    export OPENROUTER_API_KEY=sk-or-...
    python scripts/accuracy_benchmark.py \\
        --base-url https://openrouter.ai/api/v1 \\
        --model google/gemma-4-31b-it:free \\
        --api-key-env OPENROUTER_API_KEY

    # Offline harness check (no network, deterministic oracle "model")
    python scripts/accuracy_benchmark.py --self-test
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass

import httpx

from contextly.compressors.json_smart import JsonSmartCompressor
from contextly.compressors.json_table import JsonTableCompressor
from contextly.tokenizer.registry import get_tokenizer

# ── Synthetic dataset (deterministic) ───────────────────────────────────────────

_CITIES = ["Reykjavik", "Montevideo", "Gaborone", "Tbilisi", "Paramaribo", "Vientiane"]
_PLANS = ["bronze", "silver", "gold", "platinum"]
_STATUSES = ["active", "paused", "churned", "trial"]


def _build_records(n: int, seed: int = 7) -> list[dict[str, object]]:
    """Deterministic record set with hard-to-guess field values."""
    rng = random.Random(seed)
    records = []
    for i in range(n):
        records.append(
            {
                "id": 1000 + i,
                "name": f"user_{i:03d}",
                "city": rng.choice(_CITIES),
                "plan": rng.choice(_PLANS),
                "status": rng.choice(_STATUSES),
                "mrr": rng.choice([19, 49, 99, 199, 499]),
            }
        )
    return records


@dataclass(frozen=True)
class Question:
    kind: str  # "lookup" | "aggregate"
    prompt: str
    gold: str


def _build_questions(
    records: list[dict[str, object]], n_lookup: int, seed: int = 13
) -> list[Question]:
    rng = random.Random(seed)
    questions: list[Question] = []
    # Lookup questions target one specific record — the lossy worst case.
    for rec in rng.sample(records, min(n_lookup, len(records))):
        field = rng.choice(["city", "plan", "status", "mrr"])
        questions.append(
            Question(
                kind="lookup",
                prompt=(
                    f"In the JSON records, what is the {field} of the record with id {rec['id']}? "
                    f"Answer with only the {field} value, nothing else."
                ),
                gold=str(rec[field]),
            )
        )
    # A couple of aggregate questions — where a representative sample can still work.
    most_common_plan = max(_PLANS, key=lambda p: sum(1 for r in records if r["plan"] == p))
    questions.append(
        Question(
            kind="aggregate",
            prompt="Which plan value appears most frequently across the JSON records? Answer with one word.",
            gold=most_common_plan,
        )
    )
    return questions


# ── Grading ─────────────────────────────────────────────────────────────────────


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _is_correct(answer: str, gold: str) -> bool:
    a, g = _normalise(answer), _normalise(gold)
    if not g:
        return False
    # Token-level containment handles "The city is Tbilisi." vs "Tbilisi".
    return g in a or g in a.split()


# ── Model client ────────────────────────────────────────────────────────────────


class RateLimited(RuntimeError):
    """Raised when the provider keeps returning 429 after all retries."""


class Client:
    """Minimal OpenAI-compatible chat client with 429 retry/backoff."""

    def __init__(self, base_url: str, api_key: str, model: str, sleep: float) -> None:
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._model = model
        self._sleep = sleep
        self._http = httpx.Client(timeout=120)

    def ask(self, system: str, user: str) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": 30,
            "temperature": 0,
        }
        for attempt in range(8):
            resp = self._http.post(self._url, headers=self._headers, json=payload)
            if resp.status_code == 200:
                time.sleep(self._sleep)
                return str(resp.json()["choices"][0]["message"]["content"]).strip()
            if resp.status_code in (429, 402, 503):
                time.sleep(min(30.0, 3.0 * (attempt + 1)))
                continue
            raise RuntimeError(f"LLM error {resp.status_code}: {resp.text[:200]}")
        raise RateLimited("rate-limited after retries")


class OracleClient:
    """Offline stand-in: answers correctly only if the gold value is in context.

    Used by --self-test to verify the harness end-to-end without a network call.
    It deterministically models the core hypothesis (you can only answer a lookup
    if the relevant record survived compression), so the self-test exercises the
    full pipeline and grading logic.
    """

    def __init__(self) -> None:
        self.calls = 0

    def ask(self, system: str, user: str) -> str:
        self.calls += 1
        # The system prompt carries the context; the user prompt carries the gold.
        gold = user.split("__GOLD__:", 1)[1].strip() if "__GOLD__:" in user else ""
        return gold if gold and gold in system else "unknown"


# ── Arms ────────────────────────────────────────────────────────────────────────

_SYSTEM = "You answer questions strictly from the provided JSON records. Be terse."


@dataclass
class ArmResult:
    name: str
    correct: int
    total: int
    tokens_sum: int

    @property
    def accuracy(self) -> float:
        return 100.0 * self.correct / self.total if self.total else 0.0

    @property
    def mean_tokens(self) -> int:
        return round(self.tokens_sum / self.total) if self.total else 0


def _context_for_arm(arm: str, records: list[dict[str, object]], query: str) -> str:
    raw = json.dumps(records)
    if arm == "full":
        return raw
    if arm == "table":
        # Default lossless compression: every record kept, fewer tokens.
        return JsonTableCompressor().compress(raw, query).content
    if arm == "sampled":
        # Opt-in lossy record sampling — kept for comparison.
        return JsonSmartCompressor().compress(raw, query).content
    raise ValueError(arm)


def _run(
    arms: list[str],
    questions: list[Question],
    records: list[dict[str, object]],
    client: object,
    model: str,
    self_test: bool,
) -> list[ArmResult]:
    tok = get_tokenizer(model if not self_test else "gpt-4o")
    results = {a: ArmResult(a, 0, 0, 0) for a in arms}
    for q in questions:
        # Dedup identical contexts (full and safe are the same payload for JSON)
        # so we never spend two API calls on one prompt.
        answer_cache: dict[str, str] = {}
        pending = {a: ArmResult(a, 0, 0, 0) for a in arms}
        try:
            for arm in arms:
                context = _context_for_arm(arm, records, q.prompt)
                system = f"{_SYSTEM}\n\nJSON records:\n{context}"
                user = q.prompt
                if self_test:
                    user += f"\n__GOLD__:{q.gold}"
                if context not in answer_cache:
                    answer_cache[context] = client.ask(system, user)  # type: ignore[attr-defined]
                answer = answer_cache[context]
                pending[arm].total = 1
                pending[arm].tokens_sum = tok.count(system)
                pending[arm].correct = 1 if _is_correct(answer, q.gold) else 0
        except RateLimited:
            print(
                f"warning: rate-limited; reporting {results[arms[0]].total} completed questions",
                file=sys.stderr,
            )
            break
        # Commit the question only if every arm succeeded (keeps arms comparable).
        for arm in arms:
            results[arm].total += pending[arm].total
            results[arm].tokens_sum += pending[arm].tokens_sum
            results[arm].correct += pending[arm].correct
    return [results[a] for a in arms]


def _print_report(arms: list[ArmResult], model: str, n_q: int) -> None:
    full_tokens = next((a.mean_tokens for a in arms if a.name == "full"), 0)
    print(f"\n# Accuracy benchmark (model={model!r}, {n_q} questions)\n")
    print("| Strategy | Accuracy | Mean context tokens | Tokens vs full |")
    print("|---|---:|---:|---:|")
    for a in arms:
        delta = (
            f"-{100 * (1 - a.mean_tokens / full_tokens):.0f}%"
            if full_tokens and a.name != "full"
            else "—"
        )
        print(
            f"| {a.name} | {a.accuracy:.0f}% ({a.correct}/{a.total}) | {a.mean_tokens} | {delta} |"
        )
    print(
        "\n> `table` is the lossless default (columnar JSON, every record kept) — it "
        "should match `full` accuracy at far fewer tokens. `sampled` is the opt-in lossy "
        "record sampler; it saves the most tokens but breaks record-level lookups, so "
        "reserve it for gist/aggregate workloads."
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    p.add_argument("--model", default="google/gemma-4-31b-it:free")
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--records", type=int, default=120)
    p.add_argument("--lookups", type=int, default=12)
    p.add_argument(
        "--sleep", type=float, default=1.5, help="Pause between calls (free-tier friendly)"
    )
    p.add_argument("--arms", default="full,table,sampled")
    p.add_argument(
        "--self-test", action="store_true", help="Run offline with a deterministic oracle model"
    )
    args = p.parse_args()

    records = _build_records(args.records)
    questions = _build_questions(records, args.lookups)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    if args.self_test:
        client: object = OracleClient()
        results = _run(arms, questions, records, client, args.model, self_test=True)
        _print_report(results, "oracle(self-test)", len(questions))
        by = {r.name: r for r in results}
        # Lossless table must preserve every answer; lossy sampling cannot beat full.
        assert by["table"].accuracy == by["full"].accuracy, "lossless table changed an answer"
        assert by["full"].accuracy >= by["sampled"].accuracy, "harness invariant broken"
        print("\nself-test OK")
        return 0

    key = os.environ.get(args.api_key_env, "")
    if not key:
        print(f"error: ${args.api_key_env} is not set", file=sys.stderr)
        return 2
    client = Client(args.base_url, key, args.model, args.sleep)
    results = _run(arms, questions, records, client, args.model, self_test=False)
    _print_report(results, args.model, len(questions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
