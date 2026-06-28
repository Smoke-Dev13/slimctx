"""A/B quality monitor for measuring compression-induced response degradation.

Algorithm (shadow mode):
  For a configurable fraction of non-streaming requests (ab_sample_rate):
    1. Let the main request proceed normally with compressed context.
    2. In a background asyncio task, send the ORIGINAL (uncompressed) context
       to the same upstream and collect its response.
    3. Score the compressed-context response against the original-context
       response using word-level ROUGE-1 F1 (precision/recall harmonic mean).
    4. Store the result in an in-memory ring buffer.

Quality score (0.0-1.0):
  1.0 → compressed-context response is word-for-word identical to original.
  0.0 → no shared words at all.
  Mid → partial content overlap.

Counters track every request (not just sampled ones), enabling the /stats
endpoint to report real token-savings totals without sampling bias.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import httpx
import structlog

from contextly.metrics import observe_ab_sample

logger = structlog.get_logger(__name__)


# ── Quality scorer ─────────────────────────────────────────────────────────────


def _quality_score(reference: str, candidate: str) -> float:
    """Word-level ROUGE-1 F1 between *reference* and *candidate* texts.

    Uses Counter intersection to count overlapping token occurrences
    (duplicates matter), then computes precision, recall, and their
    harmonic mean.

    Args:
        reference: The LLM response produced from the original (full) context.
        candidate: The LLM response produced from the compressed context.

    Returns:
        Float in [0.0, 1.0] — higher means compressed context produced a
        response closer to the original.
    """
    ref_words = reference.lower().split()
    cand_words = candidate.lower().split()
    if not ref_words and not cand_words:
        return 1.0
    if not ref_words or not cand_words:
        return 0.0
    ref_counter: Counter[str] = Counter(ref_words)
    cand_counter: Counter[str] = Counter(cand_words)
    overlap = sum((ref_counter & cand_counter).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(cand_words)
    recall = overlap / len(ref_words)
    return 2.0 * precision * recall / (precision + recall)


_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _extract_numbers(text: str) -> Counter[str]:
    """Return a multiset of normalised numeric tokens found in *text*.

    Thousands separators are stripped and trailing zeros normalised so that
    "1,000", "1000", and "1000.00" compare equal. Numbers are exactly where
    lossy compression silently corrupts answers (counts, prices, dates), and
    word-level ROUGE-1 treats every digit-string as an interchangeable token.
    """
    numbers: Counter[str] = Counter()
    for raw in _NUMBER_RE.findall(text):
        cleaned = raw.replace(",", "")
        try:
            numbers[repr(float(cleaned))] += 1
        except ValueError:
            continue
    return numbers


def _numeric_consistency(reference: str, candidate: str) -> float:
    """Fraction of numbers in *reference* that also appear in *candidate*.

    Returns 1.0 when the reference contains no numbers (nothing to corrupt).
    A low score flags that compression changed a factual figure even when the
    overall ROUGE-1 score looks healthy.

    Args:
        reference: Response produced from the original (full) context.
        candidate: Response produced from the compressed context.

    Returns:
        Float in [0.0, 1.0]; 1.0 means every reference number is preserved.
    """
    ref_numbers = _extract_numbers(reference)
    if not ref_numbers:
        return 1.0
    cand_numbers = _extract_numbers(candidate)
    preserved = sum((ref_numbers & cand_numbers).values())
    return preserved / sum(ref_numbers.values())


def _percentile(sorted_values: list[float], p: float) -> float:
    """Return the *p*-th percentile from a pre-sorted list (0 ≤ p ≤ 100)."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    idx = max(0, min(n - 1, math.floor(p / 100.0 * n)))
    return sorted_values[idx]


def _extract_response_text(body: bytes) -> str:
    """Pull the assistant message text from an OpenAI-style JSON response body."""
    try:
        data: dict[str, Any] = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return ""
    choices = data.get("choices", [])
    if not choices:
        return ""
    content = choices[0].get("message", {}).get("content", "")
    return str(content) if content is not None else ""


# ── Data model ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ABSample:
    """Result of one shadow A/B quality comparison."""

    timestamp: float
    model: str
    compressor: str
    original_chars: int
    compressed_chars: int
    quality_score: float
    reference_response_len: int
    compressed_response_len: int
    # Fraction of numbers in the reference response preserved in the compressed
    # response (1.0 = none lost). Defaults to 1.0 for back-compat.
    numeric_consistency: float = 1.0


# ── Monitor ────────────────────────────────────────────────────────────────────


class ABMonitor:
    """Tracks compression quality via shadow A/B sampling.

    Two separate tracking paths:
      - ``record_request()``  — called on every compressed request; updates
        running counters for the /stats endpoint.
      - ``record_sample()``   — called only after a shadow A/B comparison
        completes; stores the full ABSample in the ring buffer.

    Both paths are thread-safe.

    Args:
        max_samples: Maximum number of A/B samples to retain in memory.
            Older samples are evicted when the ring is full (FIFO).
    """

    def __init__(self, max_samples: int = 1000, *, log_path: str | None = None) -> None:
        self._samples: deque[ABSample] = deque(maxlen=max_samples)
        self._lock = threading.Lock()
        # Optional persisted corpus mined by ``contextly learn``.
        self._log_path = Path(log_path) if log_path else None
        # Running totals (not capped by max_samples)
        self._total_requests: int = 0
        self._total_compressed: int = 0
        self._total_original_chars: int = 0
        self._total_compressed_chars: int = 0
        self._total_tokens_saved: int = 0
        self._total_dollars_saved: float = 0.0

    # ── Recording ─────────────────────────────────────────────────────────────

    def record_request(
        self,
        original_chars: int,
        compressed_chars: int,
        compressor_name: str,
        tokens_saved_estimate: int = 0,
        dollars_saved: float = 0.0,
    ) -> None:
        """Update running counters for a single chat/completions request.

        Called on every non-streaming request where content_router ran,
        regardless of whether A/B sampling fires.

        Args:
            original_chars: Total character length of all original messages.
            compressed_chars: Total character length of compressed messages.
            compressor_name: Name of the compressor that did the most work.
            tokens_saved_estimate: Estimated tokens saved (chars // 4 heuristic).
            dollars_saved: Estimated USD saved for this request.
        """
        _ = compressor_name  # reserved for future per-compressor counters
        with self._lock:
            self._total_requests += 1
            if compressed_chars < original_chars:
                self._total_compressed += 1
            self._total_original_chars += original_chars
            self._total_compressed_chars += compressed_chars
            self._total_tokens_saved += tokens_saved_estimate
            self._total_dollars_saved += dollars_saved

    def record_sample(self, sample: ABSample) -> None:
        """Append an A/B comparison result to the ring buffer.

        Also records the quality score in the Prometheus histogram.

        Args:
            sample: Completed ABSample to store.
        """
        with self._lock:
            self._samples.append(sample)
        self._append_log(sample)
        observe_ab_sample(compressor=sample.compressor, quality_score=sample.quality_score)
        logger.info(
            "ab_sample_recorded",
            compressor=sample.compressor,
            quality=round(sample.quality_score, 3),
            chars_saved=sample.original_chars - sample.compressed_chars,
        )

    def _append_log(self, sample: ABSample) -> None:
        """Best-effort append of *sample* to the JSONL learning corpus.

        Persistence failures never break the request path — the in-memory ring
        buffer remains the source of truth for the live dashboard.
        """
        if self._log_path is None:
            return
        try:
            record = {
                "ts": round(sample.timestamp, 3),
                "model": sample.model,
                "compressor": sample.compressor,
                "original_chars": sample.original_chars,
                "compressed_chars": sample.compressed_chars,
                "quality_score": round(sample.quality_score, 4),
                "numeric_consistency": round(sample.numeric_consistency, 4),
            }
            line = json.dumps(record, separators=(",", ":")) + "\n"
            with self._lock:
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
                with self._log_path.open("a", encoding="utf-8") as fh:
                    fh.write(line)
        except OSError:
            logger.warning("ab_log_append_failed", exc_info=True)

    # ── Reporting ─────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return aggregate request-level counters for the /stats endpoint.

        Returns:
            Dict with requests_total, requests_compressed, chars_saved_total,
            compression_ratio_mean, and ab_samples_total.
        """
        with self._lock:
            total = self._total_requests
            compressed = self._total_compressed
            orig_chars = self._total_original_chars
            comp_chars = self._total_compressed_chars
            tokens_saved = self._total_tokens_saved
            dollars_saved = self._total_dollars_saved
            n_samples = len(self._samples)

        chars_saved = orig_chars - comp_chars
        ratio = round(comp_chars / orig_chars, 4) if orig_chars > 0 else 1.0
        return {
            "requests_total": total,
            "requests_compressed": compressed,
            "chars_saved_total": chars_saved,
            "tokens_saved_estimate_total": tokens_saved,
            "dollars_saved_total": round(dollars_saved, 6),
            "compression_ratio_mean": ratio,
            "ab_samples_total": n_samples,
        }

    def quality_report(self) -> dict[str, Any]:
        """Return a detailed quality breakdown from all stored A/B samples.

        Returns:
            Dict with sample counts, score distribution, savings summary,
            and per-compressor breakdowns.  Returns a minimal "no data"
            structure when no samples are available.
        """
        with self._lock:
            samples = list(self._samples)

        n = len(samples)
        if n == 0:
            return {
                "samples_total": 0,
                "quality": None,
                "chars_saved": None,
                "by_compressor": {},
            }

        scores = sorted(s.quality_score for s in samples)
        numeric_scores = [s.numeric_consistency for s in samples]
        chars_saved_list = [s.original_chars - s.compressed_chars for s in samples]

        by_compressor: dict[str, list[ABSample]] = {}
        for s in samples:
            by_compressor.setdefault(s.compressor, []).append(s)

        return {
            "samples_total": n,
            "quality": {
                "mean": round(mean(scores), 4),
                "p10": round(_percentile(scores, 10), 4),
                "p50": round(_percentile(scores, 50), 4),
                "p90": round(_percentile(scores, 90), 4),
            },
            "numeric_consistency": {
                "mean": round(mean(numeric_scores), 4),
                "p10": round(_percentile(sorted(numeric_scores), 10), 4),
            },
            "chars_saved": {
                "mean": round(mean(chars_saved_list), 1),
                "total": sum(chars_saved_list),
            },
            "by_compressor": {
                k: {
                    "samples": len(v),
                    "mean_quality": round(mean(s.quality_score for s in v), 4),
                    "mean_numeric_consistency": round(mean(s.numeric_consistency for s in v), 4),
                }
                for k, v in sorted(by_compressor.items())
            },
        }

    def __len__(self) -> int:
        with self._lock:
            return len(self._samples)


# ── Shadow A/B coroutine ──────────────────────────────────────────────────────


async def run_shadow_ab(
    *,
    http_client: httpx.AsyncClient,
    upstream_url: str,
    headers: dict[str, str],
    original_payload: dict[str, Any],
    compressed_response_text: str,
    model: str,
    compressor_name: str,
    original_chars: int,
    compressed_chars: int,
    ab_monitor: ABMonitor,
) -> None:
    """Send original (uncompressed) context to upstream; compare with compressed response.

    This coroutine is intended to run as a fire-and-forget ``asyncio.create_task()``.
    All exceptions are caught so that failures never affect the main request path.

    Args:
        http_client: Shared async HTTP client (connection pool).
        upstream_url: Full upstream endpoint URL.
        headers: Request headers forwarded from the original request.
        original_payload: Chat payload with the ORIGINAL (uncompressed) messages.
        compressed_response_text: Text the LLM produced when given compressed context.
        model: Model identifier (for the ABSample record).
        compressor_name: Which compressor was used.
        original_chars: Total chars of original messages (for savings tracking).
        compressed_chars: Total chars of compressed messages.
        ab_monitor: Monitor to record the ABSample into.
    """
    try:
        ref_resp = await http_client.post(upstream_url, headers=headers, json=original_payload)
        reference_text = _extract_response_text(ref_resp.content)
        score = _quality_score(reference_text, compressed_response_text)
        numeric = _numeric_consistency(reference_text, compressed_response_text)
        ab_monitor.record_sample(
            ABSample(
                timestamp=time.time(),
                model=model,
                compressor=compressor_name,
                original_chars=original_chars,
                compressed_chars=compressed_chars,
                quality_score=score,
                reference_response_len=len(reference_text),
                compressed_response_len=len(compressed_response_text),
                numeric_consistency=numeric,
            )
        )
    except Exception:
        logger.warning("ab_shadow_request_failed", exc_info=True)
