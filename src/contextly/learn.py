"""Failure learning — mine the A/B quality corpus for compression regressions.

The proxy's shadow A/B monitor scores every sampled request: the response the
model gave from *compressed* context vs the response from the *original* context
(word-level ROUGE-1 plus numeric consistency). When ``ab_log_path`` is set those
samples are persisted as JSONL — a record of where compression helped and where
it hurt.

``contextly learn`` reads that corpus and turns it into **corrections**: it groups
samples by ``(compressor, model)``, finds the combinations whose mean quality (or
numeric fidelity) falls below threshold — the *failures* — and emits concrete,
ranked recommendations (use safe mode, drop a lossy compressor, raise the A/B
sample rate to gather more evidence). This closes the learn-from-failures loop
without any external service: the data is local, the analysis is stdlib-only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

# Below this mean ROUGE-1 quality a (compressor, model) combo is a "failure".
DEFAULT_MIN_QUALITY = 0.7
# Below this mean numeric-consistency we flag silent factual corruption even when
# the overall quality score looks healthy.
DEFAULT_MIN_NUMERIC = 0.9
# Groups with fewer samples than this are reported as low-confidence only.
MIN_CONFIDENT_SAMPLES = 5


@dataclass(frozen=True)
class Recommendation:
    """A single actionable correction derived from observed failures."""

    target: str  # "compressor/model"
    severity: str  # "high" | "medium" | "low"
    issue: str
    action: str
    samples: int
    mean_quality: float
    mean_numeric: float


@dataclass
class LearningReport:
    """Aggregate analysis of an A/B quality corpus."""

    samples_total: int = 0
    groups_analyzed: int = 0
    recommendations: list[Recommendation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples_total": self.samples_total,
            "groups_analyzed": self.groups_analyzed,
            "recommendations": [
                {
                    "target": r.target,
                    "severity": r.severity,
                    "issue": r.issue,
                    "action": r.action,
                    "samples": r.samples,
                    "mean_quality": r.mean_quality,
                    "mean_numeric": r.mean_numeric,
                }
                for r in self.recommendations
            ],
        }


def load_samples(log_path: str) -> list[dict[str, Any]]:
    """Read the JSONL A/B corpus at *log_path*.

    Malformed lines are skipped rather than aborting the analysis — a partially
    flushed log should still be mineable.

    Raises:
        FileNotFoundError: If the log file does not exist.
    """
    path = Path(log_path)
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except ValueError:
                continue
    return samples


def _group_key(sample: dict[str, Any]) -> str:
    return f"{sample.get('compressor', 'unknown')}/{sample.get('model', 'unknown')}"


def analyze(
    samples: list[dict[str, Any]],
    *,
    min_quality: float = DEFAULT_MIN_QUALITY,
    min_numeric: float = DEFAULT_MIN_NUMERIC,
) -> LearningReport:
    """Mine *samples* for failing compressor/model combos and emit corrections.

    Args:
        samples: A/B records (dicts with compressor, model, quality_score,
            numeric_consistency).
        min_quality: Mean ROUGE-1 quality below which a group is a failure.
        min_numeric: Mean numeric-consistency below which a group is flagged for
            factual corruption.

    Returns:
        A :class:`LearningReport` with recommendations sorted by severity then
        sample count (most evidence first).
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for s in samples:
        if "quality_score" not in s:
            continue
        groups.setdefault(_group_key(s), []).append(s)

    report = LearningReport(samples_total=len(samples), groups_analyzed=len(groups))

    for target, rows in groups.items():
        compressor = rows[0].get("compressor", "unknown")
        n = len(rows)
        q = mean(float(r.get("quality_score", 1.0)) for r in rows)
        num = mean(float(r.get("numeric_consistency", 1.0)) for r in rows)

        low_quality = q < min_quality
        low_numeric = num < min_numeric
        if not (low_quality or low_numeric):
            continue

        confident = n >= MIN_CONFIDENT_SAMPLES
        if not confident:
            severity = "low"
        elif low_quality:
            severity = "high" if q < min_quality - 0.15 else "medium"
        else:
            # Confident numeric corruption with otherwise-healthy quality: the
            # answer reads fine but a figure is wrong — quietly dangerous.
            severity = "medium"

        issues: list[str] = []
        if low_quality:
            issues.append(f"mean quality {q:.2f} < {min_quality:.2f}")
        if low_numeric:
            issues.append(f"numeric fidelity {num:.2f} < {min_numeric:.2f}")

        if not confident:
            action = (
                f"only {n} sample(s) — raise ab_sample_rate to gather more evidence before acting"
            )
        elif low_numeric and not low_quality:
            action = (
                f"compressor '{compressor}' is dropping numbers; run this workload in "
                f"safe_mode (or disable lossy record/sentence sampling) to preserve figures"
            )
        else:
            action = (
                f"compression with '{compressor}' degraded answers here; enable safe_mode "
                f"for this model or stop routing this content type to '{compressor}'"
            )

        report.recommendations.append(
            Recommendation(
                target=target,
                severity=severity,
                issue="; ".join(issues),
                action=action,
                samples=n,
                mean_quality=round(q, 4),
                mean_numeric=round(num, 4),
            )
        )

    _severity_rank = {"high": 0, "medium": 1, "low": 2}
    report.recommendations.sort(key=lambda r: (_severity_rank[r.severity], -r.samples))
    return report


def learn_from_log(
    log_path: str,
    *,
    min_quality: float = DEFAULT_MIN_QUALITY,
    min_numeric: float = DEFAULT_MIN_NUMERIC,
) -> LearningReport:
    """Convenience: load the corpus at *log_path* and analyze it."""
    return analyze(
        load_samples(log_path),
        min_quality=min_quality,
        min_numeric=min_numeric,
    )
