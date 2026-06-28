"""Unit tests for failure learning (contextly learn)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contextly.learn import analyze, learn_from_log, load_samples


def _sample(
    compressor: str = "prose",
    model: str = "gpt-4o",
    quality: float = 0.9,
    numeric: float = 1.0,
) -> dict:
    return {
        "ts": 1.0,
        "model": model,
        "compressor": compressor,
        "original_chars": 1000,
        "compressed_chars": 400,
        "quality_score": quality,
        "numeric_consistency": numeric,
    }


# ── analyze ─────────────────────────────────────────────────────────────────────


def test_no_failures_when_quality_high() -> None:
    samples = [_sample(quality=0.95) for _ in range(10)]
    report = analyze(samples)
    assert report.samples_total == 10
    assert report.groups_analyzed == 1
    assert report.recommendations == []


def test_low_quality_combo_flagged() -> None:
    samples = [_sample(compressor="json_smart", quality=0.4) for _ in range(8)]
    report = analyze(samples)
    assert len(report.recommendations) == 1
    rec = report.recommendations[0]
    assert rec.target == "json_smart/gpt-4o"
    assert rec.severity == "high"  # 0.4 < 0.7 - 0.15
    assert "quality" in rec.issue
    assert "safe_mode" in rec.action


def test_medium_severity_near_threshold() -> None:
    samples = [_sample(quality=0.65) for _ in range(8)]  # between 0.55 and 0.7
    report = analyze(samples)
    assert report.recommendations[0].severity == "medium"


def test_low_confidence_when_few_samples() -> None:
    samples = [_sample(quality=0.3) for _ in range(2)]
    report = analyze(samples)
    rec = report.recommendations[0]
    assert rec.severity == "low"
    assert "gather more evidence" in rec.action


def test_numeric_corruption_flagged_even_with_ok_quality() -> None:
    samples = [_sample(quality=0.95, numeric=0.5) for _ in range(8)]
    report = analyze(samples)
    rec = report.recommendations[0]
    assert "numeric fidelity" in rec.issue
    assert "dropping numbers" in rec.action
    assert rec.severity == "medium"  # confident factual corruption


def test_groups_separated_by_compressor_and_model() -> None:
    samples = [_sample(compressor="prose", quality=0.4) for _ in range(6)]
    samples += [_sample(compressor="code", model="o1", quality=0.95) for _ in range(6)]
    report = analyze(samples)
    assert report.groups_analyzed == 2
    # only the failing group produces a recommendation
    assert len(report.recommendations) == 1
    assert report.recommendations[0].target == "prose/gpt-4o"


def test_recommendations_sorted_by_severity() -> None:
    samples = [_sample(compressor="a", quality=0.3) for _ in range(8)]  # high
    samples += [_sample(compressor="b", quality=0.65) for _ in range(8)]  # medium
    report = analyze(samples)
    severities = [r.severity for r in report.recommendations]
    assert severities == ["high", "medium"]


def test_custom_thresholds() -> None:
    samples = [_sample(quality=0.8) for _ in range(8)]
    # default min_quality 0.7 → no issue; raise to 0.9 → flagged
    assert analyze(samples).recommendations == []
    assert len(analyze(samples, min_quality=0.9).recommendations) == 1


def test_samples_without_quality_score_ignored() -> None:
    samples = [{"compressor": "prose", "model": "gpt-4o"}]
    report = analyze(samples)
    assert report.groups_analyzed == 0


# ── load_samples / learn_from_log ───────────────────────────────────────────────


def test_load_samples_skips_malformed_lines(tmp_path: Path) -> None:
    log = tmp_path / "ab.jsonl"
    log.write_text(
        json.dumps(_sample()) + "\n" + "not json\n" + json.dumps(_sample(quality=0.4)) + "\n"
    )
    samples = load_samples(str(log))
    assert len(samples) == 2


def test_load_samples_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_samples(str(tmp_path / "nope.jsonl"))


def test_learn_from_log_end_to_end(tmp_path: Path) -> None:
    log = tmp_path / "ab.jsonl"
    log.write_text("".join(json.dumps(_sample(quality=0.3)) + "\n" for _ in range(8)))
    report = learn_from_log(str(log))
    assert report.samples_total == 8
    assert len(report.recommendations) == 1
    assert report.to_dict()["recommendations"][0]["severity"] == "high"
