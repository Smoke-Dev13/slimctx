"""Integration tests for the `contextly bench` CLI command."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from contextly.cli import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def prose_payload(tmp_path: pytest.TempPathFactory) -> str:
    """Path to a temp JSON file with prose messages."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that summarises technical content clearly. "
                "Always be concise and accurate. Avoid padding and filler phrases. "
                "Focus on the most important details and key takeaways from the input."
            ),
        },
        {
            "role": "user",
            "content": (
                "Machine learning models require careful hyperparameter tuning to achieve "
                "optimal performance. Regularisation techniques such as L1 and L2 penalties "
                "prevent overfitting by constraining the model weights. Ensemble methods "
                "like random forests and gradient boosting combine many weak learners to "
                "produce more robust predictions. Transfer learning leverages representations "
                "learned from large pre-trained models and fine-tunes them for specific tasks. "
                "Cross-validation provides a reliable estimate of generalisation performance "
                "by evaluating the model on multiple held-out folds. Data preprocessing steps "
                "such as normalisation, missing-value imputation, and outlier removal have a "
                "significant impact on model quality and training stability. "
            )
            * 3,
        },
    ]
    path = tmp_path / "payload.json"  # type: ignore[operator]
    path.write_text(json.dumps(messages), encoding="utf-8")
    return str(path)


@pytest.fixture
def chat_payload(tmp_path: pytest.TempPathFactory) -> str:
    """Path to a temp JSON file with a full chat payload object."""
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "What is 2 + 2?"},
        ],
    }
    path = tmp_path / "chat.json"  # type: ignore[operator]
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


@pytest.fixture
def json_payload(tmp_path: pytest.TempPathFactory) -> str:
    """Path to a temp JSON file with JSON-content messages."""
    messages = [
        {
            "role": "user",
            "content": json.dumps(
                [{"id": i, "name": f"user_{i}", "active": True, "score": None} for i in range(50)]
            ),
        }
    ]
    path = tmp_path / "json_payload.json"  # type: ignore[operator]
    path.write_text(json.dumps(messages), encoding="utf-8")
    return str(path)


# ── Exit codes ────────────────────────────────────────────────────────────────


def test_bench_exits_zero_with_valid_file(runner: CliRunner, prose_payload: str) -> None:
    result = runner.invoke(main, ["bench", prose_payload])
    assert result.exit_code == 0, result.output


def test_bench_exits_nonzero_with_missing_file(runner: CliRunner) -> None:
    result = runner.invoke(main, ["bench", "/nonexistent/payload.json"])
    assert result.exit_code != 0


# ── Output content ────────────────────────────────────────────────────────────


def test_bench_output_contains_total(runner: CliRunner, prose_payload: str) -> None:
    result = runner.invoke(main, ["bench", prose_payload])
    assert "TOTAL" in result.output


def test_bench_output_contains_saved_pct(runner: CliRunner, prose_payload: str) -> None:
    result = runner.invoke(main, ["bench", prose_payload])
    assert "saved" in result.output.lower()


def test_bench_output_contains_model(runner: CliRunner, prose_payload: str) -> None:
    result = runner.invoke(main, ["bench", prose_payload, "--model", "gpt-4o"])
    assert "gpt-4o" in result.output


def test_bench_output_model_from_payload(runner: CliRunner, chat_payload: str) -> None:
    result = runner.invoke(main, ["bench", chat_payload])
    assert "gpt-4o-mini" in result.output


def test_bench_output_has_header_columns(runner: CliRunner, prose_payload: str) -> None:
    result = runner.invoke(main, ["bench", prose_payload])
    assert "Original" in result.output
    assert "Compressed" in result.output
    assert "Saved" in result.output
    assert "Compressor" in result.output


def test_bench_output_row_count_matches_messages(runner: CliRunner, prose_payload: str) -> None:
    # 2 messages → row 0 and row 1 appear in output with fixed-width index prefix
    result = runner.invoke(main, ["bench", prose_payload])
    data_lines = [
        ln
        for ln in result.output.splitlines()
        if ln.startswith("   0  ") or ln.startswith("   1  ")
    ]
    assert len(data_lines) == 2


# ── Compressor selection ──────────────────────────────────────────────────────


def test_bench_json_content_uses_json_compressor(runner: CliRunner, json_payload: str) -> None:
    result = runner.invoke(main, ["bench", json_payload])
    assert "json_smart" in result.output


def test_bench_prose_content_uses_prose_or_passthrough(
    runner: CliRunner, prose_payload: str
) -> None:
    result = runner.invoke(main, ["bench", prose_payload])
    assert "prose" in result.output or "passthrough" in result.output


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_bench_full_chat_payload_object(runner: CliRunner, chat_payload: str) -> None:
    result = runner.invoke(main, ["bench", chat_payload])
    assert result.exit_code == 0


def test_bench_empty_messages_array(runner: CliRunner, tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "empty.json"  # type: ignore[operator]
    path.write_text("[]", encoding="utf-8")
    result = runner.invoke(main, ["bench", str(path)])
    assert result.exit_code != 0


def test_bench_invalid_json_file(runner: CliRunner, tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "bad.json"  # type: ignore[operator]
    path.write_text("{not valid json", encoding="utf-8")
    result = runner.invoke(main, ["bench", str(path)])
    assert result.exit_code != 0


def test_bench_message_with_empty_content(
    runner: CliRunner, tmp_path: pytest.TempPathFactory
) -> None:
    messages = [{"role": "user", "content": ""}]
    path = tmp_path / "empty_content.json"  # type: ignore[operator]
    path.write_text(json.dumps(messages), encoding="utf-8")
    result = runner.invoke(main, ["bench", str(path)])
    assert result.exit_code == 0
    assert "TOTAL" in result.output
