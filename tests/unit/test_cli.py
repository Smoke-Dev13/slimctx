"""Unit tests for the Click CLI using CliRunner (no server started)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from contextly.cli import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_version(runner: CliRunner) -> None:
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_bench_exits_1(runner: CliRunner, tmp_path: object) -> None:
    import pathlib
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        pathlib.Path(f.name).write_text("{}")
        tmp = f.name
    result = runner.invoke(main, ["bench", tmp])
    pathlib.Path(tmp).unlink(missing_ok=True)
    assert result.exit_code == 1


def test_stats_no_server(runner: CliRunner) -> None:
    result = runner.invoke(main, ["stats", "--port", "19999"])
    assert result.exit_code == 1


def test_proxy_calls_run(runner: CliRunner) -> None:
    """proxy command constructs Config and calls run()."""
    mock_run = MagicMock()
    with patch("contextly.server.run", mock_run):
        runner.invoke(
            main,
            ["proxy", "--upstream", "openai", "--upstream-api-key", "sk-test", "--port", "5555"],
            catch_exceptions=False,
        )
    mock_run.assert_called_once()
    called_config = mock_run.call_args[0][0]
    assert called_config.port == 5555
    assert called_config.upstream_api_key == "sk-test"


def test_proxy_no_compress_flag(runner: CliRunner) -> None:
    mock_run = MagicMock()
    with patch("contextly.server.run", mock_run):
        runner.invoke(main, ["proxy", "--no-compress"])
    called_config = mock_run.call_args[0][0]
    assert called_config.compression_enabled is False


def test_proxy_default_upstream_is_openai(runner: CliRunner) -> None:
    mock_run = MagicMock()
    with patch("contextly.server.run", mock_run):
        runner.invoke(main, ["proxy"])
    called_config = mock_run.call_args[0][0]
    assert str(called_config.upstream) == "openai"
