from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from autoinfer.cli import app

_EXAMPLE = Path(__file__).parent.parent / "examples/qwen3-8b-l1-slice/config.yaml"

runner = CliRunner()


def test_validate_example_config() -> None:
    result = runner.invoke(app, ["validate", str(_EXAMPLE)])
    assert result.exit_code == 0, result.output
    assert "ok:" in result.output
    assert "qwen3-8b-l1-slice" in result.output


def test_validate_missing_file() -> None:
    result = runner.invoke(app, ["validate", "/does/not/exist.yaml"])
    assert result.exit_code != 0


def test_validate_bad_config(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("this: is\n  not: a valid: config\n")
    result = runner.invoke(app, ["validate", str(bad)])
    assert result.exit_code != 0


def test_print_config_emits_json() -> None:
    result = runner.invoke(app, ["print-config", str(_EXAMPLE)])
    assert result.exit_code == 0, result.output
    assert '"name"' in result.output
    assert "qwen3-8b-l1-slice" in result.output
