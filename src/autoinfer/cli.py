"""Autoinfer command-line entry.

``autoinfer validate <config.yaml>`` — parse and validate a run config.
``autoinfer run <config.yaml>``      — execute a run end-to-end.

The run wiring composes the full hybrid-policy stack against a single
``L1EngineAdapter`` for iteration zero. Multi-layer composition is a
session-four concern.
"""

from __future__ import annotations

import os
from pathlib import Path

import typer

from autoinfer.builder import build_runner
from autoinfer.config import RunConfig, load_config

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def validate(config_path: Path) -> None:
    """Parse and validate a run config without executing."""
    cfg = _load_checked(config_path)
    typer.echo(f"ok: {cfg.name}")
    typer.echo(f"  target: {cfg.target.kind}")
    typer.echo(f"  layers: {_enabled_layers(cfg)}")
    if cfg.layers.l1_engine is not None:
        typer.echo(f"  l1_engine.model: {cfg.layers.l1_engine.model}")
        typer.echo(f"  l1_engine.max_trials: {cfg.layers.l1_engine.max_trials}")


@app.command()
def run(
    config_path: Path,
    max_trials: int | None = typer.Option(None, help="Override layer max_trials."),
) -> None:
    """Execute a run end-to-end."""
    cfg = _load_checked(config_path)
    runner, ledger = build_runner(cfg, max_trials_override=max_trials)
    front = runner.run()
    typer.echo(f"finished: {len(ledger.entries())} trials, {len(front)} pareto entries")
    for i, entry in enumerate(front):
        typer.echo(f"  [{i}] {entry.trial_id} layer={entry.layer}")
        if entry.measurement is not None:
            typer.echo(
                "      "
                f"tok/s={entry.measurement.tokens_per_sec:.1f} "
                f"tpot_p99={entry.measurement.tpot_p99_ms:.1f}ms "
                f"hbm={entry.measurement.peak_hbm_gb:.1f}GiB "
                f"kl={entry.measurement.kl_divergence:.4f}"
            )


@app.command()
def print_config(config_path: Path) -> None:
    """Dump the validated config as JSON."""
    cfg = _load_checked(config_path)
    typer.echo(cfg.model_dump_json(indent=2))


def _load_checked(config_path: Path) -> RunConfig:
    if not config_path.exists():
        raise typer.BadParameter(f"config not found: {config_path}")
    try:
        return load_config(config_path)
    except Exception as e:
        typer.secho(f"config error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from e


def _enabled_layers(cfg: RunConfig) -> list[str]:
    out: list[str] = []
    if cfg.layers.l1_engine is not None:
        out.append("l1_engine")
    if cfg.layers.l2_topology is not None:
        out.append("l2_topology")
    if cfg.layers.l3_kernel is not None:
        out.append("l3_kernel")
    return out


def _api_key(var: str | None) -> str | None:
    return os.environ.get(var) if var else None


if __name__ == "__main__":
    app()
