"""Autoinfer command-line entry.

``autoinfer validate <config.yaml>`` — parse and validate a run config.
``autoinfer run <config.yaml>``      — execute a run end-to-end.

The run wiring composes the full hybrid-policy stack against a single
``L1EngineAdapter`` for iteration zero. Multi-layer composition is a
session-four concern.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import typer

from autoinfer.builder import build_runner
from autoinfer.config import RunConfig, load_config
from autoinfer.telemetry import (
    build_run_summary,
    capture_hw_context,
    write_results_tsv,
    write_run_summary,
)

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
    max_trials: int | None = typer.Option(
        None, help="Uniform max_trials override for every layer."
    ),
    layer_trials: list[str] = typer.Option(  # noqa: B008
        [],
        "--layer-trials",
        help=(
            "Per-layer override as LAYER=N (repeatable); e.g. "
            "--layer-trials l1_engine=3 --layer-trials l2_topology=1. "
            "Takes precedence over --max-trials for the named layer."
        ),
    ),
) -> None:
    """Execute a run end-to-end."""
    cfg = _load_checked(config_path)
    per_layer = _parse_layer_trials(layer_trials)
    runner, ledger = build_runner(
        cfg, max_trials_override=max_trials, per_layer_overrides=per_layer,
    )
    start = time.monotonic()
    front = runner.run()
    elapsed = time.monotonic() - start

    # Write article-grade summaries alongside the per-trial JSONs.
    ledger_dir = Path(cfg.harness.ledger.output_dir)
    entries = list(ledger.entries())
    try:
        write_results_tsv(ledger_dir / "results.tsv", entries)
        hw = capture_hw_context()
        summary = build_run_summary(
            run_id=getattr(runner.events, "run_id", "unknown") if runner.events else "unknown",
            entries=entries,
            pareto=front,
            hw_context=hw,
            elapsed_s=elapsed,
        )
        write_run_summary(ledger_dir / "run_summary.json", summary)
    except Exception as e:  # noqa: BLE001
        typer.secho(f"[cli] telemetry write failed: {e}", fg=typer.colors.YELLOW, err=True)

    typer.echo(f"finished: {len(entries)} trials, {len(front)} pareto entries, {elapsed:.0f}s")
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


_VALID_LAYERS = {"l1_engine", "l2_topology", "l3_kernel"}


def _parse_layer_trials(entries: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for raw in entries:
        if "=" not in raw:
            raise typer.BadParameter(f"--layer-trials expects LAYER=N, got {raw!r}")
        name, _, value = raw.partition("=")
        name = name.strip()
        if name not in _VALID_LAYERS:
            raise typer.BadParameter(
                f"--layer-trials unknown layer {name!r}; expected one of {sorted(_VALID_LAYERS)}"
            )
        try:
            n = int(value)
        except ValueError as e:
            raise typer.BadParameter(f"--layer-trials value for {name!r} must be int, got {value!r}") from e
        if n < 1:
            raise typer.BadParameter(f"--layer-trials value for {name!r} must be >= 1")
        out[name] = n
    return out


if __name__ == "__main__":
    app()
