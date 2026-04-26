from __future__ import annotations

from autoinfer.harness.failure import FailureKind, FailureRecord
from autoinfer.harness.ledger import Entry, Measurement
from autoinfer.telemetry.summary import (
    _pip_show,
    build_run_summary,
    capture_hw_context,
)


def test_pip_show_returns_version_for_installed_package() -> None:
    """importlib.metadata path: a known-installed package returns its version."""
    v = _pip_show("autoinfer")
    assert v is not None
    assert v[0].isdigit()


def test_pip_show_returns_none_for_unknown_package() -> None:
    assert _pip_show("definitely-not-a-real-package-xyz12345") is None


def test_capture_hw_context_includes_versions_and_keys() -> None:
    ctx = capture_hw_context()
    assert "python" in ctx
    assert "platform" in ctx
    assert ctx.get("autoinfer_version") is not None
    # torch may or may not be installed in the dev env; presence of the
    # key matters more than its value
    assert "torch_version" in ctx
    assert "vllm_version" in ctx


def _measure(tok: float, tpot: float = 50.0, hbm: float = 10.0) -> Measurement:
    return Measurement(
        tokens_per_sec=tok,
        ttft_p99_ms=200.0,
        tpot_p99_ms=tpot,
        peak_hbm_gb=hbm,
        kl_divergence=0.5,
    )


def _kept(tid: str, layer: str, tok: float) -> Entry:
    return Entry(
        trial_id=tid,
        layer=layer,
        config={"x": 1},
        measurement=_measure(tok),
        failure=None,
    )


def _failed(tid: str, layer: str, kind: FailureKind) -> Entry:
    return Entry(
        trial_id=tid,
        layer=layer,
        config={"x": 1},
        measurement=None,
        failure=FailureRecord(kind=kind, message="x", trial_id=tid, layer=layer),
    )


def test_run_summary_includes_per_layer_best_and_pareto_serialised() -> None:
    """T-22: run_summary.json carries per-layer best + serialised Pareto
    so downstream tools don't have to re-load every trial JSON."""
    entries = [
        _kept("l1_w0000", "l1_engine", 400.0),
        _kept("l1_w0001", "l1_engine", 700.0),
        _kept("l2_w0000", "l2_topology", 900.0),
        _kept("l3_w0000", "l3_kernel", 850.0),
        _failed("l1_s0002", "l1_engine", FailureKind.STARTUP),
    ]
    pareto = [entries[2]]  # L2 dominates here
    summary = build_run_summary(
        run_id="r1",
        entries=entries,
        pareto=pareto,
        hw_context={"python": "3.13"},
        elapsed_s=600.0,
    )
    # Per-layer best surfaces all three layers' winners
    assert "best_by_layer" in summary
    assert summary["best_by_layer"]["l1_engine"]["trial_id"] == "l1_w0001"
    assert summary["best_by_layer"]["l2_topology"]["trial_id"] == "l2_w0000"
    assert summary["best_by_layer"]["l3_kernel"]["trial_id"] == "l3_w0000"
    # Pareto frontier serialised
    assert "pareto_frontier" in summary
    assert len(summary["pareto_frontier"]) == 1
    assert summary["pareto_frontier"][0]["trial_id"] == "l2_w0000"
    # Per-layer counts
    assert summary["n_kept_by_layer"] == {
        "l1_engine": 2, "l2_topology": 1, "l3_kernel": 1,
    }
    assert summary["n_failed_by_layer"] == {"l1_engine": 1}


def test_run_summary_handles_empty_run() -> None:
    """No trials → empty per-layer best, no top, pareto_size=0."""
    summary = build_run_summary(
        run_id="empty",
        entries=[],
        pareto=[],
        hw_context={},
        elapsed_s=0.0,
    )
    assert summary["n_trials"] == 0
    assert summary["best_by_layer"] == {}
    assert summary["pareto_frontier"] == []
    assert summary["top_by_tokens_per_sec"] is None
