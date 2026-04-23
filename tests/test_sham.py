from __future__ import annotations

from typing import Any

import pytest

from autoinfer.harness.failure import FailureKind, FailureRecord
from autoinfer.harness.ledger import Measurement
from autoinfer.layers import LayerAdapter, TrialInput
from autoinfer.layers.sham import ShamAdapter


def _surface() -> dict[str, dict[str, Any]]:
    return {"x": {"type": "float", "low": 0.0, "high": 10.0}}


def _score(cfg: dict[str, Any]) -> Measurement:
    x = float(cfg["x"])
    return Measurement(
        tokens_per_sec=1000.0 - (x - 5.0) ** 2,
        ttft_p99_ms=50.0,
        tpot_p99_ms=25.0,
        peak_hbm_gb=40.0,
        kl_divergence=0.0,
    )


def test_sham_conforms_to_protocol() -> None:
    adapter = ShamAdapter(
        layer_name="l1_engine",
        search_surface=_surface(),
        score_fn=_score,
    )
    assert isinstance(adapter, LayerAdapter)


def test_sham_run_returns_measurement() -> None:
    adapter = ShamAdapter(
        layer_name="l1_engine",
        search_surface=_surface(),
        score_fn=_score,
    )
    out = adapter.run(TrialInput(trial_id="t1", config={"x": 5.0}))
    assert out.failure is None
    assert out.measurement is not None
    assert out.measurement.tokens_per_sec == 1000.0


def test_sham_failure_fn_triggers_failure() -> None:
    def fail(cfg: dict[str, Any]) -> FailureRecord | None:
        if cfg["x"] > 9.0:
            return FailureRecord(FailureKind.OOM, "too high", "t", "l1_engine")
        return None

    adapter = ShamAdapter(
        layer_name="l1_engine",
        search_surface=_surface(),
        score_fn=_score,
        failure_fn=fail,
    )
    bad = adapter.run(TrialInput(trial_id="t_bad", config={"x": 9.5}))
    assert bad.measurement is None
    assert bad.failure is not None
    assert bad.failure.kind is FailureKind.OOM

    ok = adapter.run(TrialInput(trial_id="t_ok", config={"x": 2.0}))
    assert ok.measurement is not None


def test_sham_surface_is_defensive_copy() -> None:
    s = _surface()
    adapter = ShamAdapter(
        layer_name="l1_engine",
        search_surface=s,
        score_fn=_score,
    )
    surface_out = adapter.surface()
    surface_out["injected"] = {"type": "int", "low": 0, "high": 1}
    assert "injected" not in adapter.surface()


def test_sham_teardown_is_noop() -> None:
    adapter = ShamAdapter(
        layer_name="l1_engine",
        search_surface=_surface(),
        score_fn=_score,
    )
    adapter.teardown()
    out = adapter.run(TrialInput(trial_id="t", config={"x": 1.0}))
    assert out.measurement is not None


def test_sham_raises_propagate() -> None:
    def bad_score(cfg: dict[str, Any]) -> Measurement:
        raise RuntimeError("oops")

    adapter = ShamAdapter(
        layer_name="l1_engine",
        search_surface=_surface(),
        score_fn=bad_score,
    )
    with pytest.raises(RuntimeError):
        adapter.run(TrialInput(trial_id="t", config={"x": 1.0}))
