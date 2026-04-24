from __future__ import annotations

from pathlib import Path
from typing import Any

from autoinfer.controller.continuous import ContinuousRunner, StallTracker
from autoinfer.controller.stale import LayerScheduler, LayerSpec
from autoinfer.harness.failure import FailureKind, FailureRecord
from autoinfer.harness.ledger import Ledger, Measurement
from autoinfer.layers.sham import ShamAdapter
from autoinfer.policy.operator import Operator
from autoinfer.policy.surrogate import OptunaSurrogate
from autoinfer.policy.warmstart import DeterministicProposalLLM


def _surface() -> dict[str, dict[str, Any]]:
    return {"x": {"type": "float", "low": 0.0, "high": 10.0}}


def _quadratic(cfg: dict[str, Any]) -> Measurement:
    x = float(cfg["x"])
    return Measurement(
        tokens_per_sec=1000.0 - (x - 7.5) ** 2,
        ttft_p99_ms=50.0,
        tpot_p99_ms=25.0,
        peak_hbm_gb=40.0,
        kl_divergence=0.0,
    )


def _spec(
    layer: str,
    max_trials: int,
    warmstart_n: int = 3,
    warmstart_seed: list[dict[str, Any]] | None = None,
    score_fn: Any = _quadratic,
    failure_fn: Any = None,
) -> LayerSpec:
    return LayerSpec(
        adapter=ShamAdapter(
            layer_name=layer,
            search_surface=_surface(),
            score_fn=score_fn,
            failure_fn=failure_fn,
        ),
        surrogate=OptunaSurrogate(
            kind="tpe",
            seed=0,
            surface=_surface(),
            objective_axis="tokens_per_sec",
            maximize=True,
        ),
        warmstart=DeterministicProposalLLM(
            warmstart_seed or [{"x": 1.0}, {"x": 5.0}, {"x": 9.0}]
        ),
        max_trials=max_trials,
        warmstart_n=warmstart_n,
    )


def test_stall_tracker_counts_non_improvements() -> None:
    t = StallTracker(threshold=3)
    t.record("l1_engine", 10.0)
    t.record("l1_engine", 5.0)
    t.record("l1_engine", 5.0)
    t.record("l1_engine", 5.0)
    assert t.stalled("l1_engine") is True


def test_stall_tracker_resets_on_improvement() -> None:
    t = StallTracker(threshold=2)
    t.record("l1_engine", 10.0)
    t.record("l1_engine", 5.0)
    t.record("l1_engine", 5.0)
    assert t.stalled("l1_engine") is True
    t.record("l1_engine", 20.0)
    assert t.stalled("l1_engine") is False


def test_stall_tracker_failures_count_as_non_improvement() -> None:
    t = StallTracker(threshold=2)
    t.record("l1_engine", 10.0)
    t.record("l1_engine", None)
    t.record("l1_engine", None)
    assert t.stalled("l1_engine") is True


def test_runner_exhausts_budget(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path, pareto_axes=("tokens_per_sec",))
    spec = _spec("l1_engine", max_trials=6, warmstart_n=3)
    runner = ContinuousRunner(
        scheduler=LayerScheduler({"l1_engine": spec}),
        ledger=ledger,
        objective_axis="tokens_per_sec",
    )
    front = runner.run()
    assert len(ledger.entries()) == 6
    assert len(front) >= 1


def test_runner_records_warmstart_then_surrogate(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path, pareto_axes=("tokens_per_sec",))
    spec = _spec("l1_engine", max_trials=5, warmstart_n=2)
    runner = ContinuousRunner(
        scheduler=LayerScheduler({"l1_engine": spec}),
        ledger=ledger,
        objective_axis="tokens_per_sec",
    )
    runner.run()
    tids = [e.trial_id for e in ledger.entries()]
    w_count = sum(1 for t in tids if "_w" in t)
    s_count = sum(1 for t in tids if "_s" in t)
    assert w_count == 2
    assert s_count == 3


def test_runner_propagates_failures_to_ledger(tmp_path: Path) -> None:
    def failing(cfg: dict[str, Any]) -> FailureRecord | None:
        if float(cfg["x"]) > 8.0:
            return FailureRecord(FailureKind.OOM, "too big", "t", "l1_engine")
        return None

    ledger = Ledger(tmp_path, pareto_axes=("tokens_per_sec",))
    spec = _spec(
        "l1_engine",
        max_trials=3,
        warmstart_n=3,
        warmstart_seed=[{"x": 1.0}, {"x": 9.0}, {"x": 5.0}],
        failure_fn=failing,
    )
    runner = ContinuousRunner(
        scheduler=LayerScheduler({"l1_engine": spec}),
        ledger=ledger,
        objective_axis="tokens_per_sec",
    )
    runner.run()
    entries = ledger.entries()
    failure_entries = [e for e in entries if e.failure is not None]
    assert len(failure_entries) == 1
    assert failure_entries[0].failure is not None
    assert failure_entries[0].failure.kind is FailureKind.OOM


def test_runner_stop_callable_interrupts(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path, pareto_axes=("tokens_per_sec",))
    spec = _spec("l1_engine", max_trials=100, warmstart_n=2)
    runner = ContinuousRunner(
        scheduler=LayerScheduler({"l1_engine": spec}),
        ledger=ledger,
        objective_axis="tokens_per_sec",
        stop=lambda led: len(led.entries()) >= 4,
    )
    runner.run()
    assert len(ledger.entries()) == 4


def test_runner_operator_fires_at_cadence(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path, pareto_axes=("tokens_per_sec",))
    op_llm = DeterministicProposalLLM([{"x": 7.5}])
    op = Operator(llm=op_llm, cadence=3)
    spec = _spec("l1_engine", max_trials=8, warmstart_n=2)
    runner = ContinuousRunner(
        scheduler=LayerScheduler({"l1_engine": spec}),
        ledger=ledger,
        objective_axis="tokens_per_sec",
        operator=op,
    )
    runner.run()
    op_tids = [e for e in ledger.entries() if "_o" in e.trial_id]
    assert len(op_tids) >= 1


def test_runner_multi_layer_stale_invalidation_fires_automatically(tmp_path: Path) -> None:
    """Regression: the runner must call propagate_finding when a new
    Pareto entry lands, so cross-layer stale-signal propagation happens
    without any out-of-band calls (thesis P4).

    Setup uses two layers (L1 + L3) with sham adapters that produce
    identical measurements. Since equal measurements don't dominate
    each other, both layers contribute entries to the Pareto frontier.
    The L3 entries trigger propagate_finding("l3_kernel") which marks
    L1 entries stale — we observe that directly without calling
    propagate_finding ourselves.
    """
    ledger = Ledger(tmp_path, pareto_axes=("tokens_per_sec",))
    scheduler = LayerScheduler(
        {
            "l1_engine": _spec("l1_engine", max_trials=2, warmstart_n=2),
            "l3_kernel": _spec("l3_kernel", max_trials=2, warmstart_n=2),
        }
    )
    runner = ContinuousRunner(
        scheduler=scheduler,
        ledger=ledger,
        objective_axis="tokens_per_sec",
    )
    runner.run()
    assert len(ledger.entries()) == 4

    # runner should have auto-propagated; L1 entries must already be stale
    stale_layers = {e.layer for e in ledger.entries() if e.stale}
    assert "l1_engine" in stale_layers
    assert "l3_kernel" not in stale_layers

    # a manual re-invocation should find nothing new to stale
    n_extra = scheduler.propagate_finding("l3_kernel", ledger)
    assert n_extra == 0

    front_layers = {e.layer for e in ledger.pareto_front()}
    assert front_layers == {"l3_kernel"}


def test_runner_single_layer_does_not_propagate(tmp_path: Path) -> None:
    """Single-layer runs must never call propagate_finding — there's
    nothing to invalidate. Regression guard."""
    ledger = Ledger(tmp_path, pareto_axes=("tokens_per_sec",))
    spec = _spec("l1_engine", max_trials=3, warmstart_n=3)
    runner = ContinuousRunner(
        scheduler=LayerScheduler({"l1_engine": spec}),
        ledger=ledger,
        objective_axis="tokens_per_sec",
    )
    runner.run()
    # no entries should be stale — propagate_finding wasn't called
    assert all(not e.stale for e in ledger.entries())


def test_runner_pareto_front_non_empty_after_valid_run(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path, pareto_axes=("tokens_per_sec", "tpot_p99_ms"))
    spec = _spec("l1_engine", max_trials=8, warmstart_n=3)
    runner = ContinuousRunner(
        scheduler=LayerScheduler({"l1_engine": spec}),
        ledger=ledger,
        objective_axis="tokens_per_sec",
    )
    front = runner.run()
    assert len(front) >= 1
    for e in front:
        assert e.measurement is not None
        assert e.failure is None
        assert not e.stale
