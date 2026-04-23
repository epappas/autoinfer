from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from autoinfer.controller.stale import (
    LayerScheduler,
    LayerSpec,
    history_projection,
    is_above,
)
from autoinfer.harness.ledger import Entry, Ledger, Measurement
from autoinfer.layers.sham import ShamAdapter
from autoinfer.policy.surrogate import OptunaSurrogate
from autoinfer.policy.warmstart import DeterministicProposalLLM


def _surface() -> dict[str, dict[str, Any]]:
    return {"x": {"type": "float", "low": 0.0, "high": 10.0}}


def _score(cfg: dict[str, Any]) -> Measurement:
    return Measurement(
        tokens_per_sec=float(cfg["x"]),
        ttft_p99_ms=50.0,
        tpot_p99_ms=25.0,
        peak_hbm_gb=40.0,
        kl_divergence=0.0,
    )


def _build_spec(layer: str, max_trials: int = 5, warmstart_n: int = 2) -> LayerSpec:
    adapter = ShamAdapter(
        layer_name=layer,
        search_surface=_surface(),
        score_fn=_score,
    )
    surrogate = OptunaSurrogate(
        kind="random",
        seed=0,
        surface=_surface(),
        objective_axis="tokens_per_sec",
        maximize=True,
    )
    warmstart = DeterministicProposalLLM([{"x": 1.0}, {"x": 5.0}])
    return LayerSpec(
        adapter=adapter,
        surrogate=surrogate,
        warmstart=warmstart,
        max_trials=max_trials,
        warmstart_n=warmstart_n,
    )


def test_scheduler_requires_at_least_one_spec() -> None:
    with pytest.raises(ValueError):
        LayerScheduler({})


def test_single_layer_pick_until_budget_exhausted(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path, pareto_axes=("tokens_per_sec",))
    spec = _build_spec("l1_engine", max_trials=3)
    sch = LayerScheduler({"l1_engine": spec})
    assert sch.pick_layer(ledger) == "l1_engine"
    sch.notify_trial_done("l1_engine")
    sch.notify_trial_done("l1_engine")
    sch.notify_trial_done("l1_engine")
    assert sch.pick_layer(ledger) is None


def test_multi_layer_round_robin(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path, pareto_axes=("tokens_per_sec",))
    sch = LayerScheduler(
        {
            "l1_engine": _build_spec("l1_engine", max_trials=1),
            "l3_kernel": _build_spec("l3_kernel", max_trials=1),
        }
    )
    assert sch.pick_layer(ledger) == "l1_engine"
    sch.notify_trial_done("l1_engine")
    assert sch.pick_layer(ledger) == "l3_kernel"
    sch.notify_trial_done("l3_kernel")
    assert sch.pick_layer(ledger) is None


def test_stale_entries_take_priority(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path, pareto_axes=("tokens_per_sec",))
    ledger.record(Entry("e1", "l1_engine", {"x": 1.0}, _score({"x": 1.0}), None))
    ledger.record(Entry("e3", "l3_kernel", {"x": 2.0}, _score({"x": 2.0}), None))
    sch = LayerScheduler(
        {
            "l1_engine": _build_spec("l1_engine", max_trials=5),
            "l3_kernel": _build_spec("l3_kernel", max_trials=5),
        }
    )
    sch.notify_trial_done("l1_engine")
    sch.notify_trial_done("l3_kernel")
    n = sch.propagate_finding("l3_kernel", ledger)
    assert n == 1  # only l1_engine is above l3_kernel
    assert sch.pick_layer(ledger) == "l1_engine"


def test_propagate_finding_unknown_layer(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path, pareto_axes=("tokens_per_sec",))
    sch = LayerScheduler({"l1_engine": _build_spec("l1_engine")})
    with pytest.raises(ValueError):
        sch.propagate_finding("bogus", ledger)


def test_is_above_semantics() -> None:
    assert is_above("l1_engine", "l3_kernel") is True
    assert is_above("l2_topology", "l3_kernel") is True
    assert is_above("l1_engine", "l2_topology") is True
    assert is_above("l3_kernel", "l1_engine") is False
    assert is_above("l3_kernel", "l3_kernel") is False


def test_history_projection_excludes_stale_and_other_layers(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path, pareto_axes=("tokens_per_sec",))
    ledger.record(Entry("a", "l1_engine", {"x": 1.0}, _score({"x": 1.0}), None))
    ledger.record(
        Entry("b", "l1_engine", {"x": 2.0}, _score({"x": 2.0}), None, stale=True)
    )
    ledger.record(Entry("c", "l3_kernel", {"x": 3.0}, _score({"x": 3.0}), None))
    hist = history_projection(ledger, "l1_engine")
    ids = {row["trial_id"] for row in hist}
    assert ids == {"a"}


def test_warmstart_flag_transitions(tmp_path: Path) -> None:
    spec = _build_spec("l1_engine")
    sch = LayerScheduler({"l1_engine": spec})
    assert sch.is_warmstart_needed("l1_engine") is True
    sch.mark_warmstart_done("l1_engine")
    assert sch.is_warmstart_needed("l1_engine") is False


def test_notify_unknown_layer_raises() -> None:
    spec = _build_spec("l1_engine")
    sch = LayerScheduler({"l1_engine": spec})
    with pytest.raises(ValueError):
        sch.notify_trial_done("l99_fantasy")


def test_has_budget(tmp_path: Path) -> None:
    spec = _build_spec("l1_engine", max_trials=2)
    sch = LayerScheduler({"l1_engine": spec})
    assert sch.has_budget("l1_engine") is True
    sch.notify_trial_done("l1_engine")
    sch.notify_trial_done("l1_engine")
    assert sch.has_budget("l1_engine") is False
