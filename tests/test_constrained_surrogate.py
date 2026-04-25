"""ConstrainedOptunaSurrogate end-to-end tests.

The motivating regression: 0/4 L1 reserve hit rate in the
2026-04-25 full L1xL2xL3 campaign because penalty-encoded TPE
couldn't extract the "fp8 KV is structurally infeasible on A100"
rule. The constrained surrogate should learn the boundary from typed
failures and avoid resampling that region.
"""

from __future__ import annotations

import pytest

from autoinfer.harness.failure import FailureKind
from autoinfer.harness.ledger import Measurement
from autoinfer.policy.feasibility import FeasibilityModel
from autoinfer.policy.surrogate import (
    ConstrainedOptunaSurrogate,
    OptunaSurrogate,
    Suggestion,
    Surrogate,
)


def _surface() -> dict[str, dict[str, object]]:
    return {
        "x": {"type": "float", "low": 0.0, "high": 10.0},
        "mode": {"type": "categorical", "values": ["a", "b", "c"]},
    }


def _meas(value: float) -> Measurement:
    return Measurement(
        tokens_per_sec=value,
        ttft_p99_ms=0.0,
        tpot_p99_ms=0.0,
        peak_hbm_gb=0.0,
        kl_divergence=0.0,
    )


def _build(
    threshold: float = 0.4, max_resamples: int = 8, min_obs: int = 2
) -> ConstrainedOptunaSurrogate:
    inner = OptunaSurrogate(
        kind="random",  # use random so we explore the surface fairly
        seed=0,
        surface=_surface(),
        objective_axis="tokens_per_sec",
        maximize=True,
    )
    feas = FeasibilityModel(k=3, min_observations=min_obs)
    return ConstrainedOptunaSurrogate(
        inner=inner, feasibility=feas, threshold=threshold, max_resamples=max_resamples
    )


def test_conforms_to_surrogate_protocol() -> None:
    s = _build()
    assert isinstance(s, Surrogate)


def test_suggest_returns_valid_config() -> None:
    s = _build()
    sugg = s.suggest()
    assert isinstance(sugg, Suggestion)
    cfg = sugg.config
    assert 0.0 <= cfg["x"] <= 10.0
    assert cfg["mode"] in {"a", "b", "c"}


def test_below_min_observations_no_filtering() -> None:
    """Without history, the model returns 1.0 — no resampling."""
    s = _build(threshold=0.99, min_obs=10)  # threshold high but min_obs higher than data
    sugg = s.suggest()
    # Should accept first sample (no data → predict 1.0)
    s.record(sugg.trial_id, _meas(100.0), None)
    assert s.feasibility.n_observations() == 1


def test_record_updates_both_models() -> None:
    s = _build(min_obs=2)
    sugg = s.suggest()
    s.record(sugg.trial_id, _meas(123.0), None)
    assert s.feasibility.n_observations() == 1
    # inner perf model also recorded — pending should be empty
    assert sugg.trial_id not in s.inner._pending  # noqa: SLF001


def test_record_failure_propagates_to_feasibility() -> None:
    s = _build(min_obs=2)
    sugg = s.suggest()
    s.record(sugg.trial_id, None, FailureKind.OOM)
    hist = list(s.feasibility.history())
    assert len(hist) == 1
    assert hist[0].success is False
    assert hist[0].failure_kind is FailureKind.OOM


def test_record_unknown_trial_raises() -> None:
    s = _build()
    with pytest.raises(KeyError):
        s.record("never-suggested", _meas(1.0), None)


def test_resampling_avoids_failed_region() -> None:
    """Synthetic: x<5 fails, x>=5 succeeds. After enough trials, the
    constrained surrogate should produce more in-region samples than
    plain Optuna would."""
    s = _build(threshold=0.5, max_resamples=8, min_obs=4)
    # seed history with enough data for the model to predict
    for x in (1.0, 2.0, 3.0, 4.0):
        sugg = s.suggest()
        # override the suggestion so we record the desired x
        s.feasibility.record({"x": x, "mode": "a"}, success=False, failure_kind=FailureKind.STARTUP)
        s.inner.prune(sugg.trial_id)
        s._suggested_configs.pop(sugg.trial_id, None)  # noqa: SLF001
    for x in (6.0, 7.0, 8.0, 9.0):
        sugg = s.suggest()
        s.feasibility.record({"x": x, "mode": "a"}, success=True)
        s.inner.prune(sugg.trial_id)
        s._suggested_configs.pop(sugg.trial_id, None)  # noqa: SLF001

    # Now the model should distinguish. Run 30 suggestions and check
    # that the majority land in x>=5.
    in_region_count = 0
    for _ in range(30):
        sugg = s.suggest()
        if sugg.config["x"] >= 5.0:
            in_region_count += 1
        # tell with arbitrary (positive) score to keep state clean
        s.record(sugg.trial_id, _meas(100.0), None)
    # without filtering ~50%, with filtering should be >70%
    assert in_region_count > 21, f"expected >21/30 in-region, got {in_region_count}"


def test_fallback_when_all_resamples_below_threshold() -> None:
    """If every resample is below threshold, return the highest one
    rather than deadlocking. (Otherwise the runner can't make progress
    on a fully infeasible-looking surface.)"""
    s = _build(threshold=1.01, max_resamples=3, min_obs=2)
    # Seed every region as failed → every prediction will be 0.0
    for x in (1.0, 2.0, 3.0, 5.0, 7.0, 9.0):
        s.feasibility.record({"x": x, "mode": "a"}, success=False, failure_kind=FailureKind.STARTUP)
    # threshold=1.01 is unreachable; suggest must still return
    sugg = s.suggest()
    assert isinstance(sugg, Suggestion)
    # Track properly even though we'll just record failure
    s.record(sugg.trial_id, None, FailureKind.STARTUP)


def test_pruning_does_not_pollute_perf_model() -> None:
    """Resamples that get rejected must be told as PRUNED, not as
    real penalty observations. Optuna's KDE excludes PRUNED."""
    import optuna

    # threshold=0.99 and most-of-space failed → many rejections per
    # suggest, guaranteeing PRUNED trials.
    s = _build(threshold=0.99, max_resamples=4, min_obs=2)
    # Seed: every x<8 fails. Only x=8.5,9,9.5 succeed (narrow region).
    for x in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0):
        s.feasibility.record({"x": x, "mode": "a"}, success=False, failure_kind=FailureKind.OOM)
    for x in (8.5, 9.0, 9.5):
        s.feasibility.record({"x": x, "mode": "a"}, success=True)
    for _ in range(5):
        sugg = s.suggest()
        s.record(sugg.trial_id, _meas(100.0), None)
    trials = s.inner._study.get_trials(deepcopy=False)  # noqa: SLF001
    pruned = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]
    completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    # With most-of-space failed, the random sampler should hit the
    # failed region and trigger pruning at least once across 5 suggests.
    assert len(pruned) > 0, f"expected pruned resamples, got 0 (completed={len(completed)})"
    assert len(completed) == 5  # exactly one tell(value) per suggest


def test_max_resamples_one_means_no_resampling() -> None:
    """Edge: max_resamples=1 disables the constraint; first suggestion wins."""
    s = _build(threshold=0.99, max_resamples=1, min_obs=2)
    # Seed all failures so threshold can never be met
    for x in (1.0, 5.0, 9.0):
        s.feasibility.record({"x": x, "mode": "a"}, success=False, failure_kind=FailureKind.STARTUP)
    sugg = s.suggest()
    s.record(sugg.trial_id, _meas(100.0), None)
    # Still produced a config and recorded successfully — no infinite loop


def test_prune_pass_through() -> None:
    s = _build()
    sugg = s.suggest()
    s.prune(sugg.trial_id)
    # subsequent record should fail (pending was cleared)
    with pytest.raises(KeyError):
        s.record(sugg.trial_id, _meas(1.0), None)
