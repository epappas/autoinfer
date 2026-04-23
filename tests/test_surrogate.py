from __future__ import annotations

import pytest

from autoinfer.harness.failure import FailureKind
from autoinfer.harness.ledger import Measurement
from autoinfer.policy.surrogate import OptunaSurrogate, Suggestion, Surrogate


def _surface() -> dict[str, dict[str, object]]:
    return {
        "x": {"type": "float", "low": 0.0, "high": 10.0},
        "n": {"type": "int", "low": 1, "high": 8},
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


def test_optuna_surrogate_conforms_to_protocol() -> None:
    s = OptunaSurrogate(
        kind="tpe",
        seed=0,
        surface=_surface(),
        objective_axis="tokens_per_sec",
        maximize=True,
    )
    assert isinstance(s, Surrogate)


def test_unknown_sampler_rejected() -> None:
    with pytest.raises(ValueError):
        OptunaSurrogate(
            kind="nonsense",
            seed=0,
            surface=_surface(),
            objective_axis="tokens_per_sec",
            maximize=True,
        )


def test_suggest_respects_ranges_and_types() -> None:
    s = OptunaSurrogate(
        kind="random",
        seed=42,
        surface=_surface(),
        objective_axis="tokens_per_sec",
        maximize=True,
    )
    seen_modes: set[str] = set()
    for _ in range(25):
        sugg = s.suggest()
        assert isinstance(sugg, Suggestion)
        cfg = sugg.config
        assert 0.0 <= cfg["x"] <= 10.0
        assert 1 <= cfg["n"] <= 8
        assert cfg["mode"] in {"a", "b", "c"}
        seen_modes.add(cfg["mode"])
        s.record(sugg.trial_id, _meas(cfg["x"]), None)
    # random sampler over 25 trials should see > 1 mode
    assert len(seen_modes) >= 2


def test_record_failure_does_not_crash() -> None:
    s = OptunaSurrogate(
        kind="tpe",
        seed=0,
        surface=_surface(),
        objective_axis="tokens_per_sec",
        maximize=True,
    )
    sugg = s.suggest()
    s.record(sugg.trial_id, None, FailureKind.OOM)


def test_record_unknown_trial_raises() -> None:
    s = OptunaSurrogate(
        kind="tpe",
        seed=0,
        surface=_surface(),
        objective_axis="tokens_per_sec",
        maximize=True,
    )
    with pytest.raises(KeyError):
        s.record("never-suggested", _meas(1.0), None)


def test_record_missing_measurement_and_failure_raises() -> None:
    s = OptunaSurrogate(
        kind="tpe",
        seed=0,
        surface=_surface(),
        objective_axis="tokens_per_sec",
        maximize=True,
    )
    sugg = s.suggest()
    with pytest.raises(ValueError):
        s.record(sugg.trial_id, None, None)


def test_tpe_converges_toward_maximum() -> None:
    surface = {"x": {"type": "float", "low": 0.0, "high": 10.0}}
    s = OptunaSurrogate(
        kind="tpe",
        seed=0,
        surface=surface,
        objective_axis="tokens_per_sec",
        maximize=True,
    )
    early: list[float] = []
    late: list[float] = []
    for i in range(60):
        sugg = s.suggest()
        x = float(sugg.config["x"])
        score = -(x - 7.5) ** 2
        s.record(sugg.trial_id, _meas(score), None)
        (early if i < 20 else late).append(score)
    assert max(late) >= max(early)
