"""Tests for the self-KL calibration one-way valve in builder.

Calibration must only RAISE the gate ceiling (when reference is
unusually noisy), never tighten below the configured ``max_kl``.
Otherwise the gate rejects valid candidates whose KL falls in the
band between self_p95 and configured_max_kl — exactly what killed
the smoke v2's L2 H100 trial.
"""

from __future__ import annotations

from typing import Any

import pytest

from autoinfer.builder import _calibrate_max_kl
from autoinfer.config import RunConfig


def _cfg(max_kl: float, calibrate: bool, multiplier: float = 5.0) -> RunConfig:
    return RunConfig.model_validate(
        {
            "name": "calib-test",
            "harness": {
                "driver": {
                    "trace_path": "trace.jsonl",
                    "duration_s": 60,
                    "slo_ttft_p99_ms": 500.0,
                    "slo_tpot_p99_ms": 50.0,
                },
                "gate": {
                    "replica_uri": "http://localhost:8001",
                    "prompts_path": "prompts.jsonl",
                    "max_kl": max_kl,
                    "calibrate_self_kl": calibrate,
                    "calibration_multiplier": multiplier,
                },
                "ledger": {"output_dir": "./runs/test"},
            },
            "policy": {"warmstart": {"llm_model": "stub"}},
            "layers": {
                "l1_engine": {
                    "model": "Qwen/Qwen3-8B",
                    "knobs_path": "knobs.yaml",
                }
            },
        }
    )


def test_calibration_disabled_returns_configured_max_kl() -> None:
    cfg = _cfg(max_kl=2.0, calibrate=False)
    assert _calibrate_max_kl(cfg, "model", ["p1"], label="t") == 2.0


def test_calibration_below_configured_floors_at_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: smoke v2's L2 trial failed because self_p95 (0.08) *
    multiplier (5) = 0.40 was used as the ceiling and rejected a clean
    H100 candidate at kl=3.84. The configured max_kl=2.0 is the user's
    'this is definitely drift' floor — calibration must respect it."""
    cfg = _cfg(max_kl=2.0, calibrate=True, multiplier=5.0)

    def fake_calibrate(**kwargs: Any) -> dict[str, float]:
        # tiny self-noise: 0.08 raw_p95
        return {"n": 20.0, "mean": 0.05, "median": 0.04, "p95": 0.08, "raw_p95": 0.08, "max": 0.10}

    monkeypatch.setattr("autoinfer.harness.gate.calibrate_self_kl", fake_calibrate)
    out = _calibrate_max_kl(cfg, "model", ["p1"], label="t")
    # self_p95 * multiplier = 0.4 < configured 2.0 → take configured
    assert out == 2.0


def test_calibration_above_configured_raises_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    """When reference is genuinely noisy, calibration should loosen
    the gate beyond user intent (one-way valve)."""
    cfg = _cfg(max_kl=2.0, calibrate=True, multiplier=5.0)

    def fake_calibrate(**kwargs: Any) -> dict[str, float]:
        # noisy reference: p95 of 1.0 → 5.0 effective
        return {"n": 20.0, "mean": 0.5, "median": 0.4, "p95": 1.0, "raw_p95": 1.0, "max": 1.5}

    monkeypatch.setattr("autoinfer.harness.gate.calibrate_self_kl", fake_calibrate)
    out = _calibrate_max_kl(cfg, "model", ["p1"], label="t")
    # 5 * 1.0 = 5.0 > configured 2.0 → take calibrated
    assert out == 5.0


def test_calibration_failure_falls_back_to_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(max_kl=3.0, calibrate=True)

    def fake_calibrate(**kwargs: Any) -> dict[str, float]:
        raise ConnectionError("reference unreachable")

    monkeypatch.setattr("autoinfer.harness.gate.calibrate_self_kl", fake_calibrate)
    out = _calibrate_max_kl(cfg, "model", ["p1"], label="t")
    assert out == 3.0
