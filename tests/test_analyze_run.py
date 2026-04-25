"""Smoke tests for scripts/analyze_run.py — joint-aware analyzer.

Exercises the pure helpers (classify, pareto_front, fmt_trial_row,
_layer_knobs) against synthetic trial data so post-run analysis stays
working as the layer set evolves.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "analyze_run", Path(__file__).parent.parent / "scripts" / "analyze_run.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _trial(
    tid: str,
    layer: str,
    tok: float = 100.0,
    tpot: float = 50.0,
    hbm: float = 10.0,
    *,
    stale: bool = False,
    failure: str | None = None,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "trial_id": tid,
        "layer": layer,
        "config": {"attention_backend": "FLASHINFER"} if layer == "l1_engine" else {"gpu_type": "H100"},
        "stale": stale,
        "failure": ({"kind": failure} if failure else None),
        "measurement": (
            None
            if failure
            else {
                "tokens_per_sec": tok,
                "tpot_p99_ms": tpot,
                "ttft_p99_ms": 200.0,
                "peak_hbm_gb": hbm,
                "kl_divergence": 0.5,
            }
        ),
    }
    return base


def test_classify_distinguishes_stale_kept_fail() -> None:
    mod = _load_module()
    assert mod.classify(_trial("a", "l1_engine")) == "kept"
    assert mod.classify(_trial("b", "l1_engine", stale=True)) == "stale"
    assert mod.classify(_trial("c", "l1_engine", failure="oom")) == "fail:oom"


def test_pareto_front_drops_dominated() -> None:
    mod = _load_module()
    a = _trial("a", "l1_engine", tok=100.0, tpot=50.0, hbm=10.0)
    b = _trial("b", "l2_topology", tok=200.0, tpot=20.0, hbm=5.0)  # dominates a
    front = mod.pareto_front([a, b])
    ids = {t["trial_id"] for t in front}
    assert ids == {"b"}


def test_pareto_front_keeps_incomparable() -> None:
    mod = _load_module()
    a = _trial("a", "l1_engine", tok=200.0, tpot=100.0, hbm=10.0)  # high tok/s, slow
    b = _trial("b", "l2_topology", tok=100.0, tpot=10.0, hbm=10.0)  # lower tok/s, fast
    front = mod.pareto_front([a, b])
    ids = {t["trial_id"] for t in front}
    assert ids == {"a", "b"}


def test_layer_knobs_known_layers() -> None:
    mod = _load_module()
    assert "attention_backend" in mod._layer_knobs("l1_engine")
    assert "gpu_type" in mod._layer_knobs("l2_topology")
    assert "target_op" in mod._layer_knobs("l3_kernel")
    assert mod._layer_knobs("unknown_layer") == ()


def test_fmt_trial_row_kept_layer_aware() -> None:
    mod = _load_module()
    t = _trial("l1_engine_w0001", "l1_engine", tok=500.0, tpot=20.0)
    row = mod.fmt_trial_row(t)
    assert "KEPT" in row
    assert "l1_engine_w0001" in row
    assert "attention_backend=FLASHINFER" in row
    assert "tok/s=  500.0" in row


def test_fmt_trial_row_stale_marked() -> None:
    mod = _load_module()
    t = _trial("l1_engine_w0000", "l1_engine", tok=500.0, stale=True)
    row = mod.fmt_trial_row(t)
    assert "STALE" in row
    assert "invalidated" in row


def test_fmt_trial_row_failure() -> None:
    mod = _load_module()
    t = _trial("l1_engine_w0002", "l1_engine", failure="startup")
    row = mod.fmt_trial_row(t)
    assert "FAIL" in row
    assert "startup" in row


def test_phase_of_extracts_warmstart_surrogate_operator() -> None:
    mod = _load_module()
    assert mod.phase_of("l1_engine_w0001") == "warmstart"
    assert mod.phase_of("l2_topology_s0007") == "surrogate"
    assert mod.phase_of("l3_kernel_o0003") == "operator"
    assert mod.phase_of("nonsense") == "?"
