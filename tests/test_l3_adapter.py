from __future__ import annotations

from pathlib import Path

from autoinfer.harness.failure import FailureKind
from autoinfer.layers import TrialInput
from autoinfer.layers.l3_kernel import (
    REFERENCE_SOURCES,
    L3KernelAdapter,
    load_catalog,
)

_REPO_CATALOG = Path(__file__).parent.parent / "src/autoinfer/layers/l3_kernel/knobs.yaml"


def _adapter() -> L3KernelAdapter:
    return L3KernelAdapter(
        catalog=load_catalog(_REPO_CATALOG),
        perf_repeats=2,
        warmup_runs=1,
    )


def _base_cfg(op: str) -> dict[str, object]:
    entry, src = REFERENCE_SOURCES[op]
    return {
        "target_op": op,
        "dtype": "float32",
        "shape_regime": "small",
        "source": src,
        "entry_fn": entry,
    }


def test_reference_kernel_passes_correctness_and_perf() -> None:
    adapter = _adapter()
    for op in ("rmsnorm", "silu_mul", "rope"):
        out = adapter.run(TrialInput(trial_id=f"t_{op}", config=_base_cfg(op)))
        assert out.measurement is not None, op
        assert out.failure is None, op
        assert out.measurement.tokens_per_sec > 0.0
        assert out.measurement.extra["max_abs_err"] < 1e-3


def test_wrong_candidate_fails_quality_gate() -> None:
    adapter = _adapter()
    cfg = _base_cfg("rmsnorm")
    # deliberately wrong kernel: returns zeros
    cfg["source"] = "def wrong(x, w, eps):\n    return torch.zeros_like(x)"
    cfg["entry_fn"] = "wrong"
    out = adapter.run(TrialInput(trial_id="bad", config=cfg))
    assert out.measurement is None
    assert out.failure is not None
    assert out.failure.kind is FailureKind.QUALITY_KL


def test_raising_candidate_reports_startup_failure() -> None:
    adapter = _adapter()
    cfg = _base_cfg("rmsnorm")
    cfg["source"] = "def boom(x, w, eps):\n    raise RuntimeError('nope')"
    cfg["entry_fn"] = "boom"
    out = adapter.run(TrialInput(trial_id="boom", config=cfg))
    assert out.measurement is None
    assert out.failure is not None
    assert out.failure.kind is FailureKind.STARTUP


def test_syntax_error_source_reports_startup() -> None:
    adapter = _adapter()
    cfg = _base_cfg("rmsnorm")
    cfg["source"] = "def bad(:"
    cfg["entry_fn"] = "bad"
    out = adapter.run(TrialInput(trial_id="syntax", config=cfg))
    assert out.failure is not None
    assert out.failure.kind is FailureKind.STARTUP


def test_missing_entry_fn_reports_startup() -> None:
    adapter = _adapter()
    cfg = _base_cfg("rmsnorm")
    cfg["source"] = "x = 1"
    cfg["entry_fn"] = "nope"
    out = adapter.run(TrialInput(trial_id="miss", config=cfg))
    assert out.failure is not None
    assert out.failure.kind is FailureKind.STARTUP


def test_missing_required_keys_reports_startup() -> None:
    adapter = _adapter()
    out = adapter.run(TrialInput(trial_id="x", config={"target_op": "rmsnorm"}))
    assert out.failure is not None
    assert out.failure.kind is FailureKind.STARTUP


def test_unknown_target_op_reports_startup() -> None:
    adapter = _adapter()
    cfg = {
        "target_op": "softmax",
        "dtype": "float32",
        "shape_regime": "small",
        "source": "def f(x): return x",
        "entry_fn": "f",
    }
    out = adapter.run(TrialInput(trial_id="unknown", config=cfg))
    assert out.failure is not None
    assert out.failure.kind is FailureKind.STARTUP


def test_surrogate_fallback_uses_reference_source() -> None:
    """Surrogate-only trials lack source/entry_fn; adapter must fall
    back to the reference source and still produce a measurement."""
    adapter = _adapter()
    cfg = {
        "target_op": "rmsnorm",
        "dtype": "float32",
        "shape_regime": "small",
    }
    out = adapter.run(TrialInput(trial_id="surrogate", config=cfg))
    assert out.failure is None
    assert out.measurement is not None


def test_surface_is_non_empty_and_categorical() -> None:
    adapter = _adapter()
    surface = adapter.surface()
    assert set(surface.keys()) == {"target_op", "dtype", "shape_regime"}
    for spec in surface.values():
        assert spec["type"] == "categorical"
