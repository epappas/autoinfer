"""Tests for FeasibilityModel — the constrained-BO classifier.

Covers:
- empty/sparse history fallback (no filter)
- per-knob mixed-type distance
- k-NN predict_proba on synthetic feasibility surfaces
- predict_kind_proba diagnostics
- record() input validation
"""

from __future__ import annotations

import math

import pytest

from autoinfer.harness.failure import FailureKind
from autoinfer.policy.feasibility import (
    FeasibilityModel,
    _config_distance,
    _knob_distance,
)


def test_knob_distance_identical_returns_zero() -> None:
    assert _knob_distance(42, 42) == 0.0
    assert _knob_distance("auto", "auto") == 0.0
    assert _knob_distance(True, True) == 0.0
    assert _knob_distance(None, None) == 0.0


def test_knob_distance_different_string_returns_one() -> None:
    assert _knob_distance("FLASH_ATTN", "FLASHINFER") == 1.0


def test_knob_distance_different_bool_returns_one() -> None:
    assert _knob_distance(True, False) == 1.0


def test_knob_distance_numeric_proportional() -> None:
    # |1 - 2| / max(1, 2) = 0.5
    assert _knob_distance(1, 2) == pytest.approx(0.5)
    # very small change is small distance
    assert _knob_distance(0.85, 0.86) < 0.05
    # large change clipped at 1.0
    assert _knob_distance(1.0, 1000.0) == pytest.approx(0.999, abs=1e-3)


def test_knob_distance_one_none_returns_one() -> None:
    assert _knob_distance(None, "auto") == 1.0
    assert _knob_distance(42, None) == 1.0


def test_knob_distance_bool_vs_int_treats_as_categorical() -> None:
    """Python: True == 1, but we want booleans treated structurally."""
    assert _knob_distance(True, 1) == 1.0  # different "kinds"


def test_config_distance_empty_returns_zero() -> None:
    assert _config_distance({}, {}) == 0.0


def test_config_distance_identical_returns_zero() -> None:
    a = {"x": 1, "mode": "fast"}
    assert _config_distance(a, a) == 0.0


def test_config_distance_missing_keys_penalised() -> None:
    """Missing-on-one-side counts as max distance for that knob."""
    a = {"x": 1}
    b = {"x": 1, "y": 2}
    # union = {x, y}; x distance 0, y distance 1 (only on b) → average 0.5
    assert _config_distance(a, b) == pytest.approx(0.5)


def test_config_distance_normalised_by_arity() -> None:
    """A 10-knob mismatch on one knob shouldn't dwarf a 2-knob full mismatch."""
    a = {"x": 1, "y": 1}
    b = {"x": 2, "y": 1}
    d_small = _config_distance(a, b)  # 1 knob differs out of 2
    a_big = {f"k{i}": i for i in range(10)}
    b_big = dict(a_big)
    b_big["k0"] = 99  # 1 knob differs out of 10
    d_big = _config_distance(a_big, b_big)
    assert d_big < d_small  # bigger config, same single mismatch → lower avg distance


def test_record_rejects_inconsistent_outcome() -> None:
    m = FeasibilityModel()
    with pytest.raises(ValueError):
        m.record({"x": 1}, success=True, failure_kind=FailureKind.OOM)
    with pytest.raises(ValueError):
        m.record({"x": 1}, success=False, failure_kind=None)


def test_predict_proba_returns_one_below_min_observations() -> None:
    """No filter signal until min_observations data points."""
    m = FeasibilityModel(k=3, min_observations=4)
    m.record({"x": 1}, success=False, failure_kind=FailureKind.OOM)
    m.record({"x": 2}, success=False, failure_kind=FailureKind.OOM)
    assert m.predict_proba({"x": 99}) == 1.0


def test_predict_proba_one_when_all_neighbors_succeeded() -> None:
    m = FeasibilityModel(k=3, min_observations=2)
    for x in (1, 2, 3, 4):
        m.record({"x": x}, success=True)
    assert m.predict_proba({"x": 2}) == 1.0


def test_predict_proba_zero_when_all_neighbors_failed() -> None:
    m = FeasibilityModel(k=3, min_observations=2)
    for x in (1, 2, 3, 4):
        m.record({"x": x}, success=False, failure_kind=FailureKind.STARTUP)
    assert m.predict_proba({"x": 2}) == 0.0


def test_predict_proba_learns_step_function() -> None:
    """Synthetic 1D surface: x<5 always fails, x>=5 always succeeds.
    Predicting at x=2 should be near-zero; at x=8 near-one."""
    m = FeasibilityModel(k=3, min_observations=2)
    for x in (1, 2, 3, 4):
        m.record({"x": x}, success=False, failure_kind=FailureKind.STARTUP)
    for x in (5, 6, 7, 8):
        m.record({"x": x}, success=True)
    # Near 2: nearest neighbors are all failures
    p_low = m.predict_proba({"x": 2})
    assert p_low < 0.3, f"expected near-zero in failed region, got {p_low}"
    # Near 8: nearest neighbors are all successes
    p_high = m.predict_proba({"x": 8})
    assert p_high > 0.7, f"expected near-one in success region, got {p_high}"


def test_predict_proba_inverse_distance_weighting() -> None:
    """An exact match (distance 0) dominates the vote."""
    m = FeasibilityModel(k=3, min_observations=2)
    # distant failures
    for x in (100, 200, 300):
        m.record({"x": x}, success=False, failure_kind=FailureKind.OOM)
    # exact-match success
    m.record({"x": 5}, success=True)
    # query at x=5 → exact-match success should outweigh the 3 distant failures
    p = m.predict_proba({"x": 5})
    assert p > 0.95


def test_predict_kind_proba_segments_by_failure() -> None:
    """Diagnostic: query reports OOM-likely vs STARTUP-likely regions."""
    m = FeasibilityModel(k=3, min_observations=2)
    for x in (1, 2, 3):
        m.record({"x": x}, success=False, failure_kind=FailureKind.OOM)
    for x in (10, 11, 12):
        m.record({"x": x}, success=False, failure_kind=FailureKind.STARTUP)
    near_oom = m.predict_kind_proba({"x": 2})
    near_startup = m.predict_kind_proba({"x": 11})
    assert near_oom[FailureKind.OOM] > near_oom[FailureKind.STARTUP]
    assert near_startup[FailureKind.STARTUP] > near_startup[FailureKind.OOM]


def test_predict_kind_proba_empty_below_min_observations() -> None:
    m = FeasibilityModel(min_observations=10)
    m.record({"x": 1}, success=False, failure_kind=FailureKind.OOM)
    assert m.predict_kind_proba({"x": 1}) == {}


def test_n_observations_and_successful() -> None:
    m = FeasibilityModel()
    assert m.n_observations() == 0
    m.record({"x": 1}, success=True)
    m.record({"x": 2}, success=False, failure_kind=FailureKind.STARTUP)
    m.record({"x": 3}, success=True)
    assert m.n_observations() == 3
    assert m.n_successful() == 2


def test_history_returns_observations() -> None:
    m = FeasibilityModel()
    m.record({"x": 1}, success=True)
    hist = list(m.history())
    assert len(hist) == 1
    assert hist[0].success is True
    assert hist[0].config == {"x": 1}


def test_predict_proba_handles_categorical_knobs() -> None:
    """Categoricals (kv_cache_dtype, attention_backend) are common in L1.

    With matched-region history (many examples, one query) the signal
    converges. The fp8 region is dense with failures; the auto region
    is dense with successes. We verify the model orders them correctly,
    not that it gives extreme probabilities — at k=3 with 6
    observations and uniform per-knob weighting the model honestly
    reflects mixed-region evidence.
    """
    m = FeasibilityModel(k=3, min_observations=2)
    # FP8 KV always fails on this hardware (the autoinfer pattern)
    for kv in ("fp8", "fp8_e4m3", "fp8_e5m2"):
        m.record(
            {"kv_cache_dtype": kv, "attention_backend": "FLASHINFER"},
            success=False,
            failure_kind=FailureKind.STARTUP,
        )
    # auto KV always succeeds — same backend axis to keep the comparison
    # focused on the kv knob signal
    for backend in ("FLASHINFER", "FLASH_ATTN", "TRITON_ATTN"):
        m.record(
            {"kv_cache_dtype": "auto", "attention_backend": backend},
            success=True,
        )
    # In-region query (exact-match on kv_cache_dtype side):
    p_fp8 = m.predict_proba(
        {"kv_cache_dtype": "fp8", "attention_backend": "FLASHINFER"}
    )
    p_auto = m.predict_proba(
        {"kv_cache_dtype": "auto", "attention_backend": "FLASHINFER"}
    )
    # Ordering is what matters: fp8 < 0.5 < auto
    assert p_fp8 < p_auto, f"expected fp8 < auto, got fp8={p_fp8} auto={p_auto}"
    assert p_fp8 < 0.4, f"in-region fp8 should be solidly low, got {p_fp8}"
    assert p_auto > 0.6, f"in-region auto should be solidly high, got {p_auto}"


def test_predict_proba_handles_mixed_type_configs() -> None:
    """Real configs mix int / float / categorical / bool."""
    m = FeasibilityModel(k=3, min_observations=2)
    for max_seqs in (32, 64, 128):
        m.record(
            {
                "max_num_seqs": max_seqs,
                "attention_backend": "FLASHINFER",
                "enable_prefix_caching": True,
                "gpu_memory_utilization": 0.85,
            },
            success=True,
        )
    p = m.predict_proba(
        {
            "max_num_seqs": 96,
            "attention_backend": "FLASHINFER",
            "enable_prefix_caching": True,
            "gpu_memory_utilization": 0.86,
        }
    )
    assert p > 0.8


def test_knob_distance_class_map_collapses_intra_class_to_zero() -> None:
    """T-26: fp8 variants in one class compare at distance 0 within that knob."""
    cm = {"fp8": "kv_fp8", "fp8_e4m3": "kv_fp8", "fp8_e5m2": "kv_fp8"}
    assert _knob_distance("fp8", "fp8_e4m3", class_map=cm) == 0.0
    assert _knob_distance("fp8_e4m3", "fp8_e5m2", class_map=cm) == 0.0


def test_knob_distance_class_map_unrelated_value_stays_at_one() -> None:
    """Values not in the class_map fall back to per-string Hamming."""
    cm = {"fp8": "kv_fp8", "fp8_e4m3": "kv_fp8"}
    assert _knob_distance("fp8", "auto", class_map=cm) == 1.0
    assert _knob_distance("auto", "auto", class_map=cm) == 0.0


def test_knob_distance_class_map_does_not_affect_non_strings() -> None:
    """Bools and numerics keep their natural distances even with a class_map present."""
    cm = {"fp8": "kv_fp8"}
    assert _knob_distance(True, False, class_map=cm) == 1.0
    assert _knob_distance(1, 2, class_map=cm) == pytest.approx(0.5)


def test_predict_proba_generalises_within_knob_class() -> None:
    """T-26 core test: 3 fp8_e4m3 STARTUP failures + class collapse cause
    fp8_e5m2 to predict near-zero P(success). Without the class map, the
    legacy classifier (campaign 01 evidence) couldn't extract this — it
    saw fp8_e4m3 and fp8_e5m2 as distance 1, so neighbors were dominated
    by configs that happened to share other knobs.
    """
    classes = {
        "kv_cache_dtype": {
            "fp8": "kv_fp8",
            "fp8_e4m3": "kv_fp8",
            "fp8_e5m2": "kv_fp8",
        },
    }
    m = FeasibilityModel(k=3, min_observations=2, knob_classes=classes)
    # 3 fp8_e4m3 failures (the only fp8 variant explored)
    for _ in range(3):
        m.record(
            {"kv_cache_dtype": "fp8_e4m3", "max_num_seqs": 128},
            success=False,
            failure_kind=FailureKind.STARTUP,
        )
    # 3 auto successes to fill out the history
    for seqs in (32, 64, 256):
        m.record(
            {"kv_cache_dtype": "auto", "max_num_seqs": seqs},
            success=True,
        )
    # Query a never-seen variant in the same class
    p = m.predict_proba({"kv_cache_dtype": "fp8_e5m2", "max_num_seqs": 128})
    assert p < 0.2, f"class generalisation failed: p={p}"


def test_predict_proba_without_class_map_does_not_generalise_across_variants() -> None:
    """Counterfactual: without ``knob_classes``, the same history does NOT
    cleanly predict failure for an unseen fp8 variant — k=3 with one
    distance-1 neighbor on every other config gives a mixed verdict.
    Pinning this preserves the campaign-01 evidence that motivated T-26.

    Realistic setup: each historical trial has different non-fp8 knob
    values too (the campaign-01 surrogate explored mixed regions), so
    legacy distance is not 0-pegged on irrelevant knobs.
    """
    fp8_history = [
        {"kv_cache_dtype": "fp8_e4m3", "max_num_seqs": 32, "gmu": 0.85},
        {"kv_cache_dtype": "fp8_e4m3", "max_num_seqs": 64, "gmu": 0.90},
        {"kv_cache_dtype": "fp8_e4m3", "max_num_seqs": 256, "gmu": 0.92},
    ]
    auto_history = [
        {"kv_cache_dtype": "auto", "max_num_seqs": 128, "gmu": 0.85},
        {"kv_cache_dtype": "auto", "max_num_seqs": 128, "gmu": 0.88},
        {"kv_cache_dtype": "auto", "max_num_seqs": 128, "gmu": 0.92},
    ]
    query = {"kv_cache_dtype": "fp8_e5m2", "max_num_seqs": 128, "gmu": 0.88}

    m = FeasibilityModel(k=3, min_observations=2)  # no knob_classes
    for c in fp8_history:
        m.record(c, success=False, failure_kind=FailureKind.STARTUP)
    for c in auto_history:
        m.record(c, success=True)
    p_legacy = m.predict_proba(query)

    classes = {
        "kv_cache_dtype": {
            "fp8": "kv_fp8",
            "fp8_e4m3": "kv_fp8",
            "fp8_e5m2": "kv_fp8",
        }
    }
    m_classed = FeasibilityModel(k=3, min_observations=2, knob_classes=classes)
    for c in fp8_history:
        m_classed.record(c, success=False, failure_kind=FailureKind.STARTUP)
    for c in auto_history:
        m_classed.record(c, success=True)
    p_classed = m_classed.predict_proba(query)

    assert p_classed < p_legacy, (
        f"class collapse should sharpen P(fail) prediction: "
        f"legacy={p_legacy} classed={p_classed}"
    )
    assert p_legacy > 0.3, (
        f"legacy classifier should be inconclusive (P(success) > 0.3) on a "
        f"never-seen fp8 variant — got {p_legacy}; if this drops, the "
        f"counterfactual is no longer demonstrating the T-26 problem"
    )
    assert p_classed < 0.2, (
        f"class-aware classifier should solidly predict failure: got {p_classed}"
    )


def test_distance_floor_prevents_division_blowup() -> None:
    """Exact-match neighbor mustn't produce infinite weight."""
    m = FeasibilityModel(k=2, min_observations=2, distance_floor=1e-6)
    m.record({"x": 5}, success=True)
    m.record({"x": 5}, success=False, failure_kind=FailureKind.OOM)
    # Both have distance 0 → both get weight 1/distance_floor → tied
    p = m.predict_proba({"x": 5})
    assert math.isfinite(p)
    # 1 of 2 succeeded; weighted vote gives 0.5
    assert p == pytest.approx(0.5)
