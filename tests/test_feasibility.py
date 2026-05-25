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
from typing import Any

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


def test_config_distance_with_knob_weights_matches_weighted_average() -> None:
    """T-26b: per-knob weights produce a weighted average, not a uniform mean."""
    a = {"x": "fp8", "y": 0.85, "z": 1024}
    b = {"x": "auto", "y": 0.85, "z": 1024}
    # x mismatches (distance 1.0); y and z exact match (distance 0.0).
    # Plain average = 1/3 ≈ 0.333. Weighted with x=10x: 10/(10+1+1) ≈ 0.833.
    plain = _config_distance(a, b)
    weighted = _config_distance(a, b, knob_weights={"x": 10.0})
    assert plain == pytest.approx(1/3, abs=1e-3)
    assert weighted == pytest.approx(10/12, abs=1e-3)
    assert weighted > plain


def test_config_distance_zero_weight_floor_is_safe() -> None:
    """Defensive: empty config + empty weights returns 0, not divide-by-zero."""
    assert _config_distance({}, {}, knob_weights={"x": 5.0}) == 0.0


def test_config_distance_missing_keys_inherit_weight() -> None:
    """A knob present in only one config still contributes weight 1 (or
    its declared weight) to keep the missing-side penalty meaningful."""
    a = {"x": "auto", "y": 1}
    b = {"x": "auto"}  # y missing on b
    # plain: x dist 0, y dist 1 (missing) → avg = 0.5
    # weighted (y=10): x dist 0 with w=1, y dist 1 with w=10 → 10/11
    plain = _config_distance(a, b)
    weighted = _config_distance(a, b, knob_weights={"y": 10.0})
    assert plain == pytest.approx(0.5)
    assert weighted == pytest.approx(10/11, abs=1e-3)


def test_predict_proba_weights_recover_class_signal_in_high_dim_config() -> None:
    """T-26b core counterfactual — campaign 02 shape.

    Reproduces the exact problem: a 12-knob config space where 2 fp8
    STARTUP failures live in one neighbourhood (different other-knob
    values from the query) and 2 ``auto`` SUCCESSES live near the query
    on the other 11 knobs — matching what really happened in campaign
    02 (warmstart configs auto-KEPT cluster on common values; fp8
    surrogate fails come later in different regions).

    Without weights, T-26's class collapse is diluted: the 2 auto
    successes are ~the closest neighbours on 11 knobs, the fp8 fails
    are far on those knobs. The 3-NN vote returns near-100% success →
    classifier ACCEPTS the fp8 candidate (above
    ``feasibility_threshold=0.4``), exactly mirroring the campaign-02
    bug.

    With knob_weights upweighting kv_cache_dtype 10x, the kvc signal
    dominates: the fp8 fails are at distance ~0 on the 10x-weighted
    knob, the auto successes are at distance 1.0 on it. Top-3 flips
    to mostly fp8 fails → classifier REJECTS.
    """
    classes = {
        "kv_cache_dtype": {
            "fp8": "kv_fp8",
            "fp8_e4m3": "kv_fp8",
            "fp8_e5m2": "kv_fp8",
        },
    }
    weights = {"kv_cache_dtype": 10.0}
    threshold = 0.4  # matches example/qwen3-8b-l1-l2-l3-joint/config.yaml

    def cfg(kvc: str, **rest: Any) -> dict[str, Any]:
        return {"kv_cache_dtype": kvc, **rest}

    # Auto SUCCESSES sit ON the query's other-knob values (warmstart cluster).
    auto_other = {"k1": 4096, "k2": 0.88, "k3": "Z", "k4": True,
                   "k5": 16, "k6": 256, "k7": 0.85, "k8": "C",
                   "k9": 2048, "k10": 1024, "k11": "t"}
    history = [
        # Two fp8 STARTUP failures with other knobs FAR from the query.
        (cfg("fp8_e4m3", k1=1024, k2=0.85, k3="X", k4=True,  k5=16, k6=128,
                          k7=0.9,  k8="A", k9=4096, k10=2048, k11="r"),
         False, FailureKind.STARTUP),
        (cfg("fp8_e5m2", k1=2048, k2=0.92, k3="Y", k4=False, k5=32, k6=64,
                          k7=0.8,  k8="B", k9=8192, k10=4096, k11="s"),
         False, FailureKind.STARTUP),
        # Two auto SUCCESSES sitting in the warmstart-cluster region.
        (cfg("auto", **auto_other), True, None),
        (cfg("auto", **auto_other), True, None),
    ]
    # Query: an unseen fp8 variant whose other knobs match the
    # warmstart-cluster — this is what the surrogate actually proposes
    # mid-run (TPE samples near successful regions).
    query = cfg("fp8", **auto_other)

    m_unweighted = FeasibilityModel(
        k=3, min_observations=2, knob_classes=classes,
    )
    for c, ok, fk in history:
        m_unweighted.record(c, success=ok, failure_kind=fk)
    p_unw = m_unweighted.predict_proba(query)

    m_weighted = FeasibilityModel(
        k=3, min_observations=2, knob_classes=classes, knob_weights=weights,
    )
    for c, ok, fk in history:
        m_weighted.record(c, success=ok, failure_kind=fk)
    p_w = m_weighted.predict_proba(query)

    # The campaign-02 bug: unweighted ACCEPTS the fp8 candidate
    # (P(success) above threshold) because the auto SUCCESSES at the
    # same other-knob region dominate the 3-NN vote.
    assert p_unw >= threshold, (
        f"unweighted classifier should ACCEPT the fp8 candidate at "
        f"P(success) >= 0.4 — this is the campaign-02 bug being "
        f"reproduced. Got P={p_unw:.3f}; if this drops below 0.4, the "
        f"counterfactual no longer demonstrates the T-26b problem."
    )
    # T-26b fix: weighted REJECTS — kv_cache_dtype's 10x weight makes
    # the fp8 cluster the closest neighbourhood on the structurally-
    # decisive axis.
    assert p_w < threshold, (
        f"weighted classifier should REJECT the fp8 candidate at "
        f"P(success) < 0.4. Got P={p_w:.3f}."
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


# ---------------------------------------------------------------------------
# T-26c — per-FailureKind sub-classifier tests.
# ---------------------------------------------------------------------------


def test_predict_proba_backward_compat_no_kind_weights() -> None:
    """Empty kind_weights reproduces T-26b: shared knob_weights, aggregate vote.

    Pinning bit-for-bit equivalence between an unset kind_weights and the
    pre-T-26c behaviour ensures T-26c is purely additive — existing
    deployments that never populate kind_weights see no behavioural drift.
    """
    weights = {"x": 5.0}
    classes = {"k": {"a": "A", "b": "A"}}

    def fill(model: FeasibilityModel) -> None:
        model.record(
            {"x": 1, "k": "a"},
            success=False,
            failure_kind=FailureKind.STARTUP,
        )
        model.record(
            {"x": 2, "k": "b"},
            success=False,
            failure_kind=FailureKind.OOM,
        )
        model.record({"x": 8, "k": "c"}, success=True)
        model.record({"x": 9, "k": "c"}, success=True)

    m_t26b = FeasibilityModel(
        k=3,
        min_observations=2,
        knob_classes=classes,
        knob_weights=weights,
    )
    fill(m_t26b)
    m_t26c_empty = FeasibilityModel(
        k=3,
        min_observations=2,
        knob_classes=classes,
        knob_weights=weights,
        kind_weights={},
    )
    fill(m_t26c_empty)
    query = {"x": 1, "k": "a"}
    assert m_t26c_empty.predict_proba(query) == m_t26b.predict_proba(query)
    assert m_t26c_empty.predict_kind_proba(query) == m_t26b.predict_kind_proba(query)


def test_predict_proba_per_kind_rejects_oom_region_with_memory_drivers() -> None:
    """T-26c core counterfactual.

    Memory-knob OOM region: 3 OOM failures with high
    gpu_memory_utilization / max_num_seqs / max_num_batched_tokens /
    block_size; 3 successes with low values. Other knobs (precision,
    attention) vary freely across observations.

    Query: matches OOM-region on memory knobs, matches SUCCESS on
    catalog-rule knobs (kv_cache_dtype / attention_backend / quantization
    / enable_chunked_prefill).

    Under T-26b shared weights (catalog-rule knobs upweighted 10x,
    memory knobs default weight 1), the upweighted catalog-rule knobs
    pull the candidate toward SUCCESS history — P(success) stays high
    and the OOM-region candidate is NOT rejected.

    Under T-26c with OOM driver weights (memory knobs upweighted 10x),
    the candidate is exact on the OOM-tuned distance to OOM-fail history
    and far from SUCCESS history on those same knobs. P(fail OOM) → ~1,
    so P(success) = 1 − max_K P(fail K) drops below 0.3.
    """
    shared_weights = {
        "kv_cache_dtype": 10.0,
        "attention_backend": 10.0,
        "quantization": 10.0,
        "enable_chunked_prefill": 10.0,
    }
    oom_weights = {
        "gpu_memory_utilization": 10.0,
        "max_num_seqs": 10.0,
        "max_num_batched_tokens": 10.0,
        "block_size": 10.0,
    }

    def cfg(
        *,
        gmu: float,
        seqs: int,
        batched: int,
        block: int,
        quant: str,
        attn: str,
        chunked: bool,
    ) -> dict[str, Any]:
        return {
            "gpu_memory_utilization": gmu,
            "max_num_seqs": seqs,
            "max_num_batched_tokens": batched,
            "block_size": block,
            "kv_cache_dtype": "auto",
            "quantization": quant,
            "attention_backend": attn,
            "enable_chunked_prefill": chunked,
        }

    oom_a = cfg(gmu=0.95, seqs=512, batched=8192, block=32,
                quant="awq", attn="FLASH_ATTN", chunked=True)
    oom_b = cfg(gmu=0.95, seqs=512, batched=8192, block=32,
                quant="gptq", attn="FLASHINFER", chunked=False)
    oom_c = cfg(gmu=0.95, seqs=512, batched=8192, block=32,
                quant="fp8", attn="TRITON_ATTN", chunked=True)
    ok_a = cfg(gmu=0.82, seqs=64, batched=2048, block=16,
               quant="none", attn="FLASH_ATTN", chunked=True)
    ok_b = cfg(gmu=0.82, seqs=64, batched=2048, block=16,
               quant="none", attn="FLASH_ATTN", chunked=True)
    ok_c = cfg(gmu=0.82, seqs=64, batched=2048, block=16,
               quant="none", attn="FLASH_ATTN", chunked=True)
    query = cfg(gmu=0.95, seqs=512, batched=8192, block=32,
                quant="none", attn="FLASH_ATTN", chunked=True)

    def fill(model: FeasibilityModel) -> None:
        for c in (oom_a, oom_b, oom_c):
            model.record(c, success=False, failure_kind=FailureKind.OOM)
        for c in (ok_a, ok_b, ok_c):
            model.record(c, success=True)

    m_t26b = FeasibilityModel(
        k=3, min_observations=2, knob_weights=shared_weights,
    )
    fill(m_t26b)
    p_t26b = m_t26b.predict_proba(query)
    m_t26c = FeasibilityModel(
        k=3,
        min_observations=2,
        knob_weights=shared_weights,
        kind_weights={FailureKind.OOM: oom_weights},
    )
    fill(m_t26c)
    p_t26c = m_t26c.predict_proba(query)

    assert p_t26b > 0.5, (
        f"T-26b shared weights should fail to reject the OOM-region "
        f"candidate (the bug T-26c targets). Got P={p_t26b:.3f}; if this "
        f"drops below 0.5, the counterfactual no longer demonstrates "
        f"T-26c's value."
    )
    assert p_t26c < 0.3, (
        f"T-26c OOM sub-classifier should reject the OOM-region "
        f"candidate at P(success) < 0.3. Got P={p_t26c:.3f}."
    )


def test_predict_proba_per_kind_does_not_overreject_unrelated_regions() -> None:
    """T-26c safety: a candidate far from every failure region survives.

    Same history as the OOM-region test; query matches SUCCESS on the
    OOM-driver knobs (memory) and ALSO on catalog-rule knobs. Under
    T-26c the OOM sub-classifier sees the candidate at distance ~1 to
    OOM-failures on every weighted knob → P(fail OOM) low; STARTUP /
    QUALITY_* sub-classifiers ditto. max_K stays low → P(success) > 0.7.
    """
    oom_weights = {
        "gpu_memory_utilization": 10.0,
        "max_num_seqs": 10.0,
        "max_num_batched_tokens": 10.0,
        "block_size": 10.0,
    }
    quality_kl_weights = {
        "quantization": 10.0,
        "dtype": 10.0,
        "kv_cache_dtype": 10.0,
    }
    m = FeasibilityModel(
        k=3,
        min_observations=2,
        kind_weights={
            FailureKind.OOM: oom_weights,
            FailureKind.QUALITY_KL: quality_kl_weights,
        },
    )
    for _ in range(3):
        m.record(
            {
                "gpu_memory_utilization": 0.95,
                "max_num_seqs": 512,
                "max_num_batched_tokens": 8192,
                "block_size": 32,
                "kv_cache_dtype": "auto",
                "quantization": "awq",
                "dtype": "bfloat16",
            },
            success=False,
            failure_kind=FailureKind.OOM,
        )
    for _ in range(3):
        m.record(
            {
                "gpu_memory_utilization": 0.82,
                "max_num_seqs": 64,
                "max_num_batched_tokens": 2048,
                "block_size": 16,
                "kv_cache_dtype": "auto",
                "quantization": "none",
                "dtype": "auto",
            },
            success=True,
        )
    query = {
        "gpu_memory_utilization": 0.82,
        "max_num_seqs": 64,
        "max_num_batched_tokens": 2048,
        "block_size": 16,
        "kv_cache_dtype": "auto",
        "quantization": "none",
        "dtype": "auto",
    }
    p = m.predict_proba(query)
    assert p > 0.7, (
        f"candidate in the centre of the success cluster must survive "
        f"per-kind sub-classifiers; got P(success)={p:.3f}"
    )


def test_kind_weights_overrides_shared_knob_weights_per_kind() -> None:
    """T-26c routing: per-kind weights take precedence over shared weights
    for that kind; missing kinds fall back to the shared weights.

    Two FailureKind columns: OOM (in kind_weights) and STARTUP (not in
    kind_weights — falls back to self.knob_weights). The query lies in
    the OOM region only when distance is computed under OOM-specific
    weights; the same query under the shared knob_weights places it
    closer to STARTUP-failed neighbours.

    Asserts _predict_proba_for_kind picks each kind's weight vector
    correctly.
    """
    shared = {"shared_knob": 10.0}
    oom_only = {"oom_knob": 10.0}
    m = FeasibilityModel(
        k=3,
        min_observations=2,
        knob_weights=shared,
        kind_weights={FailureKind.OOM: oom_only},
    )
    m.record(
        {"oom_knob": 1.0, "shared_knob": 0.0, "filler": 0.0},
        success=False,
        failure_kind=FailureKind.OOM,
    )
    m.record(
        {"oom_knob": 1.0, "shared_knob": 0.0, "filler": 0.0},
        success=False,
        failure_kind=FailureKind.OOM,
    )
    m.record(
        {"oom_knob": 0.0, "shared_knob": 1.0, "filler": 0.0},
        success=False,
        failure_kind=FailureKind.STARTUP,
    )
    m.record(
        {"oom_knob": 0.0, "shared_knob": 1.0, "filler": 0.0},
        success=False,
        failure_kind=FailureKind.STARTUP,
    )
    m.record({"oom_knob": 5.0, "shared_knob": 5.0, "filler": 5.0}, success=True)
    m.record({"oom_knob": 5.0, "shared_knob": 5.0, "filler": 5.0}, success=True)

    query = {"oom_knob": 1.0, "shared_knob": 1.0, "filler": 5.0}
    p_fail_oom = m._predict_proba_for_kind(query, FailureKind.OOM)
    p_fail_startup = m._predict_proba_for_kind(query, FailureKind.STARTUP)
    assert p_fail_oom > 0.7, (
        f"OOM sub-classifier should latch onto oom_knob=1.0 match; "
        f"got P(fail OOM)={p_fail_oom:.3f}"
    )
    assert p_fail_startup > 0.7, (
        f"STARTUP sub-classifier (fallback to shared weights, "
        f"shared_knob=10x) should latch onto shared_knob=1.0 match; "
        f"got P(fail STARTUP)={p_fail_startup:.3f}"
    )
    p = m.predict_proba(query)
    assert p < 0.3, (
        f"either OOM or STARTUP sub-classifier rejects → P(success) low; "
        f"got P={p:.3f}"
    )


def test_predict_kind_proba_uses_kind_weights() -> None:
    """T-26c: predict_kind_proba routes through per-kind weights when set."""
    oom_weights = {"oom_knob": 10.0}
    m_shared = FeasibilityModel(k=3, min_observations=2)
    m_perkind = FeasibilityModel(
        k=3,
        min_observations=2,
        kind_weights={FailureKind.OOM: oom_weights},
    )
    for model in (m_shared, m_perkind):
        model.record(
            {"oom_knob": 1.0, "filler": 0.0},
            success=False,
            failure_kind=FailureKind.OOM,
        )
        model.record(
            {"oom_knob": 1.0, "filler": 10.0},
            success=False,
            failure_kind=FailureKind.OOM,
        )
        model.record({"oom_knob": 5.0, "filler": 0.0}, success=True)
        model.record({"oom_knob": 5.0, "filler": 10.0}, success=True)

    query = {"oom_knob": 1.0, "filler": 5.0}
    pk_shared = m_shared.predict_kind_proba(query)
    pk_perkind = m_perkind.predict_kind_proba(query)
    # Per-kind weights sharpen OOM signal: candidate's oom_knob matches
    # both OOM failures, but under uniform weights `filler` dilutes the
    # match. Under the per-kind 10x oom_knob weight, the match dominates.
    assert pk_perkind[FailureKind.OOM] > pk_shared[FailureKind.OOM]
    assert pk_perkind[FailureKind.OOM] > 0.95


# ---------------------------------------------------------------------------
# T-26c — derive_kind_weights static-taxonomy tests.
# ---------------------------------------------------------------------------


def _toy_catalog():  # type: ignore[no-untyped-def]
    """Minimal KnobCatalog covering the static-taxonomy entries."""
    from autoinfer.layers.l1_engine.surface import (
        CompatRule,
        KnobCatalog,
        KnobSpec,
    )

    def kspec(name: str, type_: str = "categorical") -> KnobSpec:
        return KnobSpec(name=name, type=type_, default=None)

    knobs = {
        name: kspec(name)
        for name in (
            "kv_cache_dtype",
            "gpu_memory_utilization",
            "max_num_seqs",
            "max_num_batched_tokens",
            "block_size",
            "quantization",
            "dtype",
            "enable_chunked_prefill",
            "attention_backend",
        )
    }
    constraints = (
        CompatRule(
            rule="kv_fp8_requires_compatible_backend",
            description="",
            when_knob="kv_cache_dtype",
            when_values=("fp8", "fp8_e4m3", "fp8_e5m2"),
            requires_knob="attention_backend",
            requires_values=("FLASHINFER", "FLASH_ATTN"),
        ),
        CompatRule(
            rule="chunked_prefill_bound",
            description="",
            when_knob="enable_chunked_prefill",
            when_values=(False,),
            requires_knob="max_num_batched_tokens",
            requires_values=(32768,),
        ),
    )
    return KnobCatalog(knobs=knobs, constraints=constraints)


def test_derive_kind_weights_static_taxonomy() -> None:
    """OOM / QUALITY_KL / QUALITY_INVARIANCE weights come from the static
    driver taxonomy, restricted to catalog knobs."""
    from autoinfer.layers.l1_engine.surface import derive_kind_weights

    catalog = _toy_catalog()
    out = derive_kind_weights(catalog, high_weight=10.0)

    assert set(out[FailureKind.OOM].keys()) == {
        "kv_cache_dtype",
        "gpu_memory_utilization",
        "max_num_seqs",
        "max_num_batched_tokens",
        "block_size",
    }
    assert all(v == 10.0 for v in out[FailureKind.OOM].values())

    assert set(out[FailureKind.QUALITY_KL].keys()) == {
        "quantization",
        "dtype",
        "kv_cache_dtype",
    }
    assert set(out[FailureKind.QUALITY_INVARIANCE].keys()) == {
        "max_num_seqs",
        "block_size",
        "enable_chunked_prefill",
    }


def test_derive_kind_weights_startup_mining_from_compat_rules() -> None:
    """STARTUP drivers = union of when_knob / requires_knob across rules."""
    from autoinfer.layers.l1_engine.surface import derive_kind_weights

    catalog = _toy_catalog()
    out = derive_kind_weights(catalog, high_weight=10.0)
    assert set(out[FailureKind.STARTUP].keys()) == {
        "kv_cache_dtype",
        "attention_backend",
        "enable_chunked_prefill",
        "max_num_batched_tokens",
    }
    assert all(v == 10.0 for v in out[FailureKind.STARTUP].values())


def test_derive_kind_weights_omits_kinds_with_no_catalog_overlap() -> None:
    """A kind whose driver set doesn't intersect the catalog is omitted;
    the per-kind sub-classifier falls back to self.knob_weights for it."""
    from autoinfer.layers.l1_engine.surface import (
        KnobCatalog,
        KnobSpec,
        derive_kind_weights,
    )

    knobs = {"foo": KnobSpec(name="foo", type="categorical", default=None)}
    catalog = KnobCatalog(knobs=knobs, constraints=())
    out = derive_kind_weights(catalog)
    assert FailureKind.OOM not in out
    assert FailureKind.QUALITY_KL not in out
    assert FailureKind.STARTUP not in out
    assert out == {}


def test_derive_kind_weights_real_catalog_smoke() -> None:
    """Real L1 catalog produces a non-empty mapping for OOM/QUALITY_*/STARTUP."""
    from pathlib import Path

    from autoinfer.layers.l1_engine.surface import (
        derive_kind_weights,
        load_catalog,
    )

    repo_root = Path(__file__).resolve().parents[1]
    catalog = load_catalog(repo_root / "src/autoinfer/layers/l1_engine/knobs.yaml")
    out = derive_kind_weights(catalog)
    for kind in (
        FailureKind.OOM,
        FailureKind.QUALITY_KL,
        FailureKind.QUALITY_INVARIANCE,
        FailureKind.STARTUP,
    ):
        assert kind in out, f"{kind!r} missing from real-catalog kind_weights"
        assert out[kind], f"{kind!r} weights empty"
        for knob_name, weight in out[kind].items():
            assert knob_name in catalog.knobs, (
                f"derived weight references unknown knob {knob_name!r}"
            )
            assert weight == 10.0
