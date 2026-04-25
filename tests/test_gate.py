from __future__ import annotations

import math

import pytest

from autoinfer.harness.gate import GateResult, topk_kl_divergence


def test_kl_zero_for_identical_distributions() -> None:
    lp = [{"a": math.log(0.5), "b": math.log(0.5)}]
    assert topk_kl_divergence(lp, lp) == pytest.approx(0.0, abs=1e-9)


def test_kl_zero_for_empty_sequences() -> None:
    assert topk_kl_divergence([], []) == 0.0


def test_kl_infinity_on_length_mismatch() -> None:
    ref = [{"a": math.log(1.0)}]
    cand: list[dict[str, float]] = []
    assert topk_kl_divergence(ref, cand) == float("inf")


def test_kl_positive_for_divergent_distributions() -> None:
    ref = [{"a": math.log(0.9), "b": math.log(0.1)}]
    cand = [{"a": math.log(0.1), "b": math.log(0.9)}]
    assert topk_kl_divergence(ref, cand) > 0.5


def test_kl_averages_across_positions() -> None:
    ref = [{"a": math.log(1.0)}, {"a": math.log(1.0)}]
    cand = [{"a": math.log(1.0)}, {"a": math.log(1.0)}]
    assert topk_kl_divergence(ref, cand) == pytest.approx(0.0, abs=1e-9)


def test_kl_floors_missing_candidate_token() -> None:
    ref = [{"zzz": math.log(1.0)}]
    cand = [{"aaa": math.log(1.0)}]
    kl = topk_kl_divergence(ref, cand)
    assert kl > 10.0  # large positive from log(1/1e-10) contribution


def test_js_zero_for_identical_distributions() -> None:
    from autoinfer.harness.gate import topk_js_divergence

    ref = [{"a": math.log(0.5), "b": math.log(0.5)}]
    cand = [{"a": math.log(0.5), "b": math.log(0.5)}]
    assert topk_js_divergence(ref, cand) < 1e-9


def test_js_bounded_for_completely_disjoint_topk() -> None:
    """Same case that blows up KL (missing token) — JS stays bounded by log(2)."""
    from autoinfer.harness.gate import topk_js_divergence

    ref = [{"zzz": math.log(1.0)}]
    cand = [{"aaa": math.log(1.0)}]
    js = topk_js_divergence(ref, cand)
    # log(2) is the upper bound for symmetric distributions
    assert 0.0 < js <= math.log(2) + 1e-9


def test_js_symmetric_under_swap() -> None:
    from autoinfer.harness.gate import topk_js_divergence

    ref = [{"a": math.log(0.7), "b": math.log(0.3)}]
    cand = [{"a": math.log(0.4), "b": math.log(0.6)}]
    forward = topk_js_divergence(ref, cand)
    backward = topk_js_divergence(cand, ref)
    assert abs(forward - backward) < 1e-9


def test_js_infinity_on_length_mismatch() -> None:
    from autoinfer.harness.gate import topk_js_divergence

    ref = [{"a": math.log(1.0)}, {"a": math.log(1.0)}]
    cand = [{"a": math.log(1.0)}]
    assert topk_js_divergence(ref, cand) == float("inf")


def test_js_zero_for_empty_sequences() -> None:
    from autoinfer.harness.gate import topk_js_divergence

    assert topk_js_divergence([], []) == 0.0


def test_self_kl_aggregate_well_behaved_distribution() -> None:
    """Cap doesn't change anything when the data is well-behaved and below noise floor."""
    from autoinfer.harness.gate import _aggregate_self_kl

    per = sorted([0.05] * 19 + [0.10])  # 20 samples, median 0.05, raw_p95 0.10
    out = _aggregate_self_kl(per)
    assert out["n"] == 20.0
    assert out["median"] == 0.05
    # raw_p95 = 0.10; cap = max(5*0.05, 1.0) = 1.0 → keep raw 0.10
    assert out["raw_p95"] == 0.10
    assert out["p95"] == 0.10


def test_self_kl_aggregate_tiny_median_uses_noise_floor() -> None:
    """When median is near-zero (clean reference), an outlier is still
    capped at the noise floor (1.0), not at 5*median which would be tiny."""
    from autoinfer.harness.gate import _aggregate_self_kl

    per = sorted([0.02] * 19 + [19.0])  # smoke run's actual shape
    out = _aggregate_self_kl(per)
    assert out["raw_p95"] == 19.0
    assert out["median"] == 0.02
    # cap = max(5*0.02, 1.0) = 1.0; p95 = min(19, 1.0) = 1.0
    assert out["p95"] == 1.0


def test_self_kl_aggregate_high_median_uses_5x_cap() -> None:
    """When median is well above the noise floor, 5*median takes over."""
    from autoinfer.harness.gate import _aggregate_self_kl

    per = sorted([0.5] * 19 + [50.0])
    out = _aggregate_self_kl(per)
    # cap = max(5*0.5, 1.0) = 2.5; p95 = min(50.0, 2.5) = 2.5
    assert out["p95"] == 2.5


def test_self_kl_aggregate_zero_median_uses_floor() -> None:
    """If median is exactly zero, cap falls through to the noise floor."""
    from autoinfer.harness.gate import _aggregate_self_kl

    per = [0.0] * 19 + [3.0]
    out = _aggregate_self_kl(per)
    assert out["median"] == 0.0
    assert out["p95"] == 1.0  # cap = max(0, 1.0)


def test_gate_result_passes_requires_both_conditions() -> None:
    r = GateResult(
        mean_kl=0.01,
        max_kl=0.02,
        per_prompt_kl=(0.01,),
        batch_invariant=True,
    )
    assert r.passes(0.05)
    assert not r.passes(0.005)

    bad_invariance = GateResult(
        mean_kl=0.01, max_kl=0.02, per_prompt_kl=(0.01,), batch_invariant=False
    )
    assert not bad_invariance.passes(0.05)
