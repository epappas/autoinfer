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


def test_self_kl_aggregate_well_behaved_distribution() -> None:
    """Median-cap doesn't change anything when the data is well-behaved."""
    from autoinfer.harness.gate import _aggregate_self_kl

    per = sorted([0.05] * 19 + [0.10])  # 20 samples, median 0.05, p95 0.10
    out = _aggregate_self_kl(per)
    assert out["n"] == 20.0
    assert out["median"] == 0.05
    # raw_p95 = per[19] = 0.10; cap = 5 * 0.05 = 0.25 → keep raw
    assert out["raw_p95"] == 0.10
    assert out["p95"] == 0.10


def test_self_kl_aggregate_caps_outlier_blowup() -> None:
    """One huge outlier must not blow the p95 above 5x median."""
    from autoinfer.harness.gate import _aggregate_self_kl

    per = sorted([0.5] * 19 + [95.0])  # one bad prompt at 95
    out = _aggregate_self_kl(per)
    assert out["raw_p95"] == 95.0
    # cap = 5 * 0.5 = 2.5
    assert out["p95"] == 2.5


def test_self_kl_aggregate_zero_median_keeps_raw() -> None:
    """If every well-behaved sample is exactly zero, fall through to raw p95."""
    from autoinfer.harness.gate import _aggregate_self_kl

    per = [0.0] * 19 + [3.0]
    out = _aggregate_self_kl(per)
    assert out["median"] == 0.0
    assert out["p95"] == 3.0


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
