from __future__ import annotations

import pytest

from autoinfer.policy.fidelity import Rung, promote, successive_halving_rungs


def test_rungs_final_keeps_all() -> None:
    rungs = successive_halving_rungs((100, 500), eta=3)
    assert rungs[0] == Rung(100, pytest.approx(1.0 / 3))
    assert rungs[-1] == Rung(500, 1.0)


def test_three_rung_schedule() -> None:
    rungs = successive_halving_rungs((50, 150, 500), eta=3)
    assert [r.prompt_count for r in rungs] == [50, 150, 500]
    assert rungs[0].keep_fraction == pytest.approx(1.0 / 3)
    assert rungs[1].keep_fraction == pytest.approx(1.0 / 3)
    assert rungs[2].keep_fraction == 1.0


def test_rungs_must_increase() -> None:
    with pytest.raises(ValueError):
        successive_halving_rungs((100, 100), eta=3)
    with pytest.raises(ValueError):
        successive_halving_rungs((500, 100), eta=3)


def test_eta_must_be_at_least_two() -> None:
    with pytest.raises(ValueError):
        successive_halving_rungs((100,), eta=1)


def test_rungs_non_empty() -> None:
    with pytest.raises(ValueError):
        successive_halving_rungs((), eta=3)


def test_promote_maximize() -> None:
    scores = [("a", 1.0), ("b", 5.0), ("c", 3.0), ("d", 2.0)]
    assert promote(scores, keep_fraction=0.5, maximize=True) == ["b", "c"]


def test_promote_minimize() -> None:
    scores = [("a", 1.0), ("b", 5.0), ("c", 3.0), ("d", 2.0)]
    assert promote(scores, keep_fraction=0.5, maximize=False) == ["a", "d"]


def test_promote_keeps_at_least_one() -> None:
    scores = [("a", 1.0), ("b", 2.0)]
    assert promote(scores, keep_fraction=0.1, maximize=True) == ["b"]


def test_promote_keep_all() -> None:
    scores = [("a", 1.0), ("b", 2.0)]
    assert sorted(promote(scores, keep_fraction=1.0, maximize=True)) == ["a", "b"]


def test_promote_rejects_zero_fraction() -> None:
    with pytest.raises(ValueError):
        promote([("a", 1.0)], keep_fraction=0.0, maximize=True)


def test_promote_empty_returns_empty() -> None:
    assert promote([], keep_fraction=0.5, maximize=True) == []
