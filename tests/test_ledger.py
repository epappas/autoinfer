from __future__ import annotations

from pathlib import Path

import pytest

from autoinfer.harness.failure import FailureKind, FailureRecord
from autoinfer.harness.ledger import Entry, Ledger, Measurement


def _m(tok: float, tpot: float, hbm: float = 40.0, kl: float = 0.01) -> Measurement:
    return Measurement(
        tokens_per_sec=tok,
        ttft_p99_ms=50.0,
        tpot_p99_ms=tpot,
        peak_hbm_gb=hbm,
        kl_divergence=kl,
    )


def _entry(tid: str, layer: str, meas: Measurement | None, fail: FailureRecord | None = None) -> Entry:
    return Entry(trial_id=tid, layer=layer, config={}, measurement=meas, failure=fail)


def test_single_entry_is_pareto_front(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path, pareto_axes=("tokens_per_sec", "tpot_p99_ms"))
    ledger.record(_entry("t1", "l1_engine", _m(1000.0, 30.0)))
    front = ledger.pareto_front()
    assert [e.trial_id for e in front] == ["t1"]


def test_dominated_excluded(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path, pareto_axes=("tokens_per_sec", "tpot_p99_ms"))
    ledger.record(_entry("good", "l1_engine", _m(1500.0, 25.0)))
    ledger.record(_entry("bad", "l1_engine", _m(1000.0, 30.0)))
    assert [e.trial_id for e in ledger.pareto_front()] == ["good"]


def test_tradeoff_both_kept(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path, pareto_axes=("tokens_per_sec", "tpot_p99_ms"))
    ledger.record(_entry("fast_tok", "l1_engine", _m(1500.0, 40.0)))
    ledger.record(_entry("low_tpot", "l1_engine", _m(1000.0, 25.0)))
    assert sorted(e.trial_id for e in ledger.pareto_front()) == ["fast_tok", "low_tpot"]


def test_failed_trial_excluded_from_front(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path, pareto_axes=("tokens_per_sec",))
    fail = FailureRecord(FailureKind.OOM, "oom", "t_fail", "l1_engine")
    ledger.record(_entry("t_fail", "l1_engine", None, fail))
    ledger.record(_entry("t_ok", "l1_engine", _m(1000.0, 25.0)))
    assert [e.trial_id for e in ledger.pareto_front()] == ["t_ok"]


def test_stale_marks_only_layers_above_invalidator(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path, pareto_axes=("tokens_per_sec",))
    ledger.record(_entry("a_l1", "l1_engine", _m(1000.0, 25.0)))
    ledger.record(_entry("b_l2", "l2_topology", _m(1200.0, 30.0)))
    ledger.record(_entry("c_l3", "l3_kernel", _m(1500.0, 30.0)))

    n = ledger.mark_stale("l3_kernel")

    assert n == 2
    stale = {e.layer for e in ledger.entries() if e.stale}
    assert stale == {"l1_engine", "l2_topology"}
    assert [e.trial_id for e in ledger.pareto_front()] == ["c_l3"]


def test_stale_from_l2_only_invalidates_l1(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path, pareto_axes=("tokens_per_sec",))
    ledger.record(_entry("a_l1", "l1_engine", _m(1000.0, 25.0)))
    ledger.record(_entry("b_l2", "l2_topology", _m(1200.0, 30.0)))
    ledger.record(_entry("c_l3", "l3_kernel", _m(1500.0, 30.0)))

    n = ledger.mark_stale("l2_topology")

    assert n == 1
    stale = {e.layer for e in ledger.entries() if e.stale}
    assert stale == {"l1_engine"}


def test_stale_from_l1_invalidates_nothing(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path, pareto_axes=("tokens_per_sec",))
    ledger.record(_entry("a_l1", "l1_engine", _m(1000.0, 25.0)))
    n = ledger.mark_stale("l1_engine")
    assert n == 0


def test_unknown_layer_rejected(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path, pareto_axes=("tokens_per_sec",))
    with pytest.raises(ValueError):
        ledger.mark_stale("l42_warp_drive")


def test_pareto_eligible_default_true(tmp_path: Path) -> None:
    del tmp_path  # default-flag check is purely about Entry construction
    e = _entry("a", "l1_engine", _m(100.0, 50.0))
    assert e.pareto_eligible is True


def test_pareto_front_excludes_ineligible_entries(tmp_path: Path) -> None:
    """L3 ops/sec (34000) shouldn't trivially dominate L2 token throughput
    (900) on the joint Pareto. Marking L3 entries pareto_eligible=False
    keeps the joint frontier honest."""
    ledger = Ledger(tmp_path, pareto_axes=("tokens_per_sec",))
    ledger.record(_entry("l1", "l1_engine", _m(900.0, 50.0)))
    l3 = _entry("l3", "l3_kernel", _m(34000.0, 0.0))
    l3.pareto_eligible = False
    ledger.record(l3)
    front = ledger.pareto_front()
    ids = {e.trial_id for e in front}
    assert ids == {"l1"}  # L3 excluded from joint frontier


def test_pareto_front_by_layer_groups_by_layer(tmp_path: Path) -> None:
    """Per-layer frontier still surfaces the L3 best even though it's
    excluded from the joint frontier."""
    ledger = Ledger(tmp_path, pareto_axes=("tokens_per_sec",))
    l3a = _entry("l3a", "l3_kernel", _m(100.0, 0.0))
    l3a.pareto_eligible = False
    l3b = _entry("l3b", "l3_kernel", _m(200.0, 0.0))
    l3b.pareto_eligible = False
    ledger.record(_entry("l1", "l1_engine", _m(900.0, 50.0)))
    ledger.record(l3a)
    ledger.record(l3b)
    by_layer = ledger.pareto_front_by_layer()
    assert {e.trial_id for e in by_layer["l1_engine"]} == {"l1"}
    # L3 layer's best is l3b (200 > 100)
    assert {e.trial_id for e in by_layer["l3_kernel"]} == {"l3b"}


def test_pareto_eligible_persists_to_disk(tmp_path: Path) -> None:
    """Disk JSON must include the eligibility flag so analysis tools
    loading from disk see the same state as in-memory."""
    import json

    ledger = Ledger(tmp_path, pareto_axes=("tokens_per_sec",))
    l3 = _entry("l3a", "l3_kernel", _m(100.0, 0.0))
    l3.pareto_eligible = False
    ledger.record(l3)
    payload = json.loads((tmp_path / "l3a.json").read_text())
    assert payload["pareto_eligible"] is False


def test_mark_stale_re_persists_to_disk(tmp_path: Path) -> None:
    """Regression: in-memory stale flag must reach disk so analysis
    tools loading from JSON see the same state as ``Ledger.entries()``."""
    import json

    ledger = Ledger(tmp_path, pareto_axes=("tokens_per_sec",))
    ledger.record(_entry("a_l1", "l1_engine", _m(1000.0, 25.0)))
    ledger.record(_entry("b_l2", "l2_topology", _m(1200.0, 30.0)))

    # Before staleness: l1 entry is fresh on disk
    payload_before = json.loads((tmp_path / "a_l1.json").read_text())
    assert payload_before["stale"] is False

    ledger.mark_stale("l2_topology")

    # After: l1 entry must show stale=True on disk too
    payload_after = json.loads((tmp_path / "a_l1.json").read_text())
    assert payload_after["stale"] is True

    # l2 entry untouched (its layer is not above the invalidator)
    payload_l2 = json.loads((tmp_path / "b_l2.json").read_text())
    assert payload_l2["stale"] is False


def test_entries_persisted_to_disk(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path, pareto_axes=("tokens_per_sec",))
    ledger.record(_entry("t1", "l1_engine", _m(1000.0, 25.0)))
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    assert files[0].name == "t1.json"


def test_measurement_value_unknown_axis_raises() -> None:
    m = _m(1000.0, 25.0)
    with pytest.raises(KeyError):
        m.value("does_not_exist")


def test_measurement_extra_axis() -> None:
    m = Measurement(
        tokens_per_sec=1000.0,
        ttft_p99_ms=50.0,
        tpot_p99_ms=25.0,
        peak_hbm_gb=40.0,
        kl_divergence=0.01,
        extra={"tokens_per_dollar": 3.14},
    )
    assert m.value("tokens_per_dollar") == 3.14
