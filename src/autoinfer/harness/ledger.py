"""Keep-discard Pareto ledger.

The ledger is the only component that decides "is this trial kept?". It
holds ``(config, measurement | failure)`` entries and exposes the Pareto
frontier over a configured metric tuple.

Stale-signal invalidation (P4): when a finding at layer ``N`` publishes
a stale flag via :meth:`Ledger.mark_stale`, entries from layers *above*
``N`` (earlier/shallower in the stack) are flagged stale. Stale entries
are excluded from ``pareto_front`` until re-evaluated.

Axis convention: axes in ``pareto_axes`` are minimized by default.
Throughput-like metrics that should be maximized are listed in
``_MAXIMIZE_AXES`` and sign-flipped internally.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from autoinfer.harness.failure import FailureRecord

_MAXIMIZE_AXES: frozenset[str] = frozenset(
    {
        "tokens_per_sec",
        "goodput",
        "tokens_per_dollar",
    }
)

_LAYER_ORDER: dict[str, int] = {
    "l1_engine": 1,
    "l2_topology": 2,
    "l3_kernel": 3,
}


def _to_min(axis: str, value: float) -> float:
    """Flip sign for maximize axes so the ledger minimizes uniformly."""
    return -value if axis in _MAXIMIZE_AXES else value


@dataclass(frozen=True)
class Measurement:
    """Numeric result of a successful trial."""

    tokens_per_sec: float
    ttft_p99_ms: float
    tpot_p99_ms: float
    peak_hbm_gb: float
    kl_divergence: float
    extra: dict[str, float] = field(default_factory=dict)

    def value(self, axis: str) -> float:
        if hasattr(self, axis) and axis != "extra":
            return float(getattr(self, axis))
        if axis in self.extra:
            return float(self.extra[axis])
        raise KeyError(f"Unknown measurement axis: {axis}")


@dataclass
class Entry:
    trial_id: str
    layer: str
    config: dict[str, Any]
    measurement: Measurement | None
    failure: FailureRecord | None
    stale: bool = False

    @property
    def kept(self) -> bool:
        return self.failure is None and self.measurement is not None and not self.stale


class Ledger:
    """Trial ledger with Pareto frontier + stale-signal invalidation."""

    def __init__(self, output_dir: Path, pareto_axes: tuple[str, ...]) -> None:
        assert pareto_axes, "pareto_axes must be non-empty"
        self._dir = Path(output_dir)
        self._axes = pareto_axes
        self._entries: list[Entry] = []
        self._dir.mkdir(parents=True, exist_ok=True)

    def record(self, entry: Entry) -> None:
        self._entries.append(entry)
        self._persist(entry)

    def entries(self) -> tuple[Entry, ...]:
        return tuple(self._entries)

    def mark_stale(self, invalidating_layer: str) -> int:
        """Flag every non-stale entry at a layer *above* the invalidator.

        Returns the count of entries newly marked stale. Re-persists each
        flagged entry so the on-disk JSON stays in sync with the
        in-memory ledger (analysis tools that load from disk should see
        the same state as ``Ledger.entries()``).
        """
        if invalidating_layer not in _LAYER_ORDER:
            raise ValueError(f"unknown layer: {invalidating_layer}")
        i = _LAYER_ORDER[invalidating_layer]
        n = 0
        for e in self._entries:
            if e.stale:
                continue
            eo = _LAYER_ORDER.get(e.layer)
            if eo is None or eo >= i:
                continue
            e.stale = True
            self._persist(e)
            n += 1
        return n

    def pareto_front(self) -> list[Entry]:
        kept = [e for e in self._entries if e.kept]
        return [e for e in kept if not any(self._dominates(o, e) for o in kept if o is not e)]

    def _dominates(self, a: Entry, b: Entry) -> bool:
        assert a.measurement is not None and b.measurement is not None
        better_any = False
        for axis in self._axes:
            av = _to_min(axis, a.measurement.value(axis))
            bv = _to_min(axis, b.measurement.value(axis))
            if av > bv:
                return False
            if av < bv:
                better_any = True
        return better_any

    def _persist(self, entry: Entry) -> None:
        path = self._dir / f"{entry.trial_id}.json"
        payload = {
            "trial_id": entry.trial_id,
            "layer": entry.layer,
            "config": entry.config,
            "measurement": asdict(entry.measurement) if entry.measurement else None,
            "failure": entry.failure.to_dict() if entry.failure else None,
            "stale": entry.stale,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
