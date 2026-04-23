"""Cross-layer scheduler (P4).

Owns the ``which layer runs next`` decision and the stale-signal flow.
A finding at layer N publishes stale flags at layers above N via
``Ledger.mark_stale``; the scheduler then prioritizes re-evaluation of
those layers before advancing.

Single-layer runs degenerate to ``always return that layer until
budget exhausted``; multi-layer runs interleave per registration order
with stale priority on top.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from autoinfer.harness.ledger import Entry, Ledger
from autoinfer.layers import LayerAdapter
from autoinfer.policy.surrogate import Surrogate
from autoinfer.policy.warmstart import ProposalLLM


@dataclass
class LayerSpec:
    """One layer's full search configuration."""

    adapter: LayerAdapter
    surrogate: Surrogate
    warmstart: ProposalLLM
    max_trials: int
    warmstart_n: int = 10
    warmstart_prior: str = ""


class LayerScheduler:
    """Per-layer budget tracking + stale-priority next-layer selection."""

    def __init__(self, specs: dict[str, LayerSpec]) -> None:
        if not specs:
            raise ValueError("at least one layer spec required")
        self._specs = dict(specs)
        self._done: dict[str, int] = dict.fromkeys(self._specs, 0)
        self._warmstart_done: dict[str, bool] = dict.fromkeys(self._specs, False)

    @property
    def specs(self) -> dict[str, LayerSpec]:
        return dict(self._specs)

    def is_warmstart_needed(self, layer: str) -> bool:
        return not self._warmstart_done[layer]

    def mark_warmstart_done(self, layer: str) -> None:
        self._warmstart_done[layer] = True

    def notify_trial_done(self, layer: str) -> None:
        if layer not in self._done:
            raise ValueError(f"unknown layer: {layer!r}")
        self._done[layer] += 1

    def done(self, layer: str) -> int:
        return self._done[layer]

    def has_budget(self, layer: str) -> bool:
        return self._done[layer] < self._specs[layer].max_trials

    def pick_layer(self, ledger: Ledger) -> str | None:
        """Return the next layer to run.

        Stale-priority first: if any layer has stale entries, pick it.
        Otherwise round-robin in registration order over layers with budget.
        """
        stale_by_layer = self._count_stale(ledger)
        for name in self._specs:
            if stale_by_layer.get(name, 0) > 0 and self.has_budget(name):
                return name
        for name in self._specs:
            if self.has_budget(name):
                return name
        return None

    def propagate_finding(self, finding_layer: str, ledger: Ledger) -> int:
        """On a new finding at ``finding_layer``, invalidate layers above.

        Delegates to ``Ledger.mark_stale``. Returns the number of entries
        newly flagged stale.
        """
        if finding_layer not in self._specs:
            raise ValueError(f"unknown layer: {finding_layer!r}")
        return ledger.mark_stale(finding_layer)

    @staticmethod
    def _count_stale(ledger: Ledger) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in ledger.entries():
            if e.stale:
                counts[e.layer] = counts.get(e.layer, 0) + 1
        return counts


def history_projection(ledger: Ledger, layer: str) -> list[dict[str, Any]]:
    """Compact history view for a single layer, suitable for LLM operator context."""
    out: list[dict[str, Any]] = []
    for e in ledger.entries():
        if e.layer != layer or e.stale:
            continue
        row: dict[str, Any] = {"trial_id": e.trial_id, "config": e.config}
        if e.measurement is not None:
            row["metrics"] = {
                "tokens_per_sec": e.measurement.tokens_per_sec,
                "tpot_p99_ms": e.measurement.tpot_p99_ms,
                "kl_divergence": e.measurement.kl_divergence,
            }
        if e.failure is not None:
            row["failure"] = e.failure.kind.value
        out.append(row)
    return out


def get_layer_order(layer: str) -> int:
    from autoinfer.harness.ledger import _LAYER_ORDER

    return _LAYER_ORDER[layer]


def is_above(entry_layer: str, invalidator_layer: str) -> bool:
    """True if ``entry_layer`` is strictly above ``invalidator_layer``."""
    from autoinfer.harness.ledger import _LAYER_ORDER

    return _LAYER_ORDER[entry_layer] < _LAYER_ORDER[invalidator_layer]


__all__ = [
    "Entry",
    "LayerScheduler",
    "LayerSpec",
    "get_layer_order",
    "history_projection",
    "is_above",
]
