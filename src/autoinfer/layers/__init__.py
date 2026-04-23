"""Per-layer adapter protocol (P3).

Every layer (l1_engine, l2_topology, l3_kernel) implements the same
``LayerAdapter`` protocol. Adapters expose a search surface, execute
trials against the shared harness substrate, and return either a
``Measurement`` or a ``FailureRecord``. Adapters never touch harness
internals; they hand signals back through ``TrialOutput``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from autoinfer.harness.failure import FailureRecord
from autoinfer.harness.ledger import Measurement


@dataclass(frozen=True)
class TrialInput:
    trial_id: str
    config: dict[str, Any]


@dataclass(frozen=True)
class TrialOutput:
    measurement: Measurement | None
    failure: FailureRecord | None

    def __post_init__(self) -> None:
        if (self.measurement is None) == (self.failure is None):
            raise ValueError(
                "TrialOutput must carry exactly one of measurement or failure"
            )


@runtime_checkable
class LayerAdapter(Protocol):
    """Contract every layer implements.

    Implementations MUST not raise from ``run``; all exceptions must be
    converted to a ``FailureRecord`` so the surrogate policy sees the
    failure as a typed signal (P9).
    """

    layer_name: str

    def surface(self) -> dict[str, Any]:
        """Return the search-space schema (knob names, types, ranges, constraints)."""
        ...

    def run(self, trial: TrialInput) -> TrialOutput:
        """Execute one trial synchronously; never raise."""
        ...

    def teardown(self) -> None:
        """Release any resources held across trials."""
        ...


__all__ = ["LayerAdapter", "TrialInput", "TrialOutput"]
