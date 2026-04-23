"""Deterministic dev adapter (real LayerAdapter; not a mock).

Maps a trial config to a ``Measurement`` (or ``FailureRecord``) via
user-supplied pure functions. Useful for validating the controller
and cross-layer scheduler on CPU before a GPU-bound adapter lands,
and for local reproducibility checks.

This is a legitimate alternate implementation of ``LayerAdapter``,
same status as ``DeterministicProposalLLM`` relative to ``ProposalLLM``
— it has real uses beyond testing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from autoinfer.harness.failure import FailureRecord
from autoinfer.harness.ledger import Measurement
from autoinfer.layers import TrialInput, TrialOutput

ScoreFn = Callable[[dict[str, Any]], Measurement]
FailureFn = Callable[[dict[str, Any]], FailureRecord | None]


@dataclass
class ShamAdapter:
    """LayerAdapter whose trial outcome is a pure function of the config."""

    layer_name: str
    search_surface: dict[str, dict[str, Any]]
    score_fn: ScoreFn
    failure_fn: FailureFn | None = None

    def surface(self) -> dict[str, Any]:
        return dict(self.search_surface)

    def run(self, trial: TrialInput) -> TrialOutput:
        if self.failure_fn is not None:
            fail = self.failure_fn(trial.config)
            if fail is not None:
                return TrialOutput(measurement=None, failure=fail)
        return TrialOutput(measurement=self.score_fn(trial.config), failure=None)

    def teardown(self) -> None:
        return None
