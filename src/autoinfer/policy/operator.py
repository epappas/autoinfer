"""LLM proposal operator invoked on stall or at fixed cadence (P7)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from autoinfer.policy.warmstart import ProposalLLM


@dataclass
class Operator:
    llm: ProposalLLM
    cadence: int = 10

    def __post_init__(self) -> None:
        if self.cadence < 1:
            raise ValueError("cadence must be >= 1")

    def should_propose(self, trials_since_last: int, stalled: bool) -> bool:
        return stalled or trials_since_last >= self.cadence

    def propose(
        self,
        surface: dict[str, dict[str, Any]],
        n: int,
        prior_notes: str,
        history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return self.llm.propose_configs(surface, n, prior_notes, history)
