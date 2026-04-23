"""LLM proposer protocol + deterministic alt-implementation.

``ProposalLLM`` is the contract every LLM proposer satisfies — Anthropic,
OpenAI, or any other. ``DeterministicProposalLLM`` is a real alternate
implementation (not a mock) useful for tests, reproducibility, warm-starting
from prior run artifacts, and local dev without API keys.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ProposalLLM(Protocol):
    def propose_configs(
        self,
        surface: dict[str, dict[str, Any]],
        n: int,
        prior_notes: str,
        history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]: ...


class DeterministicProposalLLM:
    """Cycles through a pre-registered list of configs in order.

    Validates each emitted config against ``surface`` (keys must match).
    Useful for deterministic tests, warm-starting iteration N+1 from
    iteration N's Pareto frontier, and local dev without cloud LLM
    credentials.
    """

    def __init__(self, configs: list[dict[str, Any]]) -> None:
        if not configs:
            raise ValueError("configs must be non-empty")
        self._configs: list[dict[str, Any]] = [dict(c) for c in configs]
        self._idx = 0

    def propose_configs(
        self,
        surface: dict[str, dict[str, Any]],
        n: int,
        prior_notes: str,
        history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if n <= 0:
            raise ValueError("n must be positive")
        out: list[dict[str, Any]] = []
        for _ in range(n):
            cfg = dict(self._configs[self._idx % len(self._configs)])
            self._validate(cfg, surface)
            out.append(cfg)
            self._idx += 1
        return out

    @staticmethod
    def _validate(cfg: dict[str, Any], surface: dict[str, dict[str, Any]]) -> None:
        for key in cfg:
            if key not in surface:
                raise ValueError(f"config key {key!r} not in surface {sorted(surface)!r}")
