"""Hyperband-style successive-halving scheduler (C7 evidence favors multi-fidelity).

Rungs are evaluated at increasing prompt counts; at each rung only the
top ``1/eta`` fraction advances. The final rung keeps everything that
reaches it. Pure logic; no dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rung:
    prompt_count: int
    keep_fraction: float


def successive_halving_rungs(rungs: tuple[int, ...], eta: int) -> list[Rung]:
    """Build rungs ``[(r, 1/eta)…, (last, 1.0)]``.

    - ``rungs``: prompt counts, strictly increasing.
    - ``eta``: halving ratio (>= 2).
    """
    if eta < 2:
        raise ValueError("eta must be >= 2")
    if not rungs:
        raise ValueError("rungs must be non-empty")
    for a, b in zip(rungs, rungs[1:], strict=False):
        if a >= b:
            raise ValueError(f"rungs must be strictly increasing, got {rungs}")
    last = len(rungs) - 1
    keep = 1.0 / eta
    return [
        Rung(prompt_count=p, keep_fraction=keep if i < last else 1.0)
        for i, p in enumerate(rungs)
    ]


def promote(scores: list[tuple[str, float]], keep_fraction: float, maximize: bool) -> list[str]:
    """Return the top ``keep_fraction`` trial_ids by score."""
    if not scores:
        return []
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction must be in (0, 1]")
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=maximize)
    n_keep = max(1, int(round(len(sorted_scores) * keep_fraction)))
    return [tid for tid, _ in sorted_scores[:n_keep]]
