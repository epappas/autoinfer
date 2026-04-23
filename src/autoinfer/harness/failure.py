"""Typed failure outcomes (P9).

Failure is a first-class signal that the surrogate policy consumes; it is
never a zero-reward hole. Every adapter wraps its ``run`` in a try/except
that produces a ``FailureRecord`` rather than raising.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FailureKind(str, Enum):
    OOM = "oom"
    HANG = "hang"
    NCCL = "nccl"
    QUALITY_KL = "quality_kl"
    QUALITY_INVARIANCE = "quality_invariance"
    STARTUP = "startup"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FailureRecord:
    kind: FailureKind
    message: str
    trial_id: str
    layer: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "message": self.message,
            "trial_id": self.trial_id,
            "layer": self.layer,
            "metadata": dict(self.metadata),
        }


def classify_stderr(stderr: str) -> FailureKind:
    """Heuristic classification of adapter stderr.

    Order matters: NCCL before HANG because NCCL-hang logs mention both
    'nccl' and 'timeout' and we want the more specific label.
    """
    lower = stderr.lower()
    if "cuda out of memory" in lower or "torch.cuda.outofmemoryerror" in lower:
        return FailureKind.OOM
    if "nccl" in lower:
        return FailureKind.NCCL
    if "timeout" in lower or "deadline exceeded" in lower:
        return FailureKind.HANG
    if "importerror" in lower or "cannot load" in lower:
        return FailureKind.STARTUP
    return FailureKind.UNKNOWN
