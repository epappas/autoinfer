"""Shared substrate (P2).

All layer adapters and policies depend on these types. This package owns
the contracts; implementations live in ``harness/{driver, gate, replica,
ledger, failure}.py``.
"""

from autoinfer.harness.driver import (
    DriverResult,
    build_bench_command,
    parse_bench_output,
    run_driver,
)
from autoinfer.harness.failure import FailureKind, FailureRecord, classify_stderr
from autoinfer.harness.gate import (
    GateResult,
    batch_invariance_check,
    fetch_completion,
    fetch_logprobs,
    run_gate,
    topk_kl_divergence,
)
from autoinfer.harness.ledger import Entry, Ledger, Measurement
from autoinfer.harness.replica import ReferenceReplica

__all__ = [
    "DriverResult",
    "Entry",
    "FailureKind",
    "FailureRecord",
    "GateResult",
    "Ledger",
    "Measurement",
    "ReferenceReplica",
    "batch_invariance_check",
    "build_bench_command",
    "classify_stderr",
    "fetch_completion",
    "fetch_logprobs",
    "parse_bench_output",
    "run_driver",
    "run_gate",
    "topk_kl_divergence",
]
