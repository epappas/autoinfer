"""Rich telemetry for article-grade run analysis.

- events.EventLog — streaming JSONL.
- summary — aggregate writers (results.tsv, run_summary.json, hw_context.json).
"""

from autoinfer.telemetry.events import EventLog
from autoinfer.telemetry.summary import (
    build_run_summary,
    capture_hw_context,
    write_hw_context,
    write_results_tsv,
    write_run_summary,
)

__all__ = [
    "EventLog",
    "build_run_summary",
    "capture_hw_context",
    "write_hw_context",
    "write_results_tsv",
    "write_run_summary",
]
