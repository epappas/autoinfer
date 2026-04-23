"""Streaming event log (JSONL), modeled on ../autoresearch-rl.

Every meaningful moment in a run emits one JSON line with a stable
schema so downstream analysis (article plots, regression comparisons,
post-hoc debugging) can reconstruct the full trajectory without
parsing controller internals.

Event types (see autoinfer/controller/continuous.py for emission sites):

- ``run_start``       once at campaign launch
- ``warmstart_batch`` LLM proposer returns N configs
- ``operator_call``   operator fired, returns M configs
- ``surrogate_ask``   Optuna picks a config
- ``trial_start``     adapter.run about to fire
- ``trial_complete``  measurement or failure recorded
- ``ledger_stale``    stale-signal propagation fires
- ``run_end``         final aggregate
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any


class EventLog:
    """Append-only JSONL writer with monotonic event IDs."""

    def __init__(self, path: Path, run_id: str, schema: str = "v1") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._run_id = run_id
        self._schema = schema
        self._counter = 0

    @property
    def run_id(self) -> str:
        return self._run_id

    def emit(self, type_: str, **fields: Any) -> None:
        self._counter += 1
        payload = {
            "schema": self._schema,
            "run_id": self._run_id,
            "event_id": uuid.uuid4().hex[:12],
            "event_n": self._counter,
            "ts": time.time(),
            "type": type_,
            **fields,
        }
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str, ensure_ascii=False) + "\n")
