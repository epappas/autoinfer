"""Workload driver that replays a trace through a running engine endpoint.

Wraps ``vllm bench serve``. The parser and command builder are pure; the
subprocess runner is impure and raises on failure so adapters can convert
to ``FailureRecord``.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DriverResult:
    tokens_per_sec: float
    request_throughput: float
    ttft_ms: dict[str, float]
    tpot_ms: dict[str, float]
    goodput_req_per_sec: float
    raw: dict[str, Any] = field(default_factory=dict)


_TTFT_KEYS = {
    "p50": ("median_ttft_ms", "mean_ttft_ms"),
    "p95": ("p95_ttft_ms",),
    "p99": ("p99_ttft_ms",),
}

_TPOT_KEYS = {
    "p50": ("median_tpot_ms", "mean_tpot_ms"),
    "p95": ("p95_tpot_ms",),
    "p99": ("p99_tpot_ms",),
}


def _first_present(payload: dict[str, Any], keys: tuple[str, ...]) -> float:
    for k in keys:
        if k in payload and payload[k] is not None:
            return float(payload[k])
    return 0.0


def parse_bench_output(payload: dict[str, Any]) -> DriverResult:
    """Parse ``vllm bench serve --save-result`` JSON payload."""
    tok = payload.get("output_throughput")
    if tok is None:
        tok = payload.get("total_token_throughput", 0.0)
    return DriverResult(
        tokens_per_sec=float(tok),
        request_throughput=float(payload.get("request_throughput", 0.0)),
        ttft_ms={p: _first_present(payload, keys) for p, keys in _TTFT_KEYS.items()},
        tpot_ms={p: _first_present(payload, keys) for p, keys in _TPOT_KEYS.items()},
        goodput_req_per_sec=float(
            payload.get("request_goodput", payload.get("request_throughput", 0.0))
        ),
        raw=payload,
    )


def build_bench_command(
    endpoint: str,
    trace_path: Path,
    model: str,
    result_dir: Path,
    result_name: str,
    num_prompts: int | None = None,
    request_rate: float | None = None,
) -> list[str]:
    """Assemble ``vllm bench serve`` arguments. Pure."""
    cmd: list[str] = [
        "vllm", "bench", "serve",
        "--backend", "openai-chat",
        "--base-url", endpoint,
        "--model", model,
        "--dataset-name", "custom",
        "--dataset-path", str(trace_path),
        "--save-result",
        "--result-filename", result_name,
        "--result-dir", str(result_dir),
    ]
    if num_prompts is not None:
        cmd.extend(["--num-prompts", str(num_prompts)])
    if request_rate is not None:
        cmd.extend(["--request-rate", str(request_rate)])
    return cmd


def run_driver(
    endpoint: str,
    trace_path: Path,
    model: str,
    result_dir: Path,
    result_name: str = "bench.json",
    num_prompts: int | None = None,
    request_rate: float | None = None,
    timeout_s: int = 1800,
) -> DriverResult:
    """Execute ``vllm bench serve``; parse and return ``DriverResult``.

    Raises ``RuntimeError`` on non-zero exit. Adapters catch and translate.
    """
    result_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_bench_command(
        endpoint=endpoint,
        trace_path=trace_path,
        model=model,
        result_dir=result_dir,
        result_name=result_name,
        num_prompts=num_prompts,
        request_rate=request_rate,
    )
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        raise RuntimeError(
            f"vllm bench serve failed (exit {proc.returncode}): {proc.stderr[-800:]}"
        )
    save_path = result_dir / result_name
    with save_path.open() as f:
        payload = json.load(f)
    return parse_bench_output(payload)
