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
    e2el_ms: dict[str, float]
    goodput_req_per_sec: float
    chosen_request_rate: float | None = None
    """T-38. When the L1 adapter invokes ``run_driver_with_rate_search``,
    this is the request_rate (req/s) the rate-down search settled on —
    the highest sustainable rate meeting the SLO. ``None`` for single-
    shot ``run_driver`` invocations (legacy single-rate behaviour).
    ``float('inf')`` when the initial ``--request-rate inf`` measurement
    already met the SLO."""
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

_E2EL_KEYS = {
    "p50": ("median_e2el_ms", "mean_e2el_ms"),
    "p95": ("p95_e2el_ms",),
    "p99": ("p99_e2el_ms",),
}


def _first_present(payload: dict[str, Any], keys: tuple[str, ...]) -> float:
    for k in keys:
        v = payload.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return 0.0


def _num(value: Any, default: float = 0.0) -> float:
    """Coerce ``value`` to float, substituting ``default`` for None/missing."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_bench_output(payload: dict[str, Any]) -> DriverResult:
    """Parse ``vllm bench serve --save-result`` JSON payload.

    Every value extraction is None-safe because vLLM's bench emits
    null for percentile fields when it lacks enough samples, and for
    `request_goodput` when no SLO was supplied.
    """
    tok = payload.get("output_throughput") or payload.get("total_token_throughput")
    goodput = payload.get("request_goodput")
    if goodput is None:
        goodput = payload.get("request_throughput")
    return DriverResult(
        tokens_per_sec=_num(tok),
        request_throughput=_num(payload.get("request_throughput")),
        ttft_ms={p: _first_present(payload, keys) for p, keys in _TTFT_KEYS.items()},
        tpot_ms={p: _first_present(payload, keys) for p, keys in _TPOT_KEYS.items()},
        e2el_ms={p: _first_present(payload, keys) for p, keys in _E2EL_KEYS.items()},
        goodput_req_per_sec=_num(goodput),
        raw=payload,
    )


_GOODPUT_KEYS: tuple[str, ...] = ("TTFT", "TPOT", "E2E")

# vLLM's ``check_goodput_args`` accepts only lowercase metric names:
# ttft, tpot, itl, e2el. We accept the human-readable upper-case keys
# in the dict (TTFT/TPOT/E2E) for backwards-compat with the T-34 API,
# but translate to vLLM's accepted case at emission time. Confirmed by
# T-37's auto_tune.sh which uses ``--goodput e2el:$MAX_LATENCY_MS``
# and by C04a attempt 6 (2026-05-26) where ``--goodput E2E:500`` was
# rejected with "vllm bench serve failed (exit 1) ... goodput_config_dict
# = check_goodput_args(args)".
_GOODPUT_KEY_TO_VLLM: dict[str, str] = {"TTFT": "ttft", "TPOT": "tpot", "E2E": "e2el"}


def _format_goodput_args(goodput_slo_ms: dict[str, float]) -> list[str]:
    """Emit ``--goodput ttft:X tpot:Y e2el:Z`` argv tokens. Pure.

    vLLM's ``--goodput`` uses ``nargs='+'`` — multiple ``name:value``
    tokens after the flag. Values are millisecond thresholds. Metric
    names MUST be lowercase ``ttft``, ``tpot``, ``e2el`` (vLLM's
    ``check_goodput_args`` is strict). Order is fixed (TTFT, TPOT,
    E2E mapped to ttft, tpot, e2el respectively) so the rendered
    command is stable across re-runs; missing keys are dropped.

    The accepted input dict still uses upper-case keys (TTFT/TPOT/E2E)
    for human readability and backwards-compat with T-34's API; the
    lowercase translation happens here at emission.

    T-34. C04 head-to-head with vLLM's ``benchmarks/auto_tune`` requires
    a goodput SLO to be enforced during the bench; without it
    ``vllm bench serve`` returns plain throughput and the comparable
    objective is lost.
    """
    pieces: list[str] = []
    for key in _GOODPUT_KEYS:
        if key in goodput_slo_ms:
            vllm_key = _GOODPUT_KEY_TO_VLLM[key]
            pieces.append(f"{vllm_key}:{goodput_slo_ms[key]:g}")
    if not pieces:
        return []
    return ["--goodput", *pieces]


def build_bench_command(
    endpoint: str,
    trace_path: Path,
    model: str,
    result_dir: Path,
    result_name: str,
    num_prompts: int | None = None,
    request_rate: float | None = None,
    dataset_name: str = "random",
    random_input_len: int = 128,
    random_output_len: int = 64,
    goodput_slo_ms: dict[str, float] | None = None,
    seed: int | None = None,
) -> list[str]:
    """Assemble ``vllm bench serve`` arguments. Pure.

    ``dataset_name`` defaults to ``random`` because vLLM's CustomDataset
    format is stricter than a simple {"prompt": "..."} JSONL; for
    iteration-zero smoke validation we generate synthetic prompts.
    Set to ``custom`` and supply a compatible ``trace_path`` for real
    workload replay once the harness is validated.

    ``goodput_slo_ms`` (T-34) enables vLLM's SLO-aware goodput
    accounting: dict keys in {"TTFT", "TPOT", "E2E"}, values in
    milliseconds. When present, the result's ``request_goodput`` becomes
    "requests that met every SLO per second" instead of falling through
    to ``request_throughput``.

    ``seed`` (T-35 partial) is the request-generator seed used by
    ``vllm bench serve`` for deterministic sampling. Plumbed here so
    every campaign run can pin reproducibility.
    """
    cmd: list[str] = [
        "vllm", "bench", "serve",
        "--backend", "openai",
        "--base-url", endpoint,
        "--endpoint", "/v1/completions",
        "--model", model,
        "--dataset-name", dataset_name,
        "--save-result",
        "--result-filename", result_name,
        "--result-dir", str(result_dir),
        # T-40: vLLM bench's ``--save-result`` JSON only writes the
        # percentiles for metrics LISTED in --percentile-metrics. The
        # default at the namespace level includes e2el, but the save
        # JSON only includes percentile fields for the explicit list.
        # Without an explicit ``e2el``, the JSON omits ``p99_e2el_ms``
        # entirely; rate-search's SLO check (E2EL P99 <= max_e2e_slo_ms)
        # then reads 0 and falsely concludes "no rate meets SLO" →
        # iterates every rate → hangs the trial. Pass all four metrics
        # explicitly. Confirmed by C04a attempt 10 (2026-05-27): the
        # rate=42 bench saved p99_ttft + p99_tpot + p99_itl but no
        # p99_e2el field.
        "--percentile-metrics", "ttft,tpot,itl,e2el",
    ]
    if dataset_name in ("custom", "sharegpt", "sonnet"):
        cmd.extend(["--dataset-path", str(trace_path)])
    if dataset_name == "random":
        cmd.extend([
            "--random-input-len", str(random_input_len),
            "--random-output-len", str(random_output_len),
        ])
    if num_prompts is not None:
        cmd.extend(["--num-prompts", str(num_prompts)])
    if request_rate is not None:
        cmd.extend(["--request-rate", str(request_rate)])
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    if goodput_slo_ms:
        cmd.extend(_format_goodput_args(goodput_slo_ms))
    return cmd


def run_driver(
    endpoint: str,
    trace_path: Path,
    model: str,
    result_dir: Path,
    result_name: str = "bench.json",
    num_prompts: int | None = 64,
    request_rate: float | None = None,
    timeout_s: int = 1800,
    dataset_name: str = "random",
    goodput_slo_ms: dict[str, float] | None = None,
    seed: int | None = None,
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
        dataset_name=dataset_name,
        goodput_slo_ms=goodput_slo_ms,
        seed=seed,
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


def run_driver_with_rate_search(
    endpoint: str,
    trace_path: Path,
    model: str,
    result_dir: Path,
    result_name_prefix: str,
    *,
    max_e2e_slo_ms: float,
    num_prompts: int | None = 64,
    timeout_per_bench_s: int = 1800,
    dataset_name: str = "random",
    goodput_slo_ms: dict[str, float] | None = None,
    seed: int | None = None,
    min_request_rate: int = 1,
) -> DriverResult:
    """Mirror ``auto_tune.sh``'s rate-down search: find the highest
    sustainable request rate that meets the E2E SLO.

    T-38. autoinfer's single-shot driver invocation always saturated
    the queue (default ``--request-rate inf``), so every cell measured
    goodput=0 regardless of config quality. ``auto_tune.sh`` iterates
    rates downward from ``int(throughput) + 1`` until P99 E2EL <=
    ``MAX_LATENCY_ALLOWED_MS``. This function reproduces that loop.

    Algorithm (matches benchmarks/auto_tune/auto_tune.sh lines 161-228):
    1. Initial measurement at ``--request-rate inf``. If P99 E2EL <=
       ``max_e2e_slo_ms``, return that result (the server can handle
       saturated load within SLO).
    2. Otherwise: start at ``rate = int(request_throughput) + 1`` and
       decrement by 1 each iteration. The first rate where P99 E2EL
       meets the SLO is the answer.
    3. If we drop to ``min_request_rate`` (default 1) without meeting
       SLO, return the last measurement with ``chosen_request_rate=
       min_request_rate``. The caller decides whether
       ``goodput_req_per_sec`` (which will be 0 in that case) is a
       failure.

    Each per-rate bench writes its own JSON file under ``result_dir``
    named ``{result_name_prefix}_rate_{rate}.json`` (or ``_rate_inf``
    for the initial measurement). Matches ``auto_tune.sh``'s
    ``bm_log_..._requestrate_*`` pattern; lets post-hoc analysis see
    the rate-vs-latency curve per cell.

    Returns a single ``DriverResult`` — the one that met the SLO (or
    the last one tried if none met it). ``chosen_request_rate`` is
    populated with the rate value (``float('inf')`` for inf-rate
    success, or the integer rate that met).
    """
    initial = run_driver(
        endpoint=endpoint,
        trace_path=trace_path,
        model=model,
        result_dir=result_dir,
        result_name=f"{result_name_prefix}_rate_inf.json",
        num_prompts=num_prompts,
        request_rate=None,
        timeout_s=timeout_per_bench_s,
        dataset_name=dataset_name,
        goodput_slo_ms=goodput_slo_ms,
        seed=seed,
    )
    if initial.e2el_ms.get("p99", 0.0) <= max_e2e_slo_ms and initial.e2el_ms.get("p99", 0.0) > 0:
        # initial inf-rate already meets SLO
        return _with_chosen_rate(initial, float("inf"))

    # Rate-down search.
    start_rate = max(int(initial.request_throughput) + 1, min_request_rate)
    last_result = initial
    for rate in range(start_rate, min_request_rate - 1, -1):
        res = run_driver(
            endpoint=endpoint,
            trace_path=trace_path,
            model=model,
            result_dir=result_dir,
            result_name=f"{result_name_prefix}_rate_{rate}.json",
            num_prompts=num_prompts,
            request_rate=float(rate),
            timeout_s=timeout_per_bench_s,
            dataset_name=dataset_name,
            goodput_slo_ms=goodput_slo_ms,
            seed=seed,
        )
        last_result = res
        if 0 < res.e2el_ms.get("p99", 0.0) <= max_e2e_slo_ms:
            return _with_chosen_rate(res, float(rate))

    # Never met SLO; return the last measurement with the floor rate.
    return _with_chosen_rate(last_result, float(min_request_rate))


def _with_chosen_rate(result: DriverResult, rate: float) -> DriverResult:
    """Return a copy of ``result`` with ``chosen_request_rate=rate``.

    DriverResult is frozen so we have to reconstruct it.
    """
    return DriverResult(
        tokens_per_sec=result.tokens_per_sec,
        request_throughput=result.request_throughput,
        ttft_ms=dict(result.ttft_ms),
        tpot_ms=dict(result.tpot_ms),
        e2el_ms=dict(result.e2el_ms),
        goodput_req_per_sec=result.goodput_req_per_sec,
        chosen_request_rate=rate,
        raw=dict(result.raw),
    )
