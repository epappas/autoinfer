"""Real L1 engine-config adapter.

Spawns a vLLM subprocess with the trial config, drives it with
``vllm bench serve`` against a trace, gates the output via a live
reference-replica KL check, composes a ``Measurement``. All exceptions
convert to ``FailureRecord`` (P9) rather than raising.

Startup / teardown / subprocess management is impure and requires a GPU
+ vLLM install for end-to-end tests (marked ``gpu``). The
``compose_measurement`` function and the surface helpers are pure.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autoinfer.harness.driver import (
    DriverResult,
    run_driver,
    run_driver_with_rate_search,
)
from autoinfer.harness.failure import FailureKind, FailureRecord
from autoinfer.harness.gate import GateResult, run_gate
from autoinfer.harness.ledger import Measurement
from autoinfer.layers import TrialInput, TrialOutput
from autoinfer.layers.l1_engine.surface import (
    KnobCatalog,
    build_vllm_serve_args,
    to_surrogate_surface,
    violates_constraints,
)


def query_gpu_memory_used_gb(device_id: int = 0, timeout_s: float = 3.0) -> float | None:
    """Best-effort read of current GPU memory usage in GiB via ``nvidia-smi``."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                f"-i={device_id}",
            ],
            text=True,
            timeout=timeout_s,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None
    try:
        return float(out.strip().splitlines()[0]) / 1024.0
    except (ValueError, IndexError):
        return None


def _kl_percentiles(per_prompt_kl: tuple[float, ...]) -> dict[str, float]:
    """Per-prompt KL distribution shape — addresses TODO T-25.

    Trial JSON's ``Measurement.extra`` is ``dict[str, float]`` so we
    can't pack the full per-prompt list there; percentiles capture the
    distribution shape without bloating the artifact. Useful for
    post-hoc analysis: a kept trial with mean_kl=2.0 but kl_p99=12 is
    different from a trial with mean_kl=2.0 and kl_p99=2.5.
    """
    if not per_prompt_kl:
        return {}
    sorted_kl = sorted(per_prompt_kl)
    n = len(sorted_kl)

    def pct(p: float) -> float:
        idx = min(n - 1, max(0, int(p * n)))
        return sorted_kl[idx]

    return {
        "kl_min": sorted_kl[0],
        "kl_p50": pct(0.50),
        "kl_p90": pct(0.90),
        "kl_p95": pct(0.95),
        "kl_p99": pct(0.99),
    }


def compose_measurement(
    driver: DriverResult, gate: GateResult, peak_hbm_gb: float
) -> Measurement:
    """Build a ``Measurement`` from driver + gate results. Pure.

    ``extra["goodput"]`` is the canonical goodput axis (legacy alias);
    T-34 adds the more descriptive ``extra["goodput_req_per_sec"]`` so
    downstream tools that select on goodput-under-SLO have a clearly-
    named field. Both fields carry the same value to keep older
    analysers working.
    """
    goodput = driver.goodput_req_per_sec
    extra: dict[str, float] = {
        "ttft_p50_ms": driver.ttft_ms.get("p50", 0.0),
        "tpot_p50_ms": driver.tpot_ms.get("p50", 0.0),
        "e2el_p50_ms": driver.e2el_ms.get("p50", 0.0),
        "e2el_p99_ms": driver.e2el_ms.get("p99", 0.0),
        "goodput": goodput,
        "goodput_req_per_sec": goodput,
        "max_kl": gate.max_kl,
    }
    # T-38: surface the rate the rate-down search settled on. When the
    # single-shot driver path runs (legacy non-SLO mode),
    # chosen_request_rate is None and the field is omitted.
    if driver.chosen_request_rate is not None:
        # Measurement.extra is dict[str, float] so float('inf') survives
        # round-trip through JSON as "Infinity"; the post-hoc analyzer
        # treats it as "saturated load also met SLO".
        extra["chosen_request_rate"] = driver.chosen_request_rate
    extra.update(_kl_percentiles(gate.per_prompt_kl))
    return Measurement(
        tokens_per_sec=driver.tokens_per_sec,
        ttft_p99_ms=driver.ttft_ms.get("p99", 0.0),
        tpot_p99_ms=driver.tpot_ms.get("p99", 0.0),
        peak_hbm_gb=peak_hbm_gb,
        kl_divergence=gate.mean_kl,
        extra=extra,
    )


@dataclass
class L1EngineAdapter:
    model: str
    catalog: KnobCatalog
    trace_path: Path
    reference_uri: str
    quality_prompts: list[str]
    max_kl: float
    result_dir: Path
    layer_name: str = "l1_engine"
    batch_sizes: tuple[int, ...] = (1, 8, 64)
    candidate_port: int = 8000
    startup_timeout_s: int = 600
    driver_timeout_s: int = 1800
    gpu_device_id: int = 0
    dataset_name: str = "random"
    num_prompts: int = 64
    gate_concurrency: int = 4
    goodput_slo_ms: dict[str, float] | None = None
    """Optional ``{TTFT|TPOT|E2E -> ms}`` SLO passed to ``vllm bench
    serve --goodput``. When set, ``driver.goodput_req_per_sec`` measures
    "requests meeting all SLOs per second" instead of falling through to
    raw throughput. T-34."""
    bench_seed: int | None = None
    """Deterministic ``vllm bench serve`` request-generator seed. T-35
    partial: the L1 adapter passes this seed on every trial so the same
    config produces the same request stream across re-runs."""
    multiprocessing_v1: bool = True
    """T-36 determinism lever. When False, the candidate subprocess is
    started with ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` so V1's
    multi-process backend (a known source of batch-composition
    nondeterminism) is disabled. Costs throughput, stabilises KL."""
    enforce_batch_invariance: bool = True
    """T-36 determinism lever. When True, a gate run that reports
    ``batch_invariant=False`` is rejected as QUALITY_INVARIANCE; when
    False, the gate accepts on KL alone. Default True; set False only
    for kernels with a known valid invariance violation."""
    _process: subprocess.Popen[bytes] | None = field(default=None, init=False, repr=False)
    _current_trial_id: str | None = field(default=None, init=False, repr=False)

    def surface(self) -> dict[str, Any]:
        return to_surrogate_surface(self.catalog)

    def run(self, trial: TrialInput) -> TrialOutput:
        violations = violates_constraints(trial.config, self.catalog)
        if violations:
            return TrialOutput(
                measurement=None,
                failure=self._fail(
                    trial,
                    FailureKind.STARTUP,
                    f"config violates {len(violations)} constraint(s): {','.join(violations)}",
                ),
            )
        try:
            self._start_candidate(trial.config, trial_id=trial.trial_id)
        except Exception as e:
            self._stop_candidate()
            return TrialOutput(
                measurement=None,
                failure=self._fail(trial, FailureKind.STARTUP, f"startup failed: {e}"),
            )
        try:
            return self._run_benchmarks(trial)
        finally:
            self._stop_candidate()

    def teardown(self) -> None:
        self._stop_candidate()

    def _run_benchmarks(self, trial: TrialInput) -> TrialOutput:
        endpoint = f"http://127.0.0.1:{self.candidate_port}"
        # T-38: when a goodput SLO is set with an E2E threshold, do a
        # rate-down search (matching auto_tune.sh's algorithm) instead
        # of a single-shot bench at request_rate=inf. Without the search,
        # every cell measures goodput=0 at saturation regardless of
        # how good the config is.
        use_rate_search = (
            self.goodput_slo_ms is not None
            and "E2E" in self.goodput_slo_ms
            and self.goodput_slo_ms["E2E"] > 0
        )
        try:
            if use_rate_search:
                assert self.goodput_slo_ms is not None  # narrowing for mypy
                driver = run_driver_with_rate_search(
                    endpoint=endpoint,
                    trace_path=self.trace_path,
                    model=self.model,
                    result_dir=self.result_dir,
                    result_name_prefix=f"{trial.trial_id}_bench",
                    max_e2e_slo_ms=self.goodput_slo_ms["E2E"],
                    num_prompts=self.num_prompts,
                    timeout_per_bench_s=self.driver_timeout_s,
                    dataset_name=self.dataset_name,
                    goodput_slo_ms=self.goodput_slo_ms,
                    seed=self.bench_seed,
                )
            else:
                driver = run_driver(
                    endpoint=endpoint,
                    trace_path=self.trace_path,
                    model=self.model,
                    result_dir=self.result_dir,
                    result_name=f"{trial.trial_id}_bench.json",
                    timeout_s=self.driver_timeout_s,
                    dataset_name=self.dataset_name,
                    num_prompts=self.num_prompts,
                    goodput_slo_ms=self.goodput_slo_ms,
                    seed=self.bench_seed,
                )
        except (subprocess.TimeoutExpired, RuntimeError) as e:
            return TrialOutput(
                measurement=None,
                failure=self._fail(trial, FailureKind.HANG, f"driver failed: {e}"),
            )
        try:
            gate = run_gate(
                candidate_endpoint=endpoint,
                reference_endpoint=self.reference_uri,
                model=self.model,
                prompts=self.quality_prompts,
                batch_sizes=self.batch_sizes,
                concurrency=self.gate_concurrency,
            )
        except Exception as e:
            return TrialOutput(
                measurement=None,
                failure=self._fail(trial, FailureKind.UNKNOWN, f"gate failed: {e}"),
            )
        kl_fail = gate.mean_kl > self.max_kl
        invariance_fail = self.enforce_batch_invariance and not gate.batch_invariant
        if kl_fail or invariance_fail:
            kind = (
                FailureKind.QUALITY_INVARIANCE
                if invariance_fail
                else FailureKind.QUALITY_KL
            )
            return TrialOutput(
                measurement=None,
                failure=self._fail(
                    trial,
                    kind,
                    f"gate rejected mean_kl={gate.mean_kl:.4f} invariant={gate.batch_invariant} enforced={self.enforce_batch_invariance}",
                ),
            )
        peak_hbm = query_gpu_memory_used_gb(self.gpu_device_id) or 0.0
        return TrialOutput(
            measurement=compose_measurement(driver, gate, peak_hbm),
            failure=None,
        )

    def _start_candidate(
        self, config: dict[str, Any], *, trial_id: str | None = None
    ) -> None:
        if self._process is not None:
            raise RuntimeError("candidate already running")
        args, extra_env = build_vllm_serve_args(
            self.model, self.candidate_port, config, self.catalog
        )
        env = os.environ.copy()
        env.update(extra_env)
        if not self.multiprocessing_v1:
            env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        # Remember the trial_id so _wait_ready can write the candidate's
        # full stderr to a per-trial log file when startup fails. The
        # FailureRecord.message still gets the truncated tail (500-char
        # slice) for backward compatibility; the file has the full
        # context needed to diagnose.
        self._current_trial_id = trial_id
        self._process = subprocess.Popen(
            args, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        self._wait_ready()

    def _wait_ready(self) -> None:
        assert self._process is not None
        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                code = self._process.returncode
                stderr = self._drain_stderr()
                self._archive_candidate_stderr(stderr)
                raise RuntimeError(f"candidate exited during startup (code {code}): {stderr[-400:]}")
            if _tcp_open("127.0.0.1", self.candidate_port, timeout_s=1.0):
                return
            time.sleep(2.0)
        # On readiness timeout, also capture whatever stderr exists so
        # the failure has diagnostic context.
        stderr = self._drain_stderr()
        self._archive_candidate_stderr(stderr)
        raise TimeoutError(f"candidate not ready on port {self.candidate_port}")

    def _archive_candidate_stderr(self, stderr: str) -> None:
        """Write the candidate's full stderr to ``result_dir/<trial_id>_candidate.stderr.log``.

        The FailureRecord.message captures a 500-char tail (kept for
        backward compatibility); the file has the full crash context.
        Empty stderr → no file written.
        """
        if not stderr or self._current_trial_id is None:
            return
        try:
            self.result_dir.mkdir(parents=True, exist_ok=True)
            target = self.result_dir / f"{self._current_trial_id}_candidate.stderr.log"
            target.write_text(stderr)
        except OSError:
            # Diagnostic-only; failure-record path is the authoritative
            # error surface.
            return

    def _drain_stderr(self) -> str:
        if self._process is None or self._process.stderr is None:
            return ""
        try:
            return self._process.stderr.read().decode("utf-8", errors="replace")
        except OSError:
            return ""

    def _stop_candidate(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self._process = None

    def _fail(
        self, trial: TrialInput, kind: FailureKind, msg: str
    ) -> FailureRecord:
        return FailureRecord(
            kind=kind,
            message=msg[:500],
            trial_id=trial.trial_id,
            layer=self.layer_name,
        )


def _tcp_open(host: str, port: int, timeout_s: float) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout_s)
        try:
            s.connect((host, port))
        except (OSError, TimeoutError):
            return False
        return True
