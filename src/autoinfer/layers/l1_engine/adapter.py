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

from autoinfer.harness.driver import DriverResult, run_driver
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


def compose_measurement(
    driver: DriverResult, gate: GateResult, peak_hbm_gb: float
) -> Measurement:
    """Build a ``Measurement`` from driver + gate results. Pure."""
    return Measurement(
        tokens_per_sec=driver.tokens_per_sec,
        ttft_p99_ms=driver.ttft_ms.get("p99", 0.0),
        tpot_p99_ms=driver.tpot_ms.get("p99", 0.0),
        peak_hbm_gb=peak_hbm_gb,
        kl_divergence=gate.mean_kl,
        extra={
            "ttft_p50_ms": driver.ttft_ms.get("p50", 0.0),
            "tpot_p50_ms": driver.tpot_ms.get("p50", 0.0),
            "goodput": driver.goodput_req_per_sec,
            "max_kl": gate.max_kl,
        },
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
    _process: subprocess.Popen[bytes] | None = field(default=None, init=False, repr=False)

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
            self._start_candidate(trial.config)
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
        try:
            driver = run_driver(
                endpoint=endpoint,
                trace_path=self.trace_path,
                model=self.model,
                result_dir=self.result_dir,
                result_name=f"{trial.trial_id}_bench.json",
                timeout_s=self.driver_timeout_s,
                dataset_name=self.dataset_name,
                num_prompts=self.num_prompts,
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
        if not gate.passes(self.max_kl):
            kind = (
                FailureKind.QUALITY_INVARIANCE
                if not gate.batch_invariant
                else FailureKind.QUALITY_KL
            )
            return TrialOutput(
                measurement=None,
                failure=self._fail(
                    trial,
                    kind,
                    f"gate rejected mean_kl={gate.mean_kl:.4f} invariant={gate.batch_invariant}",
                ),
            )
        peak_hbm = query_gpu_memory_used_gb(self.gpu_device_id) or 0.0
        return TrialOutput(
            measurement=compose_measurement(driver, gate, peak_hbm),
            failure=None,
        )

    def _start_candidate(self, config: dict[str, Any]) -> None:
        if self._process is not None:
            raise RuntimeError("candidate already running")
        args, extra_env = build_vllm_serve_args(
            self.model, self.candidate_port, config, self.catalog
        )
        env = os.environ.copy()
        env.update(extra_env)
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
                raise RuntimeError(f"candidate exited during startup (code {code}): {stderr[-400:]}")
            if _tcp_open("127.0.0.1", self.candidate_port, timeout_s=1.0):
                return
            time.sleep(2.0)
        raise TimeoutError(f"candidate not ready on port {self.candidate_port}")

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
