"""End-to-end L3 adapter — kernel-into-vLLM with real serving measurement.

This is the adapter the user's "the real fix" pushback was about. The
old ``L3KernelAdapter`` (still shipped, kept for CPU-side correctness
and dev workflows) timed kernels in isolation and reported
``ops/sec`` — not unit-comparable to L1/L2's end-to-end token
throughput. The ``pareto_eligible=False`` flag was a workaround that
excluded L3 from the joint frontier.

This adapter is the actual kernel-into-vLLM integration:
1. Validate config (target_op, dtype, shape_regime, source, entry_fn).
2. Compile the candidate kernel via ``compile_candidate`` (same path
   the standalone adapter uses).
3. Verify correctness against the PyTorch reference on the test matrix
   (atol/rtol gate). A kernel that fails correctness here saves the
   cost of spinning up vLLM only to produce wrong tokens.
4. Render an ``InjectionPlan`` into a wrapper Python script that
   patches ``vllm.RMSNorm.forward_cuda`` (or SiluAndMul etc.) before
   exec'ing ``vllm serve``.
5. Spawn the wrapper as a subprocess. Wait for the serving port.
6. Run the same L1 driver + reference-replica gate against the
   patched server. Records token throughput, TPOT, KL — the same
   axes L1/L2 measure.
7. Return ``Measurement`` with ``pareto_eligible=True``. The kernel
   win (or loss) is now directly comparable to L1/L2 entries on the
   joint Pareto frontier.

The standalone ``L3KernelAdapter`` is preserved as the cheap CPU-only
adapter for development / regression tests / Pareto-ineligible
correctness benchmarking. ``L3VllmKernelAdapter`` is the production
adapter the joint campaign should use when a GPU + vLLM are
available.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from autoinfer.harness.driver import run_driver
from autoinfer.harness.failure import FailureKind, FailureRecord
from autoinfer.harness.gate import run_gate
from autoinfer.harness.ledger import Measurement
from autoinfer.layers import TrialInput, TrialOutput
from autoinfer.layers.l3_kernel.baselines import REFERENCES, KernelFn
from autoinfer.layers.l3_kernel.injector import (
    SUPPORTED_TARGET_OPS,
    InjectionPlan,
    render_wrapper_script,
)
from autoinfer.layers.l3_kernel.surface import (
    REFERENCE_SOURCES,
    compile_candidate,
    make_inputs,
    resolve_dtype,
    test_shapes,
    to_surrogate_surface,
)


@dataclass
class L3VllmKernelAdapter:
    """L3 adapter that ships kernels into vLLM and measures end-to-end.

    Mirrors ``L1EngineAdapter`` for the bench + gate path; the only
    structural difference is how the candidate vLLM is launched
    (a wrapper that patches the target op before exec'ing
    ``vllm serve``).
    """

    model: str
    catalog: Any
    """``KnobCatalog`` from ``surface.load_catalog`` — only used for the
    surrogate's surface; the LLM proposer is what produces the kernel
    source the adapter actually runs."""

    trace_path: Path
    reference_uri: str
    quality_prompts: list[str]
    max_kl: float
    result_dir: Path
    layer_name: str = "l3_kernel"
    batch_sizes: tuple[int, ...] = (1, 8, 64)
    candidate_port: int = 8200
    """Distinct from L1's default 8000 so a campaign running L1 + L3
    in the same container doesn't collide on the local port."""

    startup_timeout_s: int = 600
    driver_timeout_s: int = 1800
    gpu_device_id: int = 0
    dataset_name: str = "random"
    num_prompts: int = 64
    gate_concurrency: int = 4
    perf_repeats: int = 5
    warmup_runs: int = 2
    atol: float = 1e-3
    rtol: float = 1e-3
    extra_vllm_args: tuple[str, ...] = ()
    """Additional ``vllm serve`` CLI args (e.g. ``--gpu-memory-utilization
    0.85``) the campaign config may pass through. These are appended
    after the canonical ``vllm serve <model> --port <port>`` prefix."""

    _process: subprocess.Popen[bytes] | None = field(default=None, init=False, repr=False)
    _wrapper_path: Path | None = field(default=None, init=False, repr=False)
    _wrapper_stdout_path: Path | None = field(default=None, init=False, repr=False)
    _wrapper_stderr_path: Path | None = field(default=None, init=False, repr=False)
    _current_trial_id: str | None = field(default=None, init=False, repr=False)

    def surface(self) -> dict[str, Any]:
        return to_surrogate_surface(self.catalog)

    def run(self, trial: TrialInput) -> TrialOutput:
        cfg = trial.config
        # Required keys: target_op, dtype, shape_regime, source, entry_fn.
        # The surrogate alone produces only the first three; the LLM
        # proposer (KernelProposer) supplies source + entry_fn. For
        # surrogate-only paths we fall back to the reference source so
        # the adapter is exercisable without an LLM.
        missing = [k for k in ("target_op", "dtype", "shape_regime") if k not in cfg]
        if missing:
            return self._fail(trial, FailureKind.STARTUP, f"missing config keys: {missing}")

        target_op = str(cfg["target_op"])
        if target_op not in SUPPORTED_TARGET_OPS:
            return self._fail(
                trial,
                FailureKind.STARTUP,
                f"target_op {target_op!r} not supported by vLLM injector "
                f"(supported: {sorted(SUPPORTED_TARGET_OPS)})",
            )
        reference = REFERENCES.get(target_op)
        if reference is None:
            return self._fail(trial, FailureKind.STARTUP, f"no reference impl for {target_op!r}")

        try:
            dtype = resolve_dtype(str(cfg["dtype"]))
            shapes = test_shapes(target_op, str(cfg["shape_regime"]))
        except ValueError as e:
            return self._fail(trial, FailureKind.STARTUP, str(e))

        source = cfg.get("source")
        entry_fn = cfg.get("entry_fn")
        if not source or not entry_fn:
            ref_entry, ref_source = REFERENCE_SOURCES[target_op]
            source = source or ref_source
            entry_fn = entry_fn or ref_entry

        # Compile + correctness gate before paying for vLLM startup.
        try:
            candidate = compile_candidate(str(source), str(entry_fn))
        except (ValueError, KeyError) as e:
            return self._fail(trial, FailureKind.STARTUP, f"compile failed: {e}")
        try:
            max_abs_err = self._check_correctness(candidate, reference, shapes, dtype)
        except Exception as e:  # noqa: BLE001
            return self._fail(trial, FailureKind.STARTUP, f"correctness raised: {e}")
        if max_abs_err is None:
            return self._fail(
                trial,
                FailureKind.QUALITY_KL,
                f"candidate exceeded atol={self.atol}/rtol={self.rtol}",
            )

        # Build the injection plan + wrapper script + spawn vLLM.
        plan = InjectionPlan(target_op=target_op, entry_fn=str(entry_fn), source=str(source))
        self._current_trial_id = trial.trial_id
        try:
            self._start_patched_vllm(plan)
        except Exception as e:  # noqa: BLE001
            log_hint = self._log_hint(trial.trial_id)
            self._stop_candidate()
            return self._fail(
                trial,
                FailureKind.STARTUP,
                f"vllm startup failed: {e}{log_hint}",
            )

        try:
            return self._run_benchmarks(trial, max_abs_err)
        finally:
            self._stop_candidate()

    def teardown(self) -> None:
        self._stop_candidate()

    def _check_correctness(
        self,
        candidate: Any,
        reference: KernelFn,
        shapes: tuple[Any, ...],
        dtype: torch.dtype,
    ) -> float | None:
        """Same gate as the standalone L3 adapter — atol/rtol vs reference."""
        max_err = 0.0
        for i, spec in enumerate(shapes):
            inputs = make_inputs(spec, dtype, seed=17 + i)
            expected = reference(*inputs)
            actual = candidate(*inputs)
            if actual.shape != expected.shape:
                return None
            diff = (actual.to(torch.float32) - expected.to(torch.float32)).abs()
            tol = self.atol + self.rtol * expected.to(torch.float32).abs()
            if (diff > tol).any().item():
                return None
            max_err = max(max_err, float(diff.max().item()))
        return max_err

    def _run_benchmarks(self, trial: TrialInput, max_abs_err: float) -> TrialOutput:
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
            return self._fail(trial, FailureKind.HANG, f"driver failed: {e}")
        try:
            gate = run_gate(
                candidate_endpoint=endpoint,
                reference_endpoint=self.reference_uri,
                model=self.model,
                prompts=self.quality_prompts,
                batch_sizes=self.batch_sizes,
                concurrency=self.gate_concurrency,
            )
        except Exception as e:  # noqa: BLE001
            return self._fail(trial, FailureKind.UNKNOWN, f"gate failed: {e}")
        if not gate.passes(self.max_kl):
            kind = (
                FailureKind.QUALITY_INVARIANCE
                if not gate.batch_invariant
                else FailureKind.QUALITY_KL
            )
            return self._fail(
                trial,
                kind,
                f"gate rejected mean_kl={gate.mean_kl:.4f} invariant={gate.batch_invariant}",
            )
        peak_hbm = _query_gpu_memory_gb(self.gpu_device_id) or 0.0
        meas = Measurement(
            tokens_per_sec=driver.tokens_per_sec,
            ttft_p99_ms=driver.ttft_ms.get("p99", 0.0),
            tpot_p99_ms=driver.tpot_ms.get("p99", 0.0),
            peak_hbm_gb=peak_hbm,
            kl_divergence=gate.mean_kl,
            extra={
                "max_abs_err": max_abs_err,
                "ttft_p50_ms": driver.ttft_ms.get("p50", 0.0),
                "tpot_p50_ms": driver.tpot_ms.get("p50", 0.0),
                "goodput": driver.goodput_req_per_sec,
                "max_kl_observed": gate.max_kl,
            },
        )
        # End-to-end serving measurement IS unit-comparable to L1/L2.
        return TrialOutput(measurement=meas, failure=None, pareto_eligible=True)

    def _start_patched_vllm(self, plan: InjectionPlan) -> None:
        if self._process is not None:
            raise RuntimeError("candidate already running")
        argv = self._build_vllm_argv()
        wrapper_src = render_wrapper_script(plan, argv)
        # Persist as a temp .py so Triton's inspect.getsource works on
        # any @triton.jit kernel inside the source.
        wrapper_dir = Path(tempfile.gettempdir()) / "autoinfer-l3-wrappers"
        wrapper_dir.mkdir(exist_ok=True)
        path = wrapper_dir / f"l3_wrapper_{uuid.uuid4().hex[:8]}.py"
        path.write_text(wrapper_src, encoding="utf-8")
        self._wrapper_path = path

        # Stream wrapper subprocess stdout/stderr to per-trial files
        # under result_dir so the FULL vLLM crash trace is recoverable
        # even after the trial JSON's failure-message truncation. Without
        # this every L3 STARTUP fail reads "Engine core ini..." with the
        # actual cause below the cutoff (smoke 2026-04-25 22:55 surfaced
        # this).
        self.result_dir.mkdir(parents=True, exist_ok=True)
        tid = self._current_trial_id or "unknown"
        self._wrapper_stdout_path = self.result_dir / f"{tid}_vllm.out"
        self._wrapper_stderr_path = self.result_dir / f"{tid}_vllm.err"
        env = os.environ.copy()
        # Use the same Python that's running autoinfer (carries the venv
        # with vllm + triton installed).
        self._process = subprocess.Popen(
            [sys.executable, str(path)],
            env=env,
            stdout=self._wrapper_stdout_path.open("wb"),
            stderr=self._wrapper_stderr_path.open("wb"),
        )
        self._wait_ready()

    def _build_vllm_argv(self) -> list[str]:
        argv: list[str] = [
            "vllm",
            "serve",
            self.model,
            "--port",
            str(self.candidate_port),
        ]
        argv.extend(self.extra_vllm_args)
        return argv

    def _wait_ready(self) -> None:
        assert self._process is not None
        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                code = self._process.returncode
                # The full stderr is in self._wrapper_stderr_path; the
                # trial JSON's failure message includes a hint pointing
                # to it. Surface the LAST chunk so the message itself
                # is informative even without opening the file.
                tail = self._read_stderr_tail(2000)
                raise RuntimeError(
                    f"patched vllm exited during startup (code {code}); "
                    f"last 2000 chars of stderr:\n{tail}"
                )
            if _tcp_open("127.0.0.1", self.candidate_port, timeout_s=1.0):
                return
            time.sleep(2.0)
        raise TimeoutError(
            f"patched vllm not ready on port {self.candidate_port} "
            f"(timeout={self.startup_timeout_s}s)"
        )

    def _read_stderr_tail(self, n: int) -> str:
        """Read the last n chars of the wrapper's stderr file, if any."""
        if self._wrapper_stderr_path is None:
            return ""
        try:
            data = self._wrapper_stderr_path.read_text(errors="replace")
        except OSError:
            return ""
        return data[-n:]

    def _log_hint(self, trial_id: str) -> str:
        """Append-friendly pointer to the per-trial vLLM log files."""
        if self._wrapper_stderr_path is None:
            return ""
        return (
            f" — full stderr at "
            f"{self._wrapper_stderr_path.relative_to(self.result_dir.parent)} "
            f"if reachable"
        )

    def _stop_candidate(self) -> None:
        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
            # Close the file handles we passed as stdout/stderr.
            import contextlib as _ctx

            for stream in (self._process.stdout, self._process.stderr):
                if stream is not None:
                    with _ctx.suppress(OSError):
                        stream.close()
            self._process = None
        # Best-effort cleanup of the wrapper file. It's small, but
        # leaving stale wrappers around offends the no-temp-file-leaks
        # invariant the original compile_candidate violated. We keep
        # the per-trial stdout/stderr files — they're under result_dir
        # so they ride along with trial JSON artifacts.
        if self._wrapper_path is not None:
            import contextlib

            with contextlib.suppress(OSError):
                self._wrapper_path.unlink(missing_ok=True)
            self._wrapper_path = None
        self._current_trial_id = None

    def _fail(self, trial: TrialInput, kind: FailureKind, msg: str) -> TrialOutput:
        # Clip at 4000 chars (vs 500 in L1/L2) so the full vLLM startup
        # traceback fits in the trial JSON. The per-trial _vllm.err file
        # has the unclipped version for debugging.
        return TrialOutput(
            measurement=None,
            failure=FailureRecord(
                kind=kind,
                message=msg[:4000],
                trial_id=trial.trial_id,
                layer=self.layer_name,
            ),
        )


def _tcp_open(host: str, port: int, timeout_s: float) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout_s)
        try:
            s.connect((host, port))
        except (OSError, TimeoutError):
            return False
        return True


def _query_gpu_memory_gb(device_id: int = 0, timeout_s: float = 3.0) -> float | None:
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


__all__ = ["L3VllmKernelAdapter"]
