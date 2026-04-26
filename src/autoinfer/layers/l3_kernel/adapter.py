"""L3 kernel-search adapter.

Executes an LLM-proposed kernel candidate against a per-op PyTorch
reference, gates correctness (elementwise atol/rtol) and perf (ops/sec
best-of-K), and returns a ``TrialOutput`` shaped like every other
adapter — ``Measurement`` or typed ``FailureRecord`` (P3, P9).

CPU-safe minimum viable: accepts Python source that operates on torch
tensors. No Triton, no GPU required. Triton-on-GPU is a follow-up
session — the adapter's contract doesn't change, only how ``source``
is compiled (Triton ``@triton.jit`` path through ``triton.compile``
instead of ``exec``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from autoinfer.harness.failure import FailureKind, FailureRecord
from autoinfer.harness.ledger import Measurement
from autoinfer.layers import TrialInput, TrialOutput
from autoinfer.layers.l3_kernel.baselines import REFERENCES, KernelFn
from autoinfer.layers.l3_kernel.surface import (
    REFERENCE_SOURCES,
    KernelCallable,
    KnobCatalog,
    compile_candidate,
    make_inputs,
    resolve_dtype,
    test_shapes,
    to_surrogate_surface,
)


@dataclass
class L3KernelAdapter:
    """Correctness + perf gate for LLM-proposed kernel candidates."""

    catalog: KnobCatalog
    layer_name: str = "l3_kernel"
    atol: float = 1e-3
    rtol: float = 1e-3
    perf_repeats: int = 5
    warmup_runs: int = 2
    perf_seed: int = 1_000_003

    def surface(self) -> dict[str, Any]:
        # Only the categorical knobs flow into the surrogate surface;
        # ``source`` / ``entry_fn`` come from the LLM proposer.
        return to_surrogate_surface(self.catalog)

    def run(self, trial: TrialInput) -> TrialOutput:
        cfg = trial.config
        missing = [k for k in ("target_op", "dtype", "shape_regime") if k not in cfg]
        if missing:
            return self._failure(trial, FailureKind.STARTUP, f"missing config keys: {missing}")

        target_op = str(cfg["target_op"])
        reference = REFERENCES.get(target_op)
        if reference is None:
            return self._failure(trial, FailureKind.STARTUP, f"unknown target_op {target_op!r}")

        try:
            dtype = resolve_dtype(str(cfg["dtype"]))
            shapes = test_shapes(target_op, str(cfg["shape_regime"]))
        except ValueError as e:
            return self._failure(trial, FailureKind.STARTUP, str(e))

        # Surrogate-only path lacks source/entry_fn (only LLM proposers
        # generate them). Fall back to the reference source for that op
        # so the CPU-only plumbing is exercisable; real kernel search
        # requires an LLM proposer to override these keys.
        source = cfg.get("source")
        entry_fn = cfg.get("entry_fn")
        if source is None or entry_fn is None:
            ref_entry, ref_source = REFERENCE_SOURCES[target_op]
            source = source if source is not None else ref_source
            entry_fn = entry_fn if entry_fn is not None else ref_entry

        try:
            candidate = compile_candidate(str(source), str(entry_fn))
        except (ValueError, KeyError) as e:
            return self._failure(trial, FailureKind.STARTUP, f"compile failed: {e}")

        try:
            max_abs_err = self._check_correctness(candidate, reference, shapes, dtype)
        except Exception as e:  # noqa: BLE001 - candidate may raise anything
            return self._failure(trial, FailureKind.STARTUP, f"candidate raised during correctness: {e}")

        if max_abs_err is None:
            return self._failure(
                trial,
                FailureKind.QUALITY_KL,
                f"candidate exceeded atol={self.atol}/rtol={self.rtol}",
            )

        try:
            ops_per_sec = self._measure_perf(candidate, shapes, dtype)
        except Exception as e:  # noqa: BLE001
            return self._failure(trial, FailureKind.HANG, f"candidate raised during perf: {e}")

        # T-02: kernel-source provenance so post-run analysis can group
        # LLM-novel vs reference-fallback trials. Same metadata as the
        # vLLM adapter so analyzers don't branch on adapter type.
        import hashlib

        source_str = str(source) if source is not None else ""
        sha = hashlib.sha256(source_str.encode("utf-8")).hexdigest()[:12]
        ref_entry, ref_source = REFERENCE_SOURCES.get(target_op, ("", ""))
        ref_sha = hashlib.sha256(ref_source.encode("utf-8")).hexdigest()[:12] if ref_source else ""
        extra: dict[str, float] = {
            "max_abs_err": max_abs_err,
            "n_shapes": float(len(shapes)),
            "kernel_source_sha_int": float(int(sha, 16)),
            "kernel_is_reference": 1.0 if sha == ref_sha else 0.0,
        }
        meas = Measurement(
            tokens_per_sec=ops_per_sec,
            ttft_p99_ms=0.0,
            tpot_p99_ms=0.0,
            peak_hbm_gb=0.0,
            kl_divergence=0.0,
            extra=extra,
        )
        # Kernel ops/sec is not unit-comparable to L1/L2 end-to-end token
        # throughput. Mark ineligible for the joint Pareto until proper
        # vLLM custom-op integration measures kernel wins as token
        # throughput improvements; per-layer ranking still works.
        return TrialOutput(measurement=meas, failure=None, pareto_eligible=False)

    def teardown(self) -> None:
        return None

    def _check_correctness(
        self,
        candidate: KernelCallable,
        reference: KernelFn,
        shapes: tuple[Any, ...],
        dtype: torch.dtype,
    ) -> float | None:
        """Return the max observed abs-err if all inputs pass; else None."""
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

    def _measure_perf(
        self,
        candidate: KernelCallable,
        shapes: tuple[Any, ...],
        dtype: torch.dtype,
    ) -> float:
        """Best-of-K ops/sec at the largest shape in the regime.

        On CUDA we time with ``torch.cuda.Event`` and force sync between
        each iteration so asynchronous kernel launches don't return
        spurious sub-microsecond elapsed times. ``time.perf_counter``
        without sync would measure launch-overhead only — exactly the
        T-11 bug captured in TODO.md.

        On CPU we keep the wall-clock path (sync is a no-op there
        anyway and ``torch.cuda.Event`` requires a CUDA build at all).
        """
        spec = shapes[-1]
        inputs = make_inputs(spec, dtype, seed=self.perf_seed)
        use_cuda_timer = (
            torch.cuda.is_available()
            and any(
                isinstance(a, torch.Tensor) and a.is_cuda for a in inputs
            )
        )
        # Warmup. On CUDA, sync after warmup so the first measured run
        # doesn't pay JIT-compile overhead from the first warmup launch.
        for _ in range(self.warmup_runs):
            candidate(*inputs)
        if use_cuda_timer:
            torch.cuda.synchronize()

        if use_cuda_timer:
            best = self._best_elapsed_cuda(candidate, inputs)
        else:
            best = self._best_elapsed_wall(candidate, inputs)
        if best <= 0.0 or not (best < float("inf")):
            raise RuntimeError(f"perf measurement degenerate: best={best}")
        return 1.0 / best

    def _best_elapsed_wall(
        self, candidate: KernelCallable, inputs: tuple[Any, ...]
    ) -> float:
        best = float("inf")
        for _ in range(self.perf_repeats):
            t0 = time.perf_counter()
            candidate(*inputs)
            elapsed = time.perf_counter() - t0
            if elapsed < best:
                best = elapsed
        return best

    def _best_elapsed_cuda(
        self, candidate: KernelCallable, inputs: tuple[Any, ...]
    ) -> float:
        """CUDA-event timing with sync — see ``_measure_perf`` docstring."""
        # torch.cuda.Event's stubs are loose in current torch; cast to Any
        # to keep this function strictly-typed without per-call ignores.
        cuda_event_factory: Any = torch.cuda.Event
        best = float("inf")
        for _ in range(self.perf_repeats):
            start_evt = cuda_event_factory(enable_timing=True)
            end_evt = cuda_event_factory(enable_timing=True)
            start_evt.record()
            candidate(*inputs)
            end_evt.record()
            torch.cuda.synchronize()
            elapsed_ms: float = start_evt.elapsed_time(end_evt)
            elapsed = elapsed_ms / 1000.0
            if elapsed < best:
                best = elapsed
        return best

    def _failure(self, trial: TrialInput, kind: FailureKind, msg: str) -> TrialOutput:
        return TrialOutput(
            measurement=None,
            failure=FailureRecord(
                kind=kind,
                message=msg[:500],
                trial_id=trial.trial_id,
                layer=self.layer_name,
            ),
        )


__all__ = ["L3KernelAdapter"]
