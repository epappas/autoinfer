# Kernel-into-vLLM integration — validated end-to-end (2026-04-26)

The load-bearing piece of the project. The user pushed back, hard, on
deferring kernel-into-vLLM as "future work" — it IS the project. This
write-up captures the path from user pushback to end-to-end validation
in three smoke runs, ~$15 of campaign budget.

## What the integration does

When an L3 trial fires with ``mode='vllm'`` in the config, the
``L3VllmKernelAdapter``:

1. Validates the candidate (target_op, dtype, shape_regime, source,
   entry_fn).
2. Compiles the candidate kernel via ``compile_candidate``.
3. Runs an atol/rtol correctness gate against the PyTorch reference.
4. Renders an ``InjectionPlan`` into a wrapper Python script that:
   - exec's the LLM-proposed kernel source in a torch+triton namespace
   - monkeypatches ``vllm.model_executor.layers.layernorm.RMSNorm.forward_cuda``
     (or ``activation.SiluAndMul.forward_cuda``) with an adapter that
     calls the LLM kernel
   - exec's ``vllm.entrypoints.cli.main`` to start ``vllm serve``
5. Spawns the wrapper as a subprocess; waits for the serving port.
6. Runs the L1 driver + reference-replica gate against the patched server.
7. Reports a ``Measurement`` with end-to-end ``tokens_per_sec`` (NOT
   isolated kernel ops/sec) — ``pareto_eligible=True``.

The candidate kernel runs at every model forward call. The serving
benchmark therefore measures the *real* impact of the kernel on
end-to-end token throughput — exactly what the user pushed for.

## The path to validation: three smoke runs

### Smoke 1 (22:55 UTC, 2026-04-25)
First launch with ``mode='vllm'``. All 3 L3 trials STARTUP-failed.
The trial JSON's failure message clipped at 500 chars mid-traceback,
hiding the real error. Saw "Engine core ini..." — vLLM v1's standard
"engine subprocess crashed" message — but the root cause was below the
clip. Wrapper subprocess stdout/stderr were piped to ``subprocess.PIPE``
and never drained or persisted, so the unclipped traceback was lost too.

### Smoke 2 (23:21 UTC, 2026-04-25), commit 39cd281
Persisted wrapper stdout/stderr to per-trial files
(``<trial_id>_vllm.{out,err}``) under ``result_dir`` so they ride along
with trial JSON artifacts. Added ``.out`` and ``.err`` to the in-container
HTTP server's listing + the orchestrator's regex so they fetch back to dev.
Bumped the failure-message clip to 4000 chars for the L3 vllm path.

This run still failed all 3 L3 trials, but now we could read the actual
traceback. Root cause:

```
File "/opt/conda/lib/python3.11/multiprocessing/spawn.py", line 297,
    in _fixup_main_from_path
        main_content = runpy.run_path(main_path, ...)
File "/tmp/autoinfer-l3-wrappers/l3_wrapper_a3716c46.py", line 125,
    in <module>
        raise SystemExit(_vllm_main())
```

vLLM v1 spawns its engine-core subprocess via ``multiprocessing.spawn``,
which on Linux RE-RUNS the parent's main script in each child. The
wrapper's body — ``patch_op + sys.argv + raise SystemExit(_vllm_main())``
— ran in EVERY spawned child. Each child therefore tried to start
another vLLM serve, which spawned another engine, which ran the wrapper,
infinite-recursing until the OS gave up. Classic Python multiprocessing-
with-spawn footgun.

### Smoke 3 (23:43 UTC, 2026-04-25), commit 78703c3
Wrapped the serve invocation in ``if __name__ == "__main__":``. Crucially,
``_patch_vllm_op()`` stays OUTSIDE the guard — the engine-core child
process loads vllm independently and DOES need the same forward_cuda
override applied to its module-cached classes (the patch must run in
both parent and child).

This run produced the first-ever real kernel-into-vLLM measurement:

```
l3_kernel_w0001 silu_mul/fp16/medium  silu_mul_fused_kernel
  → 425.4 tok/s, 160.6ms TPOT, kl=3.02, pareto_eligible=True
```

The wrapper's ``_vllm.out`` log confirmed:
``[autoinfer.l3.injector] patched vllm.model_executor.layers.activation.
SiluAndMul.forward_cuda with kernel 'silu_mul_fused_kernel'``

But ``l3_kernel_w0000`` (a Triton ``@triton.jit`` kernel) failed
correctness with "Pointer argument (at 0) cannot be accessed from Triton
(cpu tensor?)". The pre-vLLM correctness gate was running tensors on
CPU; Triton requires CUDA. Triton kernels were rejected before they
ever reached vLLM.

### Smoke 4 (00:47 UTC, 2026-04-26), commit 8924421
Made the correctness check CUDA-aware: when ``source_uses_triton(source)``
returns True and ``torch.cuda.is_available()``, lift inputs to GPU before
running both reference and candidate. When Triton + no CUDA (dev box),
defer correctness to in-vLLM execution where any kernel error surfaces
as a typed STARTUP failure. Pure-PyTorch kernels still take the CPU
path so dev workflows without a GPU work for non-Triton candidates.

This run produced **two** end-to-end L3 measurements with
``pareto_eligible=True``:

| Trial | Op | Dtype | Regime | tok/s | TPOT p99 | KL |
|---|---|---|---|---|---|---|
| `l3_kernel_w0000` | rmsnorm | bfloat16 | large | **867.4** | 127.1 ms | 1.98 |
| `l3_kernel_w0001` | silu_mul | float16 | medium | 408.8 | 123.7 ms | 2.06 |
| `l3_kernel_w0002` | rope | — | — | (correctly rejected — RoPE unsupported in v1 injector) |

Comparison within the same run, same campaign-container A100:

| Layer | Best trial | Tok/s | Notes |
|---|---|---|---|
| L1 (engine knobs) | w0001 (FLASHINFER+bf16+bs=32) | 730.2 | local A100 |
| L2 (topology) | w0000 (A100×2, bf16, gmu=0.85) | 929.7 | 2× A100 |
| **L3 (kernel)** | **w0000 (rmsnorm/bf16/large)** | **867.4** | 1× A100, kernel patched |

Per-GPU normalisation: L3 single-A100 gets 867 tok/s vs L2's 2×A100
averaging ~465 tok/s/GPU. The kernel path on a single GPU is competitive
with the topology path on two — though the smoke uses the LLM proposer's
default rmsnorm kernel (likely identical to vLLM's reference), so the
+19% over L1 is mostly other-confounders (TTFT cold-start, vllm default
config differences) rather than a real kernel improvement.

## What's now real

- **Architecture validated end-to-end.** The wrapper subprocess +
  monkeypatch + bench + gate path produces real ``Measurement`` entries
  with ``pareto_eligible=True`` and ``tokens_per_sec`` in the same
  units as L1 and L2.
- **Triton kernels supported.** ``@triton.jit``-decorated source compiles
  via the temp-file import path and runs correctness checks on CUDA when
  available.
- **Correct multiprocessing semantics.** Engine-core subprocess spawn
  doesn't recurse. Patch flows to child processes via vllm's normal
  module-cache mechanism.
- **Diagnostic artifacts persisted.** Every L3-vllm trial leaves its
  wrapper subprocess's stdout/stderr under ``result_dir`` for post-mortem.

## What's not yet shown

- **Whether LLM-proposed kernels beat the reference at end-to-end.**
  The smoke had only 1-3 L3 trials per run, all using the same warmstart
  seed (the reference kernel itself or a near-clone). To prove
  ``L3 with novel kernel > L3 with reference > L1``, we need a longer
  run with more LLM-proposed variants and the failure-aware surrogate
  exploring the kernel space. The full L1×L2×L3 campaign with
  ``mode='vllm'`` does that — next session.
- **RoPE-style ops.** v1 injector supports rmsnorm and silu_mul; RoPE
  is omitted because vLLM's ``RotaryEmbedding`` API has many shape-
  varying paths and a single kernel signature doesn't capture them.
  v2 injector work.
- **Kernel-vs-reference ablations.** A side-by-side measurement at
  identical config except for the kernel would isolate kernel impact
  from L1-knob noise. The campaign infrastructure supports this; just
  needs a config variant.

## Cost

- Campaign container: 4 smoke launches × ~10 min each = ~$3
- L1/L2/L3 trials: 3 × ~15-25 min = ~$5
- Total this session: ~$8 + the 3 prior smokes ≈ ~$15-20

## What this means for the project

The kernel-into-vLLM integration is the deliverable the user
identified as "the whole point" of the project. It's now real code
that compiles, runs, produces validated end-to-end measurements, and
ships kernels into vLLM's actual serving path — not isolated kernel
benchmarks that don't translate to real serving improvement.

The next campaign at ``mode='vllm'`` with the failure-aware surrogate
+ LLM kernel proposer + reserve-on-stale will tell us whether
LLM-proposed kernels can produce a measurable serving improvement at
all. If yes, the third thesis claim has its first quantitative
empirical support. If no, we know the proposer needs a richer prompt
or the surrogate needs better feedback signals.

## Evidence

- ``docs/research/raw/smoke_l3vllm-2026-04-26/`` — first-ever end-to-end
  measurement (smoke 3, 425.4 tok/s with silu_mul_fused_kernel patched).
- ``docs/research/raw/smoke_l3vllm_validated-2026-04-26/`` — both
  rmsnorm and silu_mul kernels measured (smoke 4, two pareto_eligible
  L3 entries).
- ``src/autoinfer/layers/l3_kernel/{injector,vllm_adapter}.py`` — the
  shipped integration.
- Commits: de04edc (initial integration), 39cd281 (log persistence),
  78703c3 (multiprocessing guard), 8924421 (Triton CUDA correctness).
