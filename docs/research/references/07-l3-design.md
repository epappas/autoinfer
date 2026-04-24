# L3 kernel-search design — minimum viable

Goal: implement an L3 layer that conforms to `LayerAdapter` Protocol,
measures kernel performance, and integrates with the existing
scheduler so cross-layer stale-signal can fire on L3 findings.

## Scope split

Full L3 (LLM-proposed Triton kernels + vLLM integration) is a multi-
session build. This doc scopes the **minimum** for autoinfer to claim
"three-layer search" honestly:

### In this session (CPU-safe, no Triton, no GPU)

- `L3KernelAdapter` conforming to `LayerAdapter` Protocol.
- A small catalog of target ops (RMSNorm, SiLU-mul, RoPE) with
  deterministic PyTorch reference implementations serving as both
  baseline and quality oracle.
- Kernel candidates delivered as Python source strings (Triton syntax
  expected at runtime; PyTorch fallback at test time).
- Adapter runs a correctness gate (max absolute / relative error vs
  reference on a small test matrix) and a perf gate (throughput in
  ops/sec on a known shape).
- Returns `Measurement(tokens_per_sec=ops_per_sec, ...)` when both
  gates pass, typed `FailureRecord` otherwise.

### Deferred to session-next (GPU required)

- Triton runtime compilation of LLM-proposed kernels.
- Integration with vLLM's custom-op registry (so an accepted L3
  kernel replaces vLLM's default for a specific op).
- Cross-layer stale actually AFFECTING L1 measurements (i.e., L3
  kernel replacement makes L1 re-search produce different numbers).

## The target-op catalog

Three ops, all stateless, all have exact reference implementations:

| Op | Inputs | Output | Why first |
|---|---|---|---|
| **rmsnorm** | `x: float[B, D]`, `w: float[D]`, `eps: float` | `float[B, D]` | Smallest/simplest; present in every transformer |
| **silu_mul** | `a: float[B, 2D]` | `float[B, D]` | Common MoE/gated-MLP bottleneck |
| **rope** | `q/k: float[B, H, D]`, `cos: float[T, D]`, `sin: float[T, D]` | `float[B, H, D]` | Attention-side, more complex indexing |

The PyTorch reference for each is the "correctness oracle" — a kernel
candidate must produce output within `atol/rtol` of the reference on
the test matrix to pass the gate.

## Search surface

Unlike L1/L2 which search numeric knobs, L3 searches **kernel source
code**. The config carries:

```
{
  "target_op": "rmsnorm",       # which op to replace
  "dtype": "float16",            # input dtype
  "shape_regime": "small",       # small / medium / large test shapes
  "source": "<Python/Triton source>",  # the candidate kernel
  "entry_fn": "rmsnorm_kernel",  # name of the callable within source
}
```

For the Optuna-surrogate path, the non-source fields are the L3
surface (target_op, dtype, shape_regime are categorical). Source
comes from the LLM proposer — the surrogate picks the op + regime,
the LLM generates source conditioned on that.

Warmstart seeds are a few reference kernels already shipped in
`layers/l3_kernel/baselines.py` to establish the "pass everything"
baseline.

## Quality gate vs perf gate

- Correctness: each of N test inputs → candidate output within
  `atol=1e-3, rtol=1e-3` of reference. Default N=8 across varied
  shapes and values (edge cases incl. zeros, large magnitudes, small
  magnitudes). Failure → `FailureKind.QUALITY_KL`.
- Perf: best-of-K timed runs of the candidate at a fixed target
  shape. Measurement is `tokens_per_sec = ops_per_sec_on_reference_shape`.

Rejecting a kernel that's correct-but-slow is fine — it simply
doesn't make the Pareto frontier. The stale-signal still fires only
when a kernel is accepted AND improves the frontier.

## Integration plan into the runner

`_build_l3_runner(cfg, max_trials_override)` in `builder.py` will
mirror L1 / L2 structure. Cross-layer wiring already exists in the
`ContinuousRunner`: a new Pareto entry at layer `l3_kernel` will
auto-fire `propagate_finding("l3_kernel", ledger)` which marks all
L1 and L2 entries stale.

## Tests (CPU-only, this session)

- `test_l3_surface.py` — catalog loading, deploy_op -> deploy_kwargs.
- `test_l3_adapter.py` — with a PyTorch reference passed as both
  candidate and reference, verify a "perfect kernel" passes. With a
  deliberately-wrong candidate, verify QUALITY_KL failure. With a
  candidate that raises, verify STARTUP failure.
- `test_l3_baselines.py` — every reference-kernel impl matches
  reference (identity check, basically a `pytest.approx` sanity).

No GPU / Triton required for this session's tests. Triton-on-GPU path
lands in session-next with a marker like `@pytest.mark.gpu`.

## What success looks like

At end of this session: L3 adapter is Protocol-conforming, has tests,
wires into builder, and the existing cross-layer stale-signal
triggers when an L3 entry joins the Pareto frontier of a multi-layer
run. Running a real L3 campaign requires a GPU + Triton, which we
defer.

The thesis-grade claim (joint L1×L2×L3 campaign with stale-signal
actually affecting L1 measurements) is still pending but the last
structural piece is in place.
