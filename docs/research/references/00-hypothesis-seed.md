# autoinfer — thesis (iteration zero, 2026-04-22)

## 1. Thesis

Inference-engine optimization for a fixed model on a target fleet can be
treated as a structured search problem whose **mutable surface** spans three
ordered layers — engine configuration, hardware topology, kernel
implementation — and whose **frozen evaluation** is a workload trace plus a
quality gate. An LLM-guided keep-discard loop over these layers, applied to
**vLLM** as the engine substrate and **Basilica** as the deployment substrate,
produces configurations that dominate hand-tuned defaults on at least one of
(tokens/s, p99 TPOT, peak HBM, effective context) with no measurable quality
regression. The largest gains sit where heterogeneity is irreducible — the
layer of search that existing tools (llm-d, SGLang, AutoKernel) each address
in isolation but none unify.

This is a *thesis* in the engineering sense: a claim we commit to, not a
position we're defending. The purpose of the document is to make the claim
falsifiable and to enumerate the evidence that would sustain or break it.

## 2. Substrate commitments

Committed on the basis of user constraints, not academic defensibility:

- **Engine substrate:** vLLM. Deployment target for the user's models; richest
  exposed knob surface of any production OSS inference engine; alexandria has
  mature coverage (`wiki/vllm/*`).
- **Deployment substrate:** Basilica. User is core contributor. Heterogeneous
  GPU fleet is where novelty and practical value coincide for decentralized
  serving.
- **Iteration-zero model anchor:** **Qwen3-8B** (primary) with
  Llama-3.1-8B-Instruct as fallback. Fits a single H100 FP16 with generous
  batch headroom; widely supported by vLLM; enough public benchmarks for
  cross-checking. Small model size is a deliberate choice to maximize
  iteration speed on the harness, not a limitation of the thesis.
- **Later-stage deployment target:** Covenant-72B on Basilica multi-node.
  Lifted to once the substrate and per-layer adapters are validated at 8B.
  Kept as the real-scale validator, not the development loop.
- **Reference substrate, not primary:** nano-vLLM retains value as a fast
  kernel-extraction rig for L3 iteration. It is *not* a valid substrate for
  L1/L2 search; discoveries there do not transfer (see C8 below).
- **Search-loop primitive:** autoresearch-rl's loop pattern (frozen/mutable
  boundary + keep-discard ledger) and `BasilicaTarget` adapter are reused.
  Per the evidence review (C6), the *policy* is not `LLMParamPolicy` alone —
  it is a hybrid LLM-warm-start + classical-surrogate stack.

## 3. Four-axis target

| Axis | Measured as | Primary failure mode |
|------|-------------|----------------------|
| tokens/sec | per-request TPOT, aggregate throughput, TTFT | latency SLO breach |
| memory | peak HBM, KV-cache footprint, paging fragmentation | OOM, eviction thrash |
| context length | max input tokens before TPOT cliff | attention quadratic wall |
| quality | task score, logit divergence vs FP reference, batch invariance | silent regression |

No dollar axis in the ledger. Time-to-compute is the objective at every
layer; `\$/token` is a post-hoc derivation outside the search loop, computed
only when someone needs it, using an externally-supplied `(gpu_class, \$/hr)`
table. The thesis does not commit to a dollar oracle — throughput (tokens
per GPU-second) and memory carry all the hardware-economic signal we need
in-loop.

## 4. The three-layer model

### 4.1 L1 — Engine-config search (vLLM as-is)

No source edits. Search over `EngineArgs` and runtime env.

| Cluster | Knobs | Coupled axis |
|---|---|---|
| Scheduling | `max-num-batched-tokens`, `max-num-seqs`, `max-model-len`, `enable-chunked-prefill` + chunk size, `scheduling-policy`, `preemption-mode`, `num-scheduler-steps` | TTFT ↔ throughput ↔ p99 TPOT |
| KV cache | `kv-cache-dtype`, `block-size`, `enable-prefix-caching`, `swap-space`, `num-gpu-blocks-override`, `gpu-memory-utilization` | memory ↔ throughput ↔ quality |
| Attention backend | `VLLM_ATTENTION_BACKEND` (FlashAttention / FlashInfer / FlashMLA / TRT-LLM / Triton-unified / XFormers), FP8 query/output dtypes | kernel selection ↔ quality (batch invariance) |
| Parallelism | `tensor-parallel-size`, `pipeline-parallel-size`, `enable-expert-parallel`, `data-parallel-size` | comm cost ↔ memory ↔ batch |
| Quantization | `quantization` (awq/gptq/fp8/bnb/int4/marlin), `load-format`, activation quant | memory ↔ quality ↔ throughput |
| Speculative decoding | draft model, `num-speculative-tokens`, MTP/EAGLE/PARD/Medusa, acceptance method | tokens/s ↔ quality ↔ p99 variance |
| Compile / graphs | `compilation-config`, cudagraph capture sizes, fusion passes | warmup cost ↔ steady-state speed |
| Offload | `cpu-offload-gb`, `swap-space`, weight streaming | memory budget ↔ TPOT tail |

~30 axes under hard (compat matrix) and soft (coupled-triple) constraints.
The policy must see the constraints or waste iterations on infeasible configs.

**Harness.** `vllm bench serve` (or guidellm) replaying a real trace; quality
gate = logit divergence vs FP16 reference on ≤500 prompts + batch-invariance
check; watchdog for OOM / hang / NCCL faults.

**Policy.** Reuse `LLMParamPolicy` with a vLLM knob schema + compat constraints
as context. Hybrid mode: param search until stall; "diff proposals" at L1 are
feature-flag flips (e.g. switch attention backend), not source edits.

### 4.2 L2 — Hardware pairing and topology search (Basilica)

| Axis | What varies | Why it matters |
|---|---|---|
| Target GPU class | H100 / H200 / A100 / 4090 / MI300X / MI250X / Gaudi 2-3 / TPU v6e-v7 | 10× \$ variance; kernel availability per platform |
| Count | 1, 2, 4, 8, 16+ | model fit × batch capacity |
| Parallelism layout | TP × PP × EP × DP product ≤ total GPUs | TP bandwidth-bound at low batch; PP bubble; EP MoE-only |
| PD-disaggregation | collocated / split / heterogeneous-per-phase | prefill=compute-bound, decode=bandwidth-bound |
| KV-cache transfer | NVLink / RDMA / IB / RoCE / TCP (NixlConnector) | dominates TTFT in disagg |
| Asymmetric quantization | prefill FP16, decode FP8/AWQ | degrade at the memory bottleneck |
| Replicas | per-role scaling | throughput at cost |

**Primary objective at L2:** tokens per GPU-second at quality. Dollar cost
is derived post-hoc from an externally-supplied `\$/hr` table if the caller
needs a currency figure; the ledger itself never touches currency.

**Harness.** vLLM with Ray/torch.distributed on Basilica for homogeneous
multi-node; llm-d on top when PD-disagg + routing matter. Extended metrics:
inter-node KV GB/s, straggler tail, failover time under simulated node drop.
IRO (llm-d Inference Resilience Operator) as the fault-recovery building
block.

**Policy.** Outer policy picks (GPU class, count, TP×PP×EP). Inner policy is
L1 inside the chosen topology, warm-started from prior L1 runs on matching
hardware.

### 4.3 L3 — Kernel-level search inside vLLM

Mutable source. Replace or extend vLLM custom ops using AutoKernel-family
agents. Last by ROI, not by importance.

| Kernel family | Why a candidate | Constraint |
|---|---|---|
| PagedAttention (prefill/decode) | Dominant on long context; shape-polymorphic | Numerics, batch invariance |
| Fused MoE dispatch | Known pain; big wins on DeepSeek / Mixtral | Sparse dispatch patterns |
| Quantized matmul (Marlin / CUTLASS / Triton) | Hot per-token on quantized | Dtype compat |
| RMSNorm / RoPE / activation | 5-15% combined; fusable | Compose with compile fusion |
| Custom positional / sparse attn | Model-specific (Mamba-2, MLA, sliding) | Tight model coupling |
| Novel dtype kernels | NVFP4, MX, 1.58-bit | Entirely new paths |

**Where L3 pays.** Novel hardware (Blackwell FP4, MI300X MoE, Gaudi graph
fusion) where vLLM is less tuned; shape regions vLLM leaves on the table
(batch=1, very-long-context sparse); joint fusions beyond the compile pass.

**Harness.** Extract kernel → isolated correctness + perf rig (nano-vLLM or
pytest) for fast agent iteration → promote to vLLM → rerun L1 harness to
verify no regression across previously-measured configs.

**Policy.** AutoKernel, KernelAgent, or KernelFalcon — drop-in candidate based
on interface ergonomics.

## 5. Cross-layer composition

Autoinfer is a **three-layer system from day one**, not a staged rollout of
single-layer projects. L1 / L2 / L3 are *adapters* onto a shared substrate;
they come online in parallel once the substrate exists, not in sequence.

Layers are **not independent**. A kernel win at L3 shifts which L1 configs
win; a new GPU class at L2 invalidates L1 configs that depended on a specific
attention backend. Autoinfer therefore treats a finding at layer *N* as a
**stale signal** at layers above *N*: cached results auto-flagged for
re-evaluation by the cross-layer scheduler, never blindly reused.

The **cross-layer scheduler** allocates evaluation budget across layers based
on marginal return. When L1 saturates, budget flows to L3 in the gap regions
named by C5; when L2 opens a new GPU class, the scheduler re-validates L1's
cached configs there before advancing. This scheduler — not a single-layer
search algorithm — is the autoinfer contribution. No public system today
routes budget across (engine, topology, kernel) with a shared quality gate
and a shared stale-signal ledger; that is the gap the three-layer composition
closes.

Sequential single-layer ROI is a *construction detail* (§7, P5: the first
slice is L1-only to exercise the shared substrate end-to-end), not the
project's frame.

## 6. Claims requiring evidence

Each claim below is a load-bearing statement. Preliminary evidence gathered in
`docs/research/raw/references-*.md`. Open to refutation.

- **C1.** The vLLM `EngineArgs` + runtime-env surface is large enough that hand
  tuning leaves measurable throughput/latency/cost on the table at realistic
  scales.
- **C2.** Speculative decoding (EAGLE / Medusa / MTP / PARD / draft-model
  variants) produces non-trivial quality-variance and p99 latency-variance
  tradeoffs that mean-throughput benchmarks miss.
- **C3.** Quantization (weight and KV-cache) can silently drift logits. Must
  be gated by logit divergence + batch invariance, not task-level eval alone.
- **C4.** Prefill/decode disaggregation across heterogeneous compute
  Pareto-dominates homogeneous collocated serving on tokens/\$ at production
  scale.
- **C5 (revised).** LLM-driven kernel search reliably beats naive PyTorch
  baselines (KernelBench-scale, 10–17× on L1/L2/L3 after agentic scaffolding)
  but beats *production-tuned* kernels modestly — ~1.3–2× on average
  (Astra = 1.32× vs SGLang; AutoKernel = 2.83–3.44× vs torch.compile max-
  autotune; per-shape wins larger only in under-optimized regions). No public
  study targets vLLM's custom ops directly. The claim autoinfer can realistically
  defend: **LLM kernel search wins measurably on specific vLLM gap regions
  (Blackwell NVFP4 MoE, MI300X fused MoE, MXFP4 on GB10, batch=1 decode)
  where config registries and schedule templates are incomplete**; elsewhere
  it matches classical autotune (Triton autotune, CUTLASS+nvMatmulHeuristics).
- **C6 (revised).** A **hybrid** policy — LLM warm-start + proposal layer on
  top of a classical surrogate (TPE, CMA-ES, or BOHB) — dominates both pure
  LLM-guided and pure BO baselines on ML-serving config spaces at ~100–500
  evaluation budgets. Pure LLM-only search is *not* the defensible choice at
  this budget; see evidence in `../raw/references-search-policy.md` (esp.
  arxiv 2603.24647 — CMA-ES and TPE beat LLM agents on autoresearch-style
  systems tuning).
- **C7 (revised).** Joint search is supported in well-scoped pairs (schedule
  × layout in Ansor; resource × parallelism per phase in DistServe); full-
  stack "kernel × engine × topology" joint search has **no published
  evidence** yet. The better-supported alternative for small budgets is
  **multi-fidelity** search (Hyperband / BOHB) over layered decoupled
  optimization. Autoinfer should treat layered sequential ROI as primary and
  joint search as a later-stage research claim.
- **C8.** nano-vLLM and related minimal rigs are valid *kernel-iteration*
  harnesses but invalid *L1/L2 substrates*: discoveries on them do not
  transfer to production vLLM behavior.
- **C9.** Batch-invariance violations and FP-rounding drift in heterogeneous
  pools make "reference value" quality checks insufficient; a reference
  *replica* is required.

## 7. Design principles

These are the invariants the project commits to. Every PR, harness, and
adapter is measured against them.

- **P1 — Three layers from day one.** L1, L2, L3 are all in scope at
  iteration zero. No layer is "phase 2." The project name is autoinfer, not
  autotune-vllm-flags.
- **P2 — Shared substrate before any layer.** The workload driver, quality
  gate, reference replica, keep-discard ledger, and policy stack must exist
  before layer adapters ship. A layer that has its own bespoke harness is a
  bug.
- **P3 — Layers are adapters, not subprojects.** Each layer exposes
  `(surface, mutate, run, measure)` against the shared substrate. Adding
  a layer ≈ adding an adapter module, not forking the codebase.
- **P4 — Cross-layer stale-signal invalidation.** A finding at layer *N*
  publishes stale flags at layers above. The scheduler does not blindly
  reuse cached results after an invalidating change.
- **P5 — First slice validates the substrate.** Iteration zero is a
  single-layer slice (L1 on Qwen3-8B on one H100) whose goal is proving the
  shared substrate end-to-end, *not* producing a benchmark number. The
  number is a side effect; the harness validation is the point.
- **P6 — Small model first, real model after.** Qwen3-8B is the development
  loop. Covenant-72B is the validator. Do not mix them.
- **P7 — Hybrid policy by default.** LLM warm-start + TPE/CMA-ES surrogate
  + multi-fidelity scheduler + LLM proposal operator on stall. Pure
  LLM-guided is ruled out by C6; pure BO is ruled out by C7's failure-
  region evidence.
- **P8 — Reference replica for quality, not cached values.** Every run is
  gated by a live FP16 reference replica (C9). No static logit hashes.
- **P9 — Failure is a first-class signal.** OOM / hang / quality-fail are
  recorded, typed, and fed to the surrogate. Failure regions are not
  zero-reward holes — they are geometry the search must learn.
- **P10 — Frozen / mutable boundary, inherited from autoresearch-rl.**
  The evaluation harness (driver + gate + ledger) is frozen per run.
  Layer adapters are mutable. Policy proposes changes to the mutable
  side, never to the frozen side. Same trust boundary as autoresearch-rl.
- **P11 — Typed, modular, SOLID.** Pydantic config at boundaries, typed
  protocols for adapters, no god-objects. Functions ≤50 LoC by default.
  Assert-and-fail-fast over deep if-else trees.
- **P12 — Evidence-driven, falsifiable.** Every claim is numbered
  (C1–C9). Every experiment produces evidence for or against a claim.
  Thesis updates land when evidence does.

## 8. First experiment (substrate validation via L1 slice)

Iteration zero. Scope: prove the shared substrate end-to-end by exercising
it through the L1 adapter. Not scoped to "find a good config" — scoped to
"every component of the substrate produced a trustworthy signal on at least
one run."

**Model.** Qwen3-8B (primary), Llama-3.1-8B-Instruct (fallback if vLLM
version skew surfaces).

**Hardware.** Single Basilica H100 80GB node.

**Frozen (substrate):**
- **Workload driver** — replayed trace (ShareGPT-derived + bursty injected)
  via `vllm bench serve`.
- **Quality gate** — live FP16 reference replica of the same model; logit
  divergence + batch-invariance check at batch ∈ {1, 8, 64}; reject on
  KL above a pre-registered threshold.
- **Keep-discard ledger** — Pareto frontier across (tokens/s, p99 TPOT,
  KL-divergence, peak HBM).
- **Failure typer** — OOM / hang / NCCL / quality-fail as labeled
  outcomes fed to the surrogate.

**Mutable (L1 adapter):**
- Search space = L1 scheduling cluster + KV-cache cluster + attention
  backend + chunked prefill + speculative-decode off/on (EAGLE-3 if
  available). ~14 axes. Knob-compat matrix supplied to policy.

**Policy (per P7).**
- LLM warm-start: 10–20 configs seeded from Red Hat / GCP / ROCm tuning
  guides + C1 evidence.
- Handoff to Optuna TPE (or CMA-ES); LLM proposal operator every N=10
  trials.
- Multi-fidelity via Hyperband: 100-prompt smoke → promote top-N to
  500-prompt gated eval.

**Budget.** ~200 configs, ~48h wall clock.

**Success criterion (revised per P5).**
- **Primary:** every shared-substrate component produced a trustworthy
  signal. Quality gate caught ≥1 known-bad config injected as a canary.
  Reference replica stayed within ULP bounds. Ledger Pareto frontier is
  non-degenerate.
- **Secondary:** any Pareto-dominant config over vLLM defaults on at
  least one axis with no quality regression.
- A pure secondary-success without primary-success is a *failure*
  (we got a number we cannot trust).

## 9. Project structure

Mirrors `../autoresearch-rl` and `../llmwiki/alexandria` conventions. uv
+ setuptools, src-layout, pydantic config, typer CLI, pytest.

```
autoinfer/
├── CLAUDE.md                 # agent guidance (build, test, architecture)
├── README.md                 # thesis pointer + quickstart
├── pyproject.toml            # uv-managed; name=autoinfer
├── uv.lock
├── docs/
│   ├── ARCHITECTURE.md       # substrate + adapter contracts
│   └── research/
│       ├── raw/              # per-source notes (immutable after write)
│       └── references/       # compiled references + thesis
├── examples/
│   ├── qwen3-8b-l1-slice/    # iteration-zero config + run.sh + program.md
│   ├── qwen3-8b-l2-homogeneous/
│   └── qwen3-8b-l3-kernel-gap/
├── scripts/                  # one-off utilities, trace fetching
├── src/
│   └── autoinfer/
│       ├── __init__.py
│       ├── cli.py            # typer entry: run / validate / print-config
│       ├── config.py         # pydantic RunConfig, Layer schemas
│       ├── controller/
│       │   ├── continuous.py # outer loop: schedule → run → ledger
│       │   └── stale.py      # cross-layer stale-signal invalidation
│       ├── harness/          # SHARED SUBSTRATE (P2)
│       │   ├── driver.py     # vllm bench serve / guidellm wrapper
│       │   ├── gate.py       # logit divergence + batch invariance
│       │   ├── replica.py    # FP16 reference replica lifecycle
│       │   ├── ledger.py     # keep-discard + Pareto frontier
│       │   └── failure.py    # typed failure outcomes (OOM/hang/quality)
│       ├── policy/           # HYBRID STACK (P7)
│       │   ├── warmstart.py  # LLM initial design
│       │   ├── surrogate.py  # Optuna TPE / CMA-ES wrapper
│       │   ├── fidelity.py   # Hyperband/BOHB scheduler
│       │   └── operator.py   # LLM proposal operator (code/flag flips)
│       ├── layers/           # PER-LAYER ADAPTERS (P3)
│       │   ├── l1_engine/
│       │   │   ├── surface.py      # EngineArgs schema + compat matrix
│       │   │   ├── adapter.py      # vLLM process target
│       │   │   └── knobs.yaml
│       │   ├── l2_topology/
│       │   │   ├── surface.py      # (gpu_class, count, TP×PP×EP, PD)
│       │   │   └── adapter.py      # Basilica deploy target
│       │   └── l3_kernel/
│       │       ├── surface.py      # vLLM custom-op registry
│       │       ├── adapter.py      # extract / mutate / promote
│       │       └── sandbox/        # isolated correctness+perf rig
│       ├── target/           # deployment substrates
│       │   ├── local.py
│       │   └── basilica.py   # reuse autoresearch-rl pattern
│       └── telemetry/
│           ├── run.py        # per-run artifact writer
│           └── trace.py      # workload trace capture
├── tests/
│   ├── test_config.py
│   ├── test_harness_gate.py
│   ├── test_ledger.py
│   ├── test_stale_signal.py
│   └── test_layers_l1_adapter.py
└── traces/                   # workload traces (gitignored if large)
```

**Module responsibilities (contract form):**
- `harness.*` — frozen per run; changes require a config version bump.
- `layers.<L>.adapter` — implements the adapter protocol:
  `surface() → Schema`, `mutate(Config) → Deployment`,
  `run(Deployment, Trace) → Measurement`, `teardown()`.
- `policy.*` — stateless over a run ledger; proposes next config given
  history + stale-flags.
- `controller.continuous` — single outer loop, scheduler-driven; does not
  contain search logic.
- `controller.stale` — the cross-layer scheduler (§5); owns the only
  "which layer runs next" decision.

**Package naming.** `autoinfer` (single word, matches autoresearch-rl
convention). CLI entry: `autoinfer run <config.yaml>`.

## 10. Open questions (for iteration 1)

1. Which workload trace for the Qwen3-8B slice — ShareGPT-replay,
   LMSYS-Chat-1M sample, synthetic bursty, or a mix? This affects whether
   L1 findings transfer to production workload classes.
2. Reference replica sizing — is 1 replica of Qwen3-8B on a separate H100
   acceptable, or do we collocate on CPU / quant-FP8 for cost during
   iteration zero?
3. Cost oracle for tokens/\$ — hardcoded Basilica prices, live from the
   scheduler, or derived from the Basilica billing API?
4. Is L2 gated on llm-d maturity on Basilica, or do we build a thinner
   PD-disagg harness directly on vLLM + Ray?
5. When does Covenant-72B enter the loop as validator — after L1 passes on
   Qwen3-8B, or only once L2 adapter is stable?
6. Which LLM backs the policy warm-start and proposal operator —
   Claude / GPT / Qwen / self-hosted? Latency and cost per proposal
   influence the operator-call cadence (N=10 default in §8).

## 11. Evidence status after preliminary review (2026-04-22)

Four research tracks ran in parallel; outputs in `../raw/references-*.md`. Net
effect on the thesis:

| Claim | Status | Key evidence | Action |
|-------|--------|--------------|--------|
| C1 vLLM surface has measurable slack | **Supported** | ROCm V1 tuning 2373→3774 tok/s; Red Hat / GCP / AMD / vLLM auto_tune benchmarks | Keep |
| C2 Speculative-decode hides variance | **Supported** | TurboSpec (arxiv 2406.14066) shows speculation can *increase* latency under continuous batching; EAGLE "lossless" is kernel-bit-exact assumption | Keep; add p99/distribution gate to harness |
| C3 Quantization needs logit gate | **Supported** | KL↔flip-rate ρ=0.981 (NeurIPS 2024 "Accuracy is Not All You Need"); FP32-or-Death shows 9% AIME delta | Keep; gate mandatory |
| C4 PD-disagg Pareto-dominates on cost | **Directionally supported; headline gap** | Splitwise 1.4× / DistServe 7.4× / Mooncake 59–498% — but **no public multi-vendor heterogeneous study** vs homogeneous H100 baseline | Keep; explicit gap-to-close = autoinfer contribution |
| C5 LLM kernels beat hand-tuned | **Weakened / scoped** | Against PyTorch: 10–17×. Against production: 1.3–2× (Astra vs SGLang). No vLLM-as-baseline study exists | Revised above — scope to vLLM gap regions only |
| C6 LLM search beats BO | **Partially falsified** | arxiv 2603.24647 (on autoresearch testbed): CMA-ES and TPE beat LLM agents; Centaur hybrid wins | Revised above — hybrid policy required |
| C7 Joint search > layered | **Weakened** | Supported for well-scoped pairs (Ansor, DistServe); no full-stack study exists; multi-fidelity is better-supported technique | Revised above — layered primary, joint later |
| C8 Minimal-rig transfer limits | **Supported** | nano-vLLM benchmark excludes every V1-distinguishing feature (chunked prefill, speculation, disagg) | Keep |
| C9 Reference replica required | **Supported** | Ada 13-bit vs H100 25-bit FP8 accum (arxiv 2512.07004); ROCm FP8 FNUZ; Thinking Machines 80/1000 unique outputs at temp=0 | Keep |

Three findings that reshape the plan:

1. **Policy choice shifted** from pure LLM-guided to LLM-warm-start + TPE/
   CMA-ES surrogate. The evidence for "LLM beats BO at 200 evals on systems
   spaces" is negative. This is the most important update — it changes the
   first-experiment policy (§8) and the `autoresearch-rl` policy we adapt.
2. **The unique contribution opportunity sharpened to C4**: no published
   multi-vendor heterogeneous PD-disagg study on tokens/\$ exists. Autoinfer +
   Basilica is uniquely placed to produce that evidence.
3. **C5 (kernel search) is defensible only as scoped gap-region work** —
   Blackwell NVFP4 MoE, MI300X fused MoE, MXFP4 on GB10, batch=1 decode.
   Not a generic "beat vLLM kernels" claim.

## 12. Ingestion list for alexandria

To close remaining cross-references before iteration 1:

1. arxiv 2603.24647 — "Can LLMs Beat Classical HPO" (falsifier for C6)
2. arxiv 2406.14066 — TurboSpec / SmartSpec (C2 primary)
3. arxiv 2407.09141 — "Accuracy is Not All You Need" (C3 primary, NeurIPS 2024)
4. arxiv 2506.09501 — "Give Me FP32 or Give Me Death?" (C9 primary)
5. arxiv 2311.18677 — Splitwise; 2401.09670 — DistServe; 2407.00079 — Mooncake (C4 triad)
6. arxiv 2502.10517 — KernelBench; 2509.07506 — Astra (C5)
7. arxiv 2403.02310 — Sarathi-Serve (C4 counter-evidence)
8. Thinking Machines "Defeating Nondeterminism" + LMSYS SGLang deterministic (C3, C9)
9. `vllm/benchmarks/auto_tune/README.md` (C1; already in alexandria as raw, promote to wiki)

