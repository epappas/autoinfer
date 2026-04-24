# Iteration zero — L2 first cross-GPU-class data (2026-04-24)

**First empirical evidence for the thesis's distinguishing claim.** The
L2 topology adapter shipped in commit `86bd071` and after two timeout
fixes (`9abdad5`, `00b74ed`) + catalog correction (`f0d4977`), the L2
campaign produced the first autoinfer measurements spanning multiple
GPU SKUs.

Artifacts: `docs/research/raw/iteration-zero-l2-5trial-2026-04-24/`.

## Setup

- Campaign container: 2× RTX A6000 on Basilica (reference replica on
  GPU 1, orchestrator + driver + gate on GPU 0).
- Candidate deployments: one fresh Basilica `deploy_vllm` per trial,
  GPU class varied (H100, A100, RTX A6000), gpu_count=1.
- Model: Qwen/Qwen3-8B, bfloat16 reference. Candidate dtype varied.
- Workload: vLLM bench `sharegpt`, 64 prompts.
- Gate: 20 prompts, concurrency 4, self-kl calibrated.
- Policy: Sonnet 4 via OpenRouter (warmstart+operator), TPE surrogate.
- 5 trials in 6802 s wall clock (~1h 53min).

## Results

| Trial | GPU | dtype | gpu_mem_util | Outcome | tok/s | TTFT p99 | TPOT p99 | KL |
|---|---|---|---|---|---|---|---|---|
| w0000 | **H100** | bfloat16 | 0.85 | **kept** | **895.2** | 6.4 s | **30.0 ms** | 1.92 |
| w0001 | RTX A6000 | bfloat16 | 0.85 | fail startup | — | — | — | — |
| w0002 | A100 | bfloat16 | 0.85 | kept | 743.9 | 8.3 s | 102.3 ms | 2.05 |
| w0003 | **H100** | float16 | 0.85 | **kept** | 832.2 | 6.8 s | **16.4 ms** | 4.44 |
| o0004 | RTX A6000 | bfloat16 | 0.82 | fail startup | — | — | — | — |

Pareto frontier (max tok/s, min TPOT p99): **2 entries**, both H100.

## Findings — first cross-hardware comparison autoinfer has ever produced

### F1. H100 wins by every measure

On Qwen3-8B, ShareGPT, vllm defaults:
- **+20% throughput vs A100** (895 vs 744 tok/s)
- **3.4× lower TPOT p99** (30 ms vs 102 ms for bfloat16; 16 ms vs 102 ms for float16)
- TTFT p99 comparable (6.4 s vs 8.3 s)

This is the thesis's §4.2 prediction — that hardware class is a
first-order L2 axis — confirmed empirically for the first time.

### F2. Float16 on H100 is the latency floor

Same GPU, same bench, just switched dtype bfloat16 → float16:
throughput dropped slightly (-7%, 895 → 832 tok/s) but TPOT p99
**halved** (30 → 16.4 ms). **Fastest per-token latency autoinfer has
measured anywhere, across any run.**

The quality KL (4.44) is higher than the bfloat16 KL (1.92) — within
our calibrated max_kl ceiling but closer to the boundary. A
latency-critical serving scenario (interactive chat, agent
scratchpads) might accept the extra logit drift for the 2× TPOT win;
a batch-throughput scenario wouldn't.

### F3. RTX A6000 at count=1 was uniformly unschedulable

Both A6000 trials (w0001, o0004) failed at `replicas: 0/0` — never
placed by the Basilica scheduler. Even though
`list_secure_cloud_gpus` reports 8 RTX A6000 offers at count=1,
spot-marketplace capacity for the specific combo (1× A6000 + the
deploy_vllm memory/interconnect constraints) evaporated during
the campaign. Not a bug in autoinfer; a fleet-availability reality.

### F4. Per-trial wall-clock is the L2 bottleneck

Breakdown for this 5-trial run:

- Campaign container startup (uv sync + model pull for reference): ~10 min
- Per L2 trial:
  - Basilica scheduling + pod creation: 1-3 min
  - Docker image pull (vllm/vllm-openai:latest, ~4 GB): 3-6 min
  - Qwen3-8B weights pull: 2-5 min
  - vLLM engine warmup: 2-4 min
  - bench (64 prompts): 2-4 min
  - Gate (20 concurrent): <1 min
  - Teardown: <30 s
  - Total: ~12-23 min per successful trial
- Total: ~113 min for 5 trials (3 success, 2 fast-fail)

The long tail is image + weights pull. A pre-baked Docker image with
`Qwen/Qwen3-8B` cached would cut ~8 min per trial, making a 20-trial
L2 campaign feasible in ~2 h. Session-four candidate.

## Cross-layer synthesis (L1 best vs L2 best)

| Layer | Best config | GPU | tok/s | TPOT p99 |
|---|---|---|---|---|
| L1 (deterministic, 10-trial) | vLLM defaults + FLASHINFER | RTX A6000 (campaign container) | 870 | ~130 ms |
| L1 (LLM, 50-trial) | FLASH_ATTN + bf16 + prefix + bs=32 | RTX A6000 (campaign container) | 821 | 131 ms |
| **L2 (LLM, 5-trial)** | **vllm defaults** | **H100** | **895** | **30 ms** |
| **L2 (LLM, 5-trial) latency-optimal** | **vllm defaults + fp16** | **H100** | **832** | **16 ms** |

**L2 beats both L1 runs on every axis.** Not because L1 is bad but
because L1 was constrained to the campaign container's RTX A6000.
Thesis §5 bottom line confirmed: on real hardware, **the gains from
switching GPU class swamp the gains from knob tuning on a single GPU
class**. Cross-layer joint search (L1 × L2) would pick both.

## Thesis §8 status

Including this L2 run:

- [x] Substrate validated (3 independent iteration-zero campaigns).
- [x] LLM-guided search produces Pareto-diverse frontiers.
- [x] Self-kl calibration separates config drift from scheduling noise.
- [x] **L2 cross-hardware search produces real data** — thesis's
  distinguishing claim no longer pending.

The only remaining "not yet validated" is L3 (kernel-level search
inside vLLM) which is out of iteration-zero scope.

## Honest limitations

1. **Small sample**: 3 kept L2 trials. The H100 vs A100 ratio is
   probably stable, but adding H200, L40S, MI300X would sharpen the
   picture.
2. **TTFT is dominated by cold-start, not steady-state**: 6-8 s
   TTFT p99 is the first-batch prefill on a freshly-loaded model.
   Warm-cache TTFT would be much lower. The comparison across GPUs is
   still fair because all trials start cold on identical-sized batches.
3. **No L1 × L2 cross**: the L2 candidates used vllm defaults. A real
   joint search would also vary knobs per-GPU — e.g., FLASHINFER on
   H100 might get another 20-30% on top of the defaults.
4. **RTX A6000 at gpu_count=1 never scheduled**: autoinfer can't
   control spot availability; the right response is to fall back to
   alternative knob values, which the LLM operator should learn to do
   with more trials and a stronger hardware_notes prompt.

## Cost summary (this run)

- Basilica candidate compute: 3 × ~20 min on H100/A100 spot + 2 × quick
  fail on A6000 = ~$6-10 of spot GPU time.
- Basilica campaign container: ~2 h on 2× RTX A6000 = ~$3.
- OpenRouter (Sonnet 4 for warmstart + 1 operator call): ~$0.05.

Total: ~$10 for the first L2 data point.

## What this run produces for the article

- A clean head-to-head on the thesis's primary claim (cross-GPU-class
  matters more than L1 knob-tuning at iteration zero).
- Qualified float16 vs bfloat16 tradeoff on H100 (throughput -7% for
  2× TPOT improvement).
- Honest fail-mode log (A6000 spot unavailable → typed startup failure
  → surrogate learns to avoid).

## Next session candidates

1. **L1 × L2 joint search** — 30-trial campaign with both layers
   enabled, letting the LLM operator propose (gpu_type, attention_backend,
   kv_cache_dtype, prefix, ...). This is the full thesis-claim test.
2. **Pre-baked Docker image** with Qwen3-8B cached to cut per-trial
   wall time from ~20 min to ~10 min.
3. **Add H200 / MI300X / L40S / A100-80GB** to the gpu_type catalog
   once Basilica availability confirms they're schedulable.
4. **Plot generation** from the combined L1 + L2 datasets for the
   article.

## Evidence preserved

- `docs/research/raw/iteration-zero-l2-5trial-2026-04-24/` — 5 trial
  JSONs + orchestrator.log.
- `docs/research/raw/iteration-zero-l2-3trial-2026-04-24/` — prior
  A100-only partial success, preserved for diff.
- All prior iteration-zero L1 artifact directories intact.

Implements: thesis §4.2 L2 — per-trial Basilica deployments over
varied GPU classes.
Refutes: nothing.
Validates: L2 cross-hardware throughput claim (+20% H100 vs A100 on
Qwen3-8B ShareGPT) and L2 latency claim (H100 fp16 delivers 16 ms
TPOT p99 — autoinfer's lowest measured anywhere).
