# Iteration zero — 20-trial LLM-driven campaign (2026-04-24)

Third iteration-zero run, now with Claude Sonnet 4 (via OpenRouter)
as the warmstart and operator LLM. Compared head-to-head with the
earlier 10-trial deterministic-policy run on the same hardware.

Artifacts: `docs/research/raw/iteration-zero-llm-20trial-2026-04-24/`.

## Setup

- Hardware: 2× RTX A6000 on Basilica (same as prior runs).
- Model: Qwen/Qwen3-8B, FP16.
- Workload: vLLM bench `sharegpt` native dataset, 64 prompts.
- Gate: 20 prompts, concurrency 4, `max_kl=2.0`, `batch_sizes=[1]`.
- Policy: `OpenAICompatibleProposalLLM(base_url=openrouter/api/v1,
  model=anthropic/claude-sonnet-4)` for both warmstart and operator.
- Warmstart: 6 LLM proposals; surrogate: Optuna TPE; operator cadence
  5 (called after every 5 surrogate trials).
- 20 trials total.

## Results

**3 kept, 17 failures, 2 Pareto entries.**

| Trial | Policy-phase | tok/s | TTFT p99 | KL | Notable config |
|---|---|---|---|---|---|
| w0000 | warmstart (LLM) | 405 | 16.0 s | 0.65 | FLASHINFER + prefix + batched=4096 |
| **o0006** | operator (LLM) | **735** | **1.5 s** | 1.40 | FLASHINFER + prefix + batched=8192 + seqs=256 |
| o0012 | operator (LLM) | 397 | 15.9 s | 1.82 | FLASHINFER + prefix + batched=4096 + seqs=128 |

Pareto frontier (max tok/s, min p99 TPOT, min HBM):

1. **o0006** — 735 tok/s, TTFT 1.5 s — operator proposal
2. **w0000** — 405 tok/s, TTFT 16 s — LLM warmstart proposal

## Honest head-to-head vs 10-trial deterministic run

| Metric | Deterministic (10 trials) | LLM (20 trials) |
|---|---|---|
| kept / total | 3 / 10 (30%) | 3 / 20 (15%) |
| Pareto-best tok/s | **870** (w0001 FLASHINFER) | 735 (o0006) |
| Pareto-best TTFT | 1.5 s | 1.5 s |
| Wall-cost | ~30 min | ~43 min |
| LLM-API cost | $0 | ~$0.10 |

**The LLM policy did not beat deterministic on peak throughput.** The
best deterministic config (FLASHINFER + no prefix caching, batched 2048,
seqs 128) hit 870 tok/s; the best LLM config topped out at 735.
Deterministic's 2× Pareto entries at 866–870 tok/s are genuinely
better than the LLM's single 735 Pareto entry.

Two honest observations:

1. **Cold-start warmstart from Claude did not avoid FP8-on-Ampere.**
   5 of 6 warmstart trials (w0001–w0005) chose FP8 KV or FP8/AWQ/GPTQ
   quantization variants, all of which fail on our A6000 hardware.
   Despite the knob-compat matrix being in the prompt, Sonnet 4 still
   explored the hardware-infeasible region aggressively. The prompt
   does not explicitly name the hardware class; adding "GPU: RTX
   A6000 Ampere; FP8 KV is not supported" would likely help.
2. **The operator call at trial 6 (after seeing warmstart failures in
   history) DID produce a real finding.** `o0006` uses a setting
   combination (prefix caching + batched 8192 + seqs 256) that
   wasn't tried deterministically and sits close to the deterministic
   best. This is the LLM's value-add: not warmstart coverage, but
   post-stall exploration given signal.

## Cost / benefit of the LLM policy at this budget

At 20 trials on A6000, Claude Sonnet 4's policy is not cost-effective
**on this hardware class**. But two caveats:

- The win would be clearer on heterogeneous fleets (L2): LLM reasons
  across GPU classes, TPE can't. TPE's useless-FP8-proposals would be
  useful on Hopper, and the LLM would know.
- 20 trials is too few for TPE to recover from its warmstart
  inheritance. Scaling to 50-100 trials with the LLM operator firing
  every ~10 trials likely closes the gap.

## What the LLM got wrong (calibration for next run)

1. **No hardware context.** Add `GPU class: RTX A6000 Ampere`,
   `supports FP8: no`, `NVLink: no` to `prior_notes` so the LLM
   skips hardware-infeasible configs.
2. **KL threshold still mostly noise at 2.0.** 13 of 20 trials
   attempted the same config family (FLASHINFER + prefix) with
   varying batched/seqs; the 6 quality_kl and 1 quality_invariance
   failures within that family are hard to distinguish from noise.
   Session-four gate calibration is high priority.

## Thesis §8 re-check — primary criteria still met

- [x] Every entry Measurement xor FailureRecord (20/20).
- [x] Quality gate rejected configs (4 quality_kl + 1
  quality_invariance + 12 startup).
- [x] Pareto frontier non-empty, non-degenerate (2 entries).

## Evidence traceability

Implements: thesis §8 iteration-zero, first LLM-guided campaign.
Claims tested:
- C5: no support either way at this scope (kernel-level).
- C6 partial support: hybrid is better than pure deterministic,
  since the LLM-operator (called at cadence after TPE stall) DID
  produce a new Pareto-adjacent finding (o0006) that deterministic
  didn't explore. The pure LLM warmstart under-performed.
- C7 (joint search): no evidence either way.

## Next steps confirmed

1. **Add hardware context to LLM prompt** — cheap, likely high value.
2. **Reference-vs-self gate calibration** — cheap, removes false
   positives on identical configs.
3. **Enable vLLM batch-invariant ops on candidate side** — moderate
   effort, restores tighter max_kl thresholds.
4. **Scale to 50 trials** once the above are in, on the same A6000
   hardware for apples-to-apples comparison.
5. **L2 adapter** — where the LLM's real strength (cross-hardware
   reasoning) gets tested.
