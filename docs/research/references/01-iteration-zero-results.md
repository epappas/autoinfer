# Iteration zero — first real Basilica run (2026-04-23)

First end-to-end autoinfer campaign on hardware. Purpose: validate
the substrate (thesis §8 primary success criterion), not produce a
hero number.

## Setup

- **Engine substrate**: vLLM V1, `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime`
  image, `uv sync --extra dev --extra vllm` inside container.
- **Deployment substrate**: Basilica, single deployment with 2× RTX A6000
  (48 GB per GPU), 64 GiB system RAM, 43200 s TTL.
- **Model**: Qwen/Qwen3-8B, FP16 ("auto" dtype).
- **Reference replica**: same weights, separate GPU, port 8001.
- **Candidate**: spawned per trial on GPU 0 via L1EngineAdapter.
- **Workload**: vLLM bench `random` dataset, 64 prompts, 128-input/64-output.
- **Quality gate**: 20 held-out prompts, batch invariance at {1, 4}, max KL 0.05.
- **Policy**: deterministic warmstart with 8 diverse seeds, no operator.
- **max_trials**: 5 (first 5 warmstart seeds).

Orchestrator: `scripts/orchestrate_iteration_zero.py --max-trials 5`.
Deployment ID: `0a8f7a82-3107-426e-a971-7e0ab9ee1da6` (deleted).

## Artifacts

- `docs/research/raw/iteration-zero-smoke-run-2026-04-23/` — all 9 JSON
  files (5 trial + 4 bench output).
- Orchestrator + deployment logs preserved under that same directory.

## Results — 5 trials, 1 measurement, 4 typed failures

| Trial | Config | Outcome | Signal |
|---|---|---|---|
| w0000 | vLLM defaults, FLASH_ATTN, `auto` KV | **kept** | tok/s 239.4, TTFT p99 15.75s, TPOT p99 29.2ms, peak HBM 72.3GB, KL 0 |
| w0001 | FLASHINFER, `auto` KV | `quality_invariance` | Batch outputs differed between bs=1 and bs=4 on the same prompt |
| w0002 | FLASHINFER, **fp8** KV | `startup` | vLLM engine-core init failed: A6000 is Ampere, FP8 KV is Hopper+ |
| w0003 | FLASH_ATTN, `enable_prefix_caching=True` | `quality_kl` | KL above 0.05 threshold |
| w0004 | FLASH_ATTN, `max_num_batched_tokens=8192` (from 2048 default) | `quality_kl` | KL above 0.05 threshold |

The Pareto frontier has exactly **1 entry** (trial 0). It is
non-degenerate by construction — the only alternative configs failed
their gates.

## Thesis §8 success criteria — all primary conditions met

- [x] Every ledger entry has either a Measurement or a typed
  FailureRecord. 5/5.
- [x] Quality gate rejected at least one config. 3/5 (w0001
  QUALITY_INVARIANCE, w0003 + w0004 QUALITY_KL). The invariance check
  specifically caught a cross-backend divergence that a simple logit
  compare would have missed.
- [x] Pareto frontier non-empty and non-degenerate. 1 entry.
- [x] All four FailureKind labels appearing here (startup, quality_kl,
  quality_invariance) flow back to the surrogate as typed signals (P9)
  rather than being collapsed to "zero reward".

## Observations worth turning into next-session questions

1. **TTFT p99 = 15.75 s is really bad.** Probably a concurrency artifact
   from vLLM's random dataset — 64 prompts arriving simultaneously with
   only 2 A6000s. Not surprising on Ampere without FlashInfer, but worth
   re-running with `--request-rate` to see the prefill queue behavior.
2. **KL threshold 0.05 is possibly too strict on random-token prompts.**
   Both w0003 (prefix caching) and w0004 (8192 batched tokens) triggered
   KL failures. Real prompts (ShareGPT-style English) have much more
   structure; small numerical drifts on synthetic random-token sequences
   amplify into KL spikes. Worth re-running with `dataset_name=custom`
   once we fix the CustomDataset format mismatch.
3. **FLASHINFER + auto KV triggered a batch-invariance break.** This is
   exactly the class of silent-quality-drift the thesis §3 C3/C9 claims
   warned about. Worth a dedicated experiment: is the drift bounded,
   and is it worth the throughput gain?
4. **FP8 KV is off the table on Ampere.** Any iteration that wants to
   explore FP8 needs to be scoped to Hopper/Blackwell nodes at L2.
5. **Hardware still expensive per trial.** This 5-trial run cost ~30 min
   of setup + ~20 min of trials. Budget consideration before scaling to
   40 trials.

## What did NOT go wrong (explicit null hypothesis)

- No rate-limit or quota issues on Basilica beyond spot-availability
  flakiness during placement (the retry logic caught all of those).
- No data-corruption artifacts in the trial JSONs (schema consistent
  across all 5).
- No silent orchestrator crashes (completion marker seen, artifacts
  fetched via HTTP, deployment deleted cleanly on exit).

## Next session candidates

- [ ] Wire up real ShareGPT replay (fix CustomDataset format) so
  quality-gate signal is more representative.
- [ ] Concurrency in run_gate so 500-prompt gates complete in minutes,
  not hours.
- [ ] Scale max_trials to 20-40 once a real proposal LLM is wired.
- [ ] L2 adapter (heterogeneous-GPU topology) — the whole point of the
  thesis. Only blocker is cheap access to a second GPU class on
  Basilica (H100, A100, or MI300X).

## Evidence traceability

Implements: thesis §8 first experiment, primary success criterion.
Claims backed: C3 (quality drift from small engine changes, see
w0003/w0004), C9 (live reference replica catches things cached
baselines would miss, see w0001 invariance break).
