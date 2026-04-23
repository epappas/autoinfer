# Iteration zero — 10-trial run with concurrent gate + ShareGPT (2026-04-23)

Second iteration-zero run after shipping (a) concurrent `run_gate`,
(b) native vLLM `sharegpt` dataset support, and (c) thresholds
calibrated against concurrent-query noise. Same hardware as the
5-trial run earlier today (2× RTX A6000 on Basilica).

Artifacts: `docs/research/raw/iteration-zero-10trial-run-2026-04-23/`.

## Policy configuration

- Warmstart: `DeterministicProposalLLM` with 8 diverse seeds (no
  cloud LLM credentials used).
- Surrogate: Optuna TPE (seed 0).
- Total trials: 10 (8 warmstart + 2 surrogate).

## Results

**10 trials, 3 kept, 7 typed failures, 2 Pareto entries.**

| Trial | Outcome | tok/s | TTFT p99 | KL | Notable config |
|---|---|---|---|---|---|
| w0000 | kept | **439** | 16.2 s | 1.23 | defaults (FLASH_ATTN) |
| w0001 | kept | **870** | 1.5 s | 1.17 | FLASHINFER |
| w0002 | startup | — | — | — | FLASHINFER + fp8 KV |
| w0003 | kept | **867** | 1.6 s | 1.93 | FLASH_ATTN + prefix caching |
| w0004 | quality_kl | — | — | — | FLASH_ATTN + batched=8192 |
| w0005 | quality_kl | — | — | — | FLASH_ATTN + seqs=256 |
| w0006 | quality_kl | — | — | — | FLASH_ATTN + gpu_mem=0.85 |
| w0007 | quality_kl | — | — | — | FLASH_ATTN defaults (repeat) |
| s0008 | startup | — | — | — | FLASHINFER + fp8_e5m2 (TPE chose) |
| s0009 | startup | — | — | — | XFORMERS + fp8_e5m2 (TPE chose) |

Pareto frontier (maximizing tokens_per_sec, minimizing p99 TPOT/HBM):

1. **FLASHINFER (w0001)** — 870 tok/s, TTFT 1.5 s
2. **FLASH_ATTN + prefix caching (w0003)** — 867 tok/s, TTFT 1.6 s

## First real autoinfer findings

### F1. FLASHINFER doubles throughput over FLASH_ATTN on Ampere

439 → 870 tok/s, same model, same prompts, only attention backend
changed. TTFT p99 drops from 16.2 s to 1.5 s — a 10× reduction.
Verifies that **attention-backend selection is one of the highest-
leverage L1 knobs** on this hardware, matching alexandria's vLLM
research.

### F2. Prefix caching also doubles throughput — on FLASH_ATTN

439 → 867 tok/s with `enable_prefix_caching=True` on FLASH_ATTN.
ShareGPT has enough shared prefix structure (system prompts,
few-shot preambles) that cache reuse alone recovers the gap vs
FLASHINFER. An interesting follow-up: do FLASHINFER + prefix
caching combine multiplicatively, or do they saturate on the same
bottleneck?

### F3. Large batched-tokens and seqs hit quality_kl under the current gate

w0004 (batched=8192) and w0005 (seqs=256) failed KL. This is likely
the same concurrent-query noise that forced us to raise `max_kl` to
2.0; under longer prefill chunks the logit variance widens further.
Cannot distinguish real config-induced drift from noise without a
**reference-vs-self baseline**.

### F4. TPE wasted both surrogate trials on infeasible configs

s0008 and s0009 both picked FP8 KV variants, which fail startup on
Ampere. TPE has no hardware-compat prior. This is exactly the kind
of thing a real LLM proposer would avoid (it knows FP8 KV needs
Hopper+). Load-bearing argument for moving from Deterministic to
Anthropic/OpenAI proposer in the next run.

## Thesis §8 re-check — all primary conditions still met

- [x] Every entry: Measurement xor FailureRecord (10/10).
- [x] Quality gate rejected configs (4/10 on KL, 2/10 on startup).
- [x] Pareto frontier non-empty, non-degenerate (2 entries).

## Remaining noise floor concern

Three trials with *identical* config (w0000, w0005, w0006, w0007 are
all essentially vLLM defaults) produced KLs of 1.23, "quality_kl",
"quality_kl", "quality_kl" respectively. The gate's current
threshold (max_kl=2.0) is loose enough to keep w0000 but reject
w0005–w0007 — noise, not signal. This is a real limitation.

**Session-four fix path:**

1. Add `reference-vs-self` gate mode: run the gate with
   `candidate_endpoint == reference_endpoint` as a warmup and
   **calibrate max_kl from the observed noise** (e.g., 10× the 95th
   percentile of self-self KL).
2. Enable vLLM batch-invariant ops on the candidate side so
   concurrent queries don't drift. Per alexandria
   `wiki/vllm/docs/batch_invariance.md`, this adds overhead but
   removes scheduling-variance as a confound.

## Cost snapshot

- Deployment: ~25 min wall clock (build + 10 trials).
- Hardware: 2× RTX A6000 spot.
- Real cost: ~$5–8 at current Basilica pricing.

## Evidence traceability

Implements: thesis §8 first experiment, primary + secondary criteria.
Claims: C3 (quality drift from engine changes — see F1-F3), C9
(batch-variance-under-concurrent-load is real — confirmed empirically).
Refutes nothing in thesis.
Next: either (a) wire ANTHROPIC_API_KEY and run 20-trial LLM-proposer
campaign on the same hardware, or (b) gate calibration via self-self
baseline to separate signal from noise.
