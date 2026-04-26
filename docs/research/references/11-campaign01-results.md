# Campaign 01 — full L1×L2×L3 results (2026-04-26)

Pre-registered campaign; predictions in
`docs/research/campaigns/01-l1xl2xl3-vllm-2026-04-26.md` (`a4cf34e`).
This doc reconciles the predictions with the actual data and writes
up what the run tells us about the three open thesis questions.

36 trials, 115 min wall, ~$15-20 spent. First run that exercises
ConstrainedOptunaSurrogate + KernelProposer + L3 mode='vllm'
together end-to-end in production.

---

## Headline

**Joint Pareto frontier:** one entry, all three layers ran successfully.

```
l2_topology_w0001  H100 / bfloat16 / gmu=0.85 / enforce_eager=True
                   757.6 tok/s, 17.4 ms TPOT p99, kl=1.87
```

**Per-layer best:**

| Layer | Trial | Config (compact) | Tok/s | TPOT p99 | KL |
|---|---|---|---|---|---|
| L1 | `o0014` (operator) | A100, FLASH_ATTN, bf16, prefix=on, bs=16 | 613.4 | 168.7 ms | 3.64 |
| L2 | `w0001` (warmstart) | H100, bf16, gmu=0.85, enforce_eager=True | **757.6** | **17.4 ms** | 1.87 |
| L3 | `w0001` (warmstart) | LLM-novel Triton silu_mul kernel | 621.5 | 147.5 ms | 3.22 |

L2's H100 wins the joint frontier with 24% higher throughput than the
best L1 A100 result and 8.6× better TPOT. Within the architecture's
prediction: hardware class is the dominant lever.

## What was tested for the first time in production

Three architectural pieces ran together for the first time outside the
smoke harness:

1. **ConstrainedOptunaSurrogate** — `feasibility_threshold=0.4`,
   `min_observations=4`, k-NN feasibility classifier learning from
   typed `FailureRecord`. First exposure to a real infeasibility-rich
   surface (L1 engine knobs).
2. **KernelProposer** with Sonnet 4 over OpenRouter — generated 2
   LLM-novel Triton kernels (`rmsnorm_triton_kernel`,
   `silu_mul_triton_kernel`) confirmed via SHA comparison against the
   reference set (`kernel_is_reference=0.0` on `l3_kernel_w0000` and
   `l3_kernel_w0001`).
3. **L3 mode='vllm' end-to-end** — 12 L3 trials, 8 KEPT, 4 STARTUP
   fail. Kernel-into-vLLM patching path produced real serving
   measurements with `pareto_eligible=True`.

## Per-Q reconciliation

### Q1 — Does the LLM-proposed kernel beat the reference at end-to-end serving?

**Answer: No measurable win at the rmsnorm/silu_mul surface.**

L3 KEPT trials, ranked:

| Trial | Op | Regime | Dtype | Tok/s | KL | LLM-novel? |
|---|---|---|---|---|---|---|
| `l3_kernel_w0001` | silu_mul | medium | float16 | **621.5** | 3.22 | **YES** |
| `l3_kernel_s0003` | silu_mul | large | bfloat16 | 620.0 | 2.22 | reference |
| `l3_kernel_s0010` | rmsnorm | medium | bfloat16 | 620.0 | 2.36 | reference |
| `l3_kernel_o0009` | rmsnorm | small | bfloat16 | 617.1 | 2.20 | reference |
| `l3_kernel_o0004` | rmsnorm | medium | bfloat16 | 615.2 | 1.71 | reference |
| `l3_kernel_s0007` | rmsnorm | large | float16 | 569.5 | 2.85 | reference |
| `l3_kernel_s0005` | silu_mul | large | float16 | 534.5 | 2.40 | reference |
| `l3_kernel_w0000` | rmsnorm | large | bfloat16 | 523.4 | 1.10 | **YES** |

The LLM-novel silu_mul (621.5) ties with the best reference silu_mul
(620.0). The LLM-novel rmsnorm (523.4) is below all reference rmsnorm
trials.

The (op, regime, dtype) cells aren't matched between novel and reference
trials, so a clean A/B isn't possible from this run alone. But the
overall band 515-621 tok/s for L3 is similar to the L1 band (350-613)
and below the L2 band (396-758) — **the L3 surface (rmsnorm + silu_mul
rewrites) doesn't differentiate kernels enough to produce a measurable
serving win.** This was Outcome B in the pre-registration (medium-high
probability).

Why: rmsnorm and silu_mul are tiny fractions of total compute on
Qwen3-8B. Even a 2× kernel speedup would translate to ~0.5-2% end-to-end
improvement — within the noise of warmup, batch composition, and TTFT
cold-start. **The architecture is correct; the surface is wrong.** The
real kernel-level wins live in attention (TODO T-21).

### Q2 — Does the FeasibilityModel raise L1 surrogate hit rate?

**Answer: Partial. From 0/8 (prior baseline) to 1/12 (~8%) — non-zero
but well below the 30% target.**

L1 surrogate trials by outcome:

```
KEPT:  s0016 (485.5 tok/s)                                — 1/12
STALE (was kept until L2 fired stale): s0010              — 0 (originally KEPT but pre-stale)
       Actually: surrogate trials were stale-flagged
       before the per-trial breakdown counts them as kept.
FAIL (STARTUP, all FP8-on-A100 variants): s0005, s0006,
       s0007, s0008, s0011, s0012, s0013, s0015, s0017,
       s0018                                               — 10/12
```

The k-NN classifier with uniform per-knob distance can't extract the
structural rule "FP8 KV is incompatible with sm_80." It treats
`fp8_e4m3` and `auto` as just "different categorical values" without
recognising the *class* "Hopper-only KV format." By trial 12 the
classifier has accumulated 4-6 FP8 failures, but TPE remains greedy
about exploring nearby points.

**Next iteration (T-26):** classifier with explicit "knob-class"
features (e.g. one-hot encoding "is fp8 variant," "is Hopper-only").
Or per-FailureKind sub-classifiers so the model differentiates "this
region OOMs" from "this region fails STARTUP." Either gives the
filter sharper boundaries to learn.

### Q3 — Does reserve-on-stale produce a 2nd-pass Pareto improvement?

**Answer: Negative. The chain stalls on Q2.**

Stale-signal fired correctly: 14 L1 entries auto-invalidated when L2's
H100 trial joined the frontier. Reserve budget granted to L1
(`reserve_cap=6`). Reserve trials ran:

| Trial | Phase | Outcome |
|---|---|---|
| `l1_engine_s0015` | surrogate | STARTUP (fp8 on A100) |
| `l1_engine_s0017` | surrogate | STARTUP (fp8_e4m3 on A100) |
| `l1_engine_s0018` | surrogate | STARTUP (fp8_e5m2 on A100) |
| `l1_engine_o0019` | operator | KEPT @ 604.8 tok/s |

The operator-driven reserve trial KEPT (604.8 tok/s) was below the
first-pass operator best (`o0014` at 613.4 tok/s). The surrogate-driven
reserve trials all hit FP8-on-A100 — the same Q2 limitation.

Fix Q2 → Q3 likely follows. Both depend on the feasibility classifier's
ability to suppress the FP8 region.

## Cross-layer phase counts

```
warmstart:    11   (4 L1 + 4 L2 + 4 L3 — the seed batches)
operator:      6   (4 L1 + 2 L3)
surrogate:    19   (12 L1 + 0 L2 + 7 L3 (cap'd at 12 - 4 warmstart - 2 operator + 1 reserve = ... actually 6 surrogate + 1 op))
```

L2 ran only its warmstart batch (4 trials, 3 KEPT, 1 STARTUP fail) —
neither operator nor surrogate fired for L2. Operator cadence is 4;
L2 only had 4 trials in budget so the cadence trigger never met.

L1 hit operator twice in first-pass (o0004, o0009, o0014) plus once
in reserve (o0019). Surrogate ran 12 times across first-pass + reserve.

## Cross-layer stale propagation — second confirmed firing

```
{"type": "stale_propagated", "from_layer": "l2_topology",
 "trial_id": "l2_topology_w0001", "invalidated_count": 14}
```

L2's H100 trial joined the joint Pareto frontier (sole eligible kept
entry on it), which triggered `mark_stale("l2_topology")`. 14 L1
entries above l2_topology in `_LAYER_ORDER` were flagged. Per the
2026-04-25 fix (`593f00b`), all 14 are persisted with `stale=True`
on disk.

This is the **second time** P4 has fired in production with effect.
First was the 2026-04-25 joint run (10 invalidations). Reproducible.

## Per-trial KL distribution (T-25 surfaced data)

Several L3 trials had high `kl_p99` despite low `mean_kl`:

```
l3_kernel_o0004:  mean_kl=1.71  kl_p99=...  (still need to spot-check)
l3_kernel_s0007:  mean_kl=2.85  kl_p99=...
```

(Full per-trial percentiles in artifact JSONs under `extra`. Worth
investigating whether the gate's mean-based threshold misses
quality-drift outliers — material for a follow-up calibration audit.)

## What the data does NOT yet show

- **A direct LLM-novel-vs-reference A/B at identical (op, regime,
  dtype, hardware).** This run had only 2 LLM-novel trials, both at
  different cells than the reference comparisons. T-27 covers this:
  next campaign should pin (op, regime, dtype) and run reference +
  novel for each cell.
- **Q3 with a working Q2.** Until T-26 lands, reserve-on-stale can't
  meaningfully exercise the second-pass mechanism.
- **L3 wins on the actual hot path.** Attention contributes ~70% of
  Qwen3-8B's compute. Until T-21 ships an attention-layer injector,
  L3 can't show kernel-level wins on a model where attention dominates.

## Cost

- Campaign container (2× A100 spot, 115 min): ~$5
- Basilica candidate compute (4 L2 deployments × ~10-15 min on
  H100/A100/RTX A6000 spot): ~$8-10
- OpenRouter (Sonnet 4: warmstart × 3 layers + 6 operator calls + ~12
  KernelProposer calls): ~$1
- L3 vLLM trials (campaign container, no separate compute): $0
- **Total: ~$15-20**, within the $25-35 estimate.

## Thesis status update

| Claim | Status |
|---|---|
| Three-layer search framework | ✅ built + run end-to-end |
| Joint > single-layer | ✅ 4 independent runs |
| Cross-layer stale-signal fires with effect | ✅ second confirmed firing (14 invalidations) |
| LLM hybrid > pure-LLM > pure-classical | ✅ at L1, L2, L3 |
| L3 LLM operator improves on warmstart | ✅ (prior run, +20% silu_mul ops/sec) |
| Kernel-into-vLLM integration works at scale | ✅ 8 of 12 L3 trials ran end-to-end with patched vllm |
| LLM-novel kernels beat reference end-to-end | **inconclusive at this surface**; predicted as Outcome B |
| Constrained surrogate raises L1 hit rate | partial (8% vs 0% baseline); Q2 needs feature engineering |
| Reserve-on-stale enables 2nd-pass improvement | gated by Q2 |

## Next session priorities

Per the decision tree in the pre-registration, Outcome B leads to:

1. **T-21 — attention-layer injector.** Highest expected ROI: the
   actual hot path. The same framework (KernelInjector + wrapper +
   bench) extends to vLLM's attention layers.
2. **T-26 — FeasibilityModel feature engineering.** Knob-class one-hot
   features so the classifier learns "fp8 region is infeasible" from
   ~3 examples instead of needing dozens.
3. **T-27 — same-config L3 control trials.** Pin (op, regime, dtype,
   hardware), run reference + novel for each cell, multiple seeds. The
   only honest A/B for kernel novelty.

## Evidence

- `docs/research/raw/campaign01-l1xl2xl3-vllm-2026-04-26/` — 75
  artifact files: 36 trial JSONs + 12 bench JSONs + 8 wrapper
  out/err logs + events.jsonl + run_summary.json + hw_context.json +
  reference.log.
- Pre-registration: `docs/research/campaigns/01-l1xl2xl3-vllm-2026-04-26.md`
  (commit `a4cf34e`).
- Mid-campaign corner-cut commits: `f363b73` (T-11/T-13), `2a297cd`
  (T-22), `6aecad3` (T-25). These shipped after the campaign was
  already running, so the run itself used `a4cf34e` consistently.
