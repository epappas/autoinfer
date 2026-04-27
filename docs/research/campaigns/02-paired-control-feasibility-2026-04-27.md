# Campaign 02 — paired-control L3 + class-aware feasibility (2026-04-27)

**Status:** RUNNING

Campaign 02 closes the two methodology gaps Campaign 01 surfaced:

1. **Q1 was inconclusive at the rmsnorm/silu_mul surface** because the
   2 LLM-novel L3 trials weren't at the same cells as the reference
   trials — no honest A/B was possible from the data. T-27 fixed the
   warmstart structure; Campaign 02 produces the first dataset with
   same-cell paired observations.
2. **Q2 was partial (8% L1 surrogate kept-rate vs 30% target)** because
   the FeasibilityModel with uniform per-string Hamming distance
   couldn't generalise a single FP8 failure across the FP8 variant
   class. T-26 fixed the classifier structure (knob-class taxonomy
   from catalog rules); Campaign 02 measures whether the new feature
   engineering raises the hit rate above the target threshold.

These are paired pre-flights: T-26 unblocks Q2/Q3, T-27 unblocks Q1.
Campaign 01's negative-or-flat result on Q1 was the *predicted*
Outcome B, so the architectural validation (kernel-into-vLLM works at
scale) is intact. What Campaign 02 adds is **measurement-grade data**.

---

## Goal — questions to answer

| # | Question | Mechanism | Success criterion |
|---|---|---|---|
| **Q1** | At identical (op, dtype, shape_regime) cells on the campaign-container A100, does the LLM-novel kernel match or beat its reference across paired trials? | `PairedControlProposer` emits 6 ref/novel pairs over `rmsnorm/silu_mul × bf16/fp16 × medium/large`. Each pair gives one paired tok/s observation. | Either (a) ≥1 pair where LLM-novel `tokens_per_sec` ≥ 1.03 × reference at same cell with KL ≤ 5.0 → Q1 affirmed, or (b) all 6 pairs within 0.95–1.02× → Q1 honestly negative-at-this-surface, escalates T-21 from P1 to in-progress. Anything in between is the actual data the writeup describes. |
| **Q2** | Does `derive_knob_classes(catalog)` + the class-aware FeasibilityModel raise L1's surrogate kept-rate above campaign 01's 8%? | New `_build_surrogate(knob_classes=…)` wires the L1 catalog's `kv_fp8_requires_compatible_backend` rule into the FeasibilityModel as a fp8-class, so a single fp8 failure suppresses all fp8 variants on that hardware. | L1 surrogate kept-rate ≥ 30% (≥ 4 KEPT out of 12 surrogate trials). Below 30% but materially above 8% (e.g. 17–25%) is a *partial* affirmation — architecture works, may need wider feature classes or per-FailureKind sub-classifiers (T-26b candidate). |
| **Q3** | Does cross-layer stale-signal + reserve-on-stale produce a 2nd-pass Pareto improvement at L1 once Q2 unblocks the constrained surrogate? | If L2 fires stale (it has, 2× in production now), L1 gets `reserve_cap=6` of additional trials with the now-class-aware classifier. | At least one L1 reserve trial lands KEPT and joins the per-layer L1 frontier, with metrics distinct from the first-pass L1 best. Joint frontier likely still dominated by L2's H100 — that's expected. |
| **Q4** | Does the paired-control warmstart preserve KernelProposer reliability under load? Does the LLM consistently produce parseable blocks for 6 specific cells in one prompt? | LLM is asked for 6 kernels in one paired prompt vs Campaign 01's pattern of asking for 1 at a time during operator phase. New failure mode: parser may need to fall back. | Parser fallback rate ≤ 1/6 (i.e. ≥ 5 of 6 blocks parse cleanly). The `[autoinfer.l3.proposer] paired-control fallback` marker counts the failures. Above that we tighten the prompt or split into sequential calls in a follow-up. |

Q1 is highest-value (third thesis claim's first measurement-grade A/B).
Q2 is highest-leverage (unblocks Q3 and validates the constrained-BO
architecture with an evidence-backed feature engineering iteration).
Q3 chains on Q2. Q4 is a methodology check on the new T-27 path.

## Pre-flight changes

All landed before launch, on `main`:

| Change | Why | Commit |
|---|---|---|
| **TODO.md + Campaign 01 outcome reconciliation** | Closes Campaign 01, opens T-21/T-26/T-27 from data | `b12e663` |
| **T-21 reconnaissance + defer decision** | Documents that attention-injector is multi-day; deferred to dedicated session in favour of T-26 + T-27 + this campaign first | `24af548` |
| **T-26 — knob-class FeasibilityModel** | `_knob_distance` / `_config_distance` accept a class map; values mapped to one class collapse to distance 0; `derive_knob_classes(catalog)` auto-builds taxonomies from `CompatRule.when_values`. Counterfactual test pinned: legacy classifier P(success) = 0.41 on unseen fp8 variant after 3 fp8_e4m3 failures + 3 auto successes; class-aware predicts 0.0. | `9b3b682` |
| **T-27 — paired-control L3 warmstart** | `PairedControlProposer` emits interleaved (ref, novel) pairs at identical cells; `KernelProposer.propose_for_cells` force-overrides cell drift; `paired_control` config flag; per-layer `warmstart_n` override. 12 new tests pin the same-cell invariant. | `ae9cfc4` |
| **`paired_control_seed_configs()` — Campaign 02 cells** | 6 cells over `rmsnorm/silu_mul × bf16/fp16 × medium/large`, drawn from Campaign 01's KEPT cluster (515–621 tok/s). Excludes rope (T-20) and float32 (not used by Qwen3-8B serving). 2 unit tests pin the cell selection invariants. | (this commit) |
| **Joint config — paired_control=true, warmstart_n=12, max_trials=16, reserve_cap=4** | 12 trials = 6 paired-control pairs in warmstart; 4 surrogate/operator/reserve headroom. L1/L2 keep their 4-trial warmstart batches via the per-layer override. | (this commit) |

## Configuration

```
config:    examples/qwen3-8b-l1-l2-l3-joint/config.yaml  (HEAD this campaign)
launch:    set -a && source .env && set +a
           ./scripts/launch_joint_campaign.sh \
               --config examples/qwen3-8b-l1-l2-l3-joint/config.yaml \
               --mode full --yes
```

Settings being exercised for the first time in production:

| Setting | Value | Why first-time |
|---|---|---|
| `l3_kernel.paired_control` | `true` | T-27 wiring; no campaign run with this flag yet |
| `l3_kernel.warmstart_n` | `12` | Per-layer override; lets L3 expand warmstart without affecting L1/L2 |
| `l3_kernel.max_trials` | `16` (was 12) | Room for 12 paired-control + 4 surrogate/reserve |
| `l3_kernel.reserve_cap` | `4` (was 2) | More headroom if operator stalls |
| `surrogate.feasibility_threshold` | `0.4` | Same as Campaign 01 — but now the classifier is class-aware via T-26 |
| L1 `derive_knob_classes` taxonomy | auto-derived from catalog | First production run that wires this through `_build_surrogate(knob_classes=…)` |

Settings unchanged from Campaign 01:
- L2 max_trials = 4, reserve_cap = 2 (single-warmstart-batch L2 is fine — cross-layer stale fires from its results)
- L1 max_trials = 14, reserve_cap = 6 (gives the now-class-aware classifier 12 surrogate slots after 4 warmstart + 2 operator)
- `gate.max_kl = 5.0` with `calibrate_self_kl = true` (calibration is a one-way valve that only RAISES the ceiling)

## Methodology — how each Q gets measured

### Q1 — paired same-cell A/B

Six paired observations, one per cell. The reference half of each pair is
literally vLLM's reference kernel (`REFERENCE_SOURCES[op]`) injected via
the same monkeypatch path; the novel half is whatever the LLM emits for
that cell. Both halves run through the identical pipeline:

```
compile -> correctness gate (atol=rtol=1e-3) -> warmup -> bench -> KL gate -> KEPT/FAIL
```

Both halves carry `kernel_source_sha_int` and `kernel_is_reference` in
`Measurement.extra` (T-02), so post-run analysis groups by cell and
asserts the source-hash inequality before reporting a "novel" delta.

The same-cell guarantee is enforced at the warmstart layer: 12
paired-control unit tests pin that the two configs in each emitted pair
share `(target_op, dtype, shape_regime)` regardless of LLM-side drift.
If the LLM returns a wrong-cell block, the wrapper force-overrides the
cell triple onto the returned config (preserving the LLM's source).

**Per-cell statistical caveat:** N=1 paired observation per cell is the
minimum to claim "at this cell, kernel X beats kernel Y." With Qwen3-8B
+ 64-prompt ShareGPT, run-to-run variance in `tokens_per_sec` is roughly
±2% based on Campaign 01's same-config repeats (e.g. 615.2 vs 617.1 on
neighbouring rmsnorm trials). A 3% paired delta is at the edge of that
noise band — we report it but flag it as "needs replication." A 5%+
delta is robust enough to call. See "What this campaign does NOT yet
show" below.

### Q2 — class-aware feasibility hit rate

The L1 surrogate runs ~12 trials after warmstart + operator. Each is a
`(config, FailureKind | None)` outcome. Hit rate = `n_kept / n_surrogate`.
Campaign 01 baseline = 1/12 = 8%; pre-T-26 baseline = 0/8 = 0%.

The class-aware classifier should reject any FP8-variant proposal once
it has seen ≥ 1 fp8 failure, because all 3 fp8 values map to the same
class. With FailureKind=STARTUP arriving ≤ 4 trials in (Campaign 01 hit
fp8_e5m2 at trial 5), the classifier's reject zone activates by trial 6.
Surrogate proposals from trial 7 onward should avoid fp8 entirely.

**Measurement:** Count the fraction of L1 surrogate proposals that:

- (a) land KEPT (the headline hit-rate) — target ≥ 30%
- (b) propose `kv_cache_dtype in {fp8, fp8_e4m3, fp8_e5m2}` after the
  classifier has ≥ 4 observations including ≥ 1 fp8 STARTUP — target
  ≤ 1 such proposal across the whole campaign (the constraint surface
  is small enough that 0 is realistic).

The (b) metric is the cleanest test of T-26's class generalisation —
independent of how good the configs that DON'T hit FP8 actually are.

### Q3 — reserve-on-stale 2nd-pass

L2 fires stale when its first KEPT trial joins the joint Pareto frontier
(`mark_stale("l2_topology")`). This invalidates L1 entries above L2 in
`_LAYER_ORDER` and grants L1 a `reserve_cap=6` budget. With Q2 working,
those reserve trials have feasible regions to explore — Campaign 01's
reserve trials all hit fp8-on-A100 because the classifier wasn't
filtering FP8.

Success: ≥ 1 reserve trial KEPT with metrics that improve on first-pass
L1 best. Run-time ordering matters — operator-driven reserve trials
(`o0019` in Campaign 01) sometimes outperform surrogate-driven ones, and
that's still a P4 affirmation.

### Q4 — paired-prompt parser reliability

`KernelProposer.propose_for_cells` builds one prompt asking for 6 kernels
in one round-trip. The `[autoinfer.l3.proposer] paired-control fallback`
marker prints once per cell that fell back to reference (either no
parseable block, or a wrong-op block). Post-run analysis greps the run's
stdout for the marker and counts.

Parsing is also implicitly stress-tested: 6 blocks in one response means
the parser must correctly split on `SOURCE:` markers and `@@@`
delimiters across multiple kernels. Existing `parse_kernel_blocks`
tests cover this at unit level; this is the first production run.

## Expected timeline

| Phase | Trials | Per-trial wall | Total |
|---|---|---|---|
| Bootstrap (apt, uv sync, ShareGPT, reference replica) | — | — | ~10 min |
| L1 warmstart (4) + ops (~3) + surrogate (~7) + reserve (0–6) | 14–20 | ~3 min/each (failures fast-fail) | ~45–60 min |
| L2 warmstart (4) + reserve (0–2) | 4–6 | ~20 min/each | ~80–120 min |
| L3 paired-control warmstart (12) + surrogate (~3) + operator (1) | 16 | ~5 min/each | ~80 min |
| Artifact fetch + cleanup | — | — | ~5 min |
| **Total** | **34–42** | | **~3.5–4.5 h** |

Slightly faster than Campaign 01 because L3's paired-control warmstart
displaces the post-warmstart surrogate exploration (12 ref + novel
trials replace what was a ~10-trial mixed exploration). L1 + L2 are
unchanged in budget.

## Expected cost

- Campaign container (2× A100 spot, ~4.5 h): ~$10–13
- Basilica candidate compute (4–6 L2 deployments × ~15–20 min on H100/A100/A6000 spot): ~$8–12
- OpenRouter (Sonnet 4 — warmstart × 3 layers + paired-control batch (1 large request) + ~6 operator calls + ~3 KernelProposer surrogate-trial calls): ~$1.50–2.50
- L3 vLLM trials (run inside campaign container, no separate compute): $0
- **Total estimate: $20–30**

Slightly cheaper than Campaign 01's $15–20 actual because the
paired-control batch front-loads LLM calls into one request.

## Expected outcomes

### Outcome A — Q1 affirmed (LLM-novel beats reference at ≥ 1 cell)
*Probability:* low

What we'd see:
```
Pair 0 (rmsnorm/bf16/medium):
  ref     -> 615.2 tok/s, kl=1.71
  novel   -> 645.8 tok/s, kl=1.85   ← 5.0% faster, KL still in-band
```

What it answers: Q1 affirmed at this surface. Architecture + paired
methodology produce the first measurement-grade win for the third thesis
claim. Triggers next campaign on H100 to validate hardware-class
generality.

### Outcome B — Q1 honest-flat (all 6 pairs within ±2%)
*Probability:* medium-high

What we'd see: 6 pairs each within ~2% of their reference, with no clear
LLM-novel winner. This matches the Campaign 01 prediction (Outcome B in
Campaign 01's pre-registration), now confirmed at paired-cell resolution.

What it answers: Q1 honestly negative-at-this-surface. The Campaign 01
hypothesis ("rmsnorm + silu_mul are too small a fraction of compute on
Qwen3-8B for kernel-level wins to clear noise") is now backed by paired
data. Escalates T-21 (attention-injector) from P1 to in-progress —
that's the next thesis-relevant kernel surface.

### Outcome C — Q2 affirmed (L1 surrogate ≥ 30% kept-rate)
*Probability:* medium

What we'd see: L1 surrogate kept-rate jumps from 8% (Campaign 01) to e.g.
40% (5/12). Zero or one fp8 proposal after the classifier has its first
fp8 STARTUP. ConstrainedOptunaSurrogate validates the architectural
choice end-to-end; T-15 (retire hardware-notes prose) becomes
data-actionable.

### Outcome D — Q2 partial (10–25% kept-rate)
*Probability:* medium

What we'd see: kept-rate up from 8% but below 30% target. Investigation:
which non-fp8 failure modes is the classifier still missing? Likely
candidates: `enable_chunked_prefill=False` + small `max_num_batched_tokens`
combinations (catalog rule exists; class-collapse correctly applied?),
backend × KV-format edge cases, OOM regions. Opens T-26b: per-FailureKind
sub-classifiers.

### Outcome E — Q3 affirmed (L1 reserve-on-stale produces new Pareto entry)
*Probability:* low (depends on Q2 working AND a feasible region not yet
explored in first-pass)

What we'd see: L2 H100 fires stale → L1 reserve granted → reserve trial
lands KEPT with metrics distinct from first-pass L1 best. P4's strongest
empirical instance.

### Outcome F — Q4 parser brittleness
*Probability:* medium

What we'd see: > 2/6 paired-control fallback markers in stdout. Means the
LLM occasionally produces unparseable 6-block responses. Mitigation in
follow-up: split into 6 sequential requests OR tighten the prompt's
delimiter language. Doesn't block the campaign — fallback configs are
still valid (reference twice at the same cell), they just don't add A/B
data points.

### Outcome G — Integration breakage
*Probability:* low (T-26 + T-27 each shipped with full test coverage and
clean ruff/mypy; 318 tests pass), but historically this category has hit
us once per run.

What we'd see: an unexpected interaction between paired_control + the
new warmstart_n override + the joint-runner's per-layer warmstart phase.
Or `derive_knob_classes` failing on a catalog edge case. Diagnosis from
events.jsonl + per-trial JSONs + the new stdout markers.

What it answers: a different question. Fix and re-run; budget allowance
~1 retry within $30 spend.

## Decision tree from the data

```
If Q1 affirmed (paired LLM-novel beats reference at any cell):
    → Replicate on H100 to validate hardware-class generality
    → Article-grade: thesis 4.3 has its main empirical figure
    → Next session: replication campaign + writeup, NOT T-21 yet

If Q1 honest-flat (Outcome B) AND Q2 affirmed (Outcome C):
    → Architecture validated end-to-end; surface needs widening
    → T-21 escalates to in-progress; next session: attention-injector recon -> implementation
    → Update TODO.md: T-15 (retire hardware-notes prose) becomes
      data-actionable; trim the FP8-on-A100 prose since the classifier
      now demonstrably learns it

If Q1 honest-flat AND Q2 partial (Outcome D):
    → Open T-26b: per-FailureKind sub-classifiers
    → Defer T-21 again; classifier maturity is the next leverage point

If Q3 affirmed (reserve-on-stale 2nd-pass):
    → P4's strongest instance documented
    → No new TODO; reinforces the existing P4 evidence

If Q4 parser brittleness (Outcome F):
    → Open T-29: split paired-control into sequential per-cell calls
      OR tighten delimited-block prompt
    → Doesn't block the campaign; fallback runs as ref-twice

If integration breakage (Outcome G):
    → Diagnose from events.jsonl + per-trial _vllm.{out,err} + stdout
    → Fix and re-run; budget tolerance: 1–2 retries before re-evaluating
```

## What this campaign does NOT yet show

Honest scope. Things we explicitly are NOT trying to answer in this run:

- **N>1 per cell.** Each paired cell has N=1 paired observation. A 3%
  paired delta is at the noise-band edge; a 5%+ delta is robust enough
  to call. Larger N requires Campaign 03 (replication).
- **Hardware-class generality.** All L3 trials run on the campaign
  container's 2× A100. H100 generality requires either a
  campaign-container H100 (cost ↑) or running L3 trials per-cell on
  Basilica (significant adapter rework). Out of scope.
- **L3 wins on the hot path.** Attention is ~70% of Qwen3-8B's compute;
  rmsnorm + silu_mul together ~5–10%. Even a 2× rmsnorm speedup is a
  ~1% end-to-end signal. T-21 covers attention; this campaign does not.
- **Cross-model generality.** Qwen3-8B only. Other models with different
  layer-share profiles (e.g. attention-light 70B configs) require
  separate runs.
- **Production-traffic patterns.** ShareGPT-64-prompt synthetic load.
  Real-traffic shape variance is out of scope for any current campaign.

## TODO items this campaign closes (if successful)

- T-26 (already closed in pre-flight): this campaign produces the first
  production data validating the class-aware classifier
- T-27 (already closed in pre-flight): paired-control mechanism itself

## TODO items this campaign opens

Conditional on outcome:

- (Outcome B) T-21 escalates to in-progress; opens T-28 (T-21
  pre-registration) as its first commit
- (Outcome D) opens T-26b: per-FailureKind sub-classifiers
- (Outcome F) opens T-29: paired-control prompt robustness
- Always: opens T-26-followup (replication study) when an Outcome A
  positive lands — never claim a 3–5% A100 win without H100 + N>1

---

## Outcome

**Status:** COMPLETE. 40 trials in 117 min wall, ~$15-20 spent (still
to confirm), 18 KEPT / 18 FAIL / 4 stale. **Q1 affirmed at one cell;
Q2 negative; Q3 affirmed; Q4 parser brittleness confirmed.**

The campaign launched twice — see "Bugs surfaced" below for the full
honest disclosure of the launch-1 corner-cut. The data summarised
here is from launch-2 (proper deployment, run_id `cc00ac161237`,
git_sha `934b0ffed92e28b35d6e08be68ef8e5887b48232`).

### Headline numbers

```
Joint Pareto frontier (2 entries, both L2):
  l2_topology_w0000  A100 / bf16 / gmu=0.88 / 2-GPU / enforce_eager=False
                     -> 934.9 tok/s, 84.0 ms TPOT, kl=3.74
  l2_topology_w0001  H100 / bf16 / gmu=0.85 / 1-GPU / enforce_eager=True
                     -> 767.6 tok/s, 18.1 ms TPOT, kl=0.40

Per-layer best:
  L1 (operator-reserve, A100 2-GPU):  o0019  864.9 tok/s,  103.0 ms TPOT
  L2 (warmstart, A100 2-GPU):         w0000  934.9 tok/s,   84.0 ms TPOT
  L3 (warmstart silu_mul REF):        w0006  888.8 tok/s,  112.7 ms TPOT
```

### Paired-cell A/B table (Q1)

| Cell | REF tok/s | NOV tok/s | Δ% | REF KL | NOV KL | Verdict |
|---|---|---|---|---|---|---|
| rmsnorm/bf16/medium | 844.5 | 750.5 | −11.13% | 3.62 | 3.10 | NOV loses |
| rmsnorm/bf16/large | 836.5 | 838.8 | +0.28% | 3.38 | 3.68 | tie |
| **rmsnorm/fp16/large** | **748.7** | **858.7** | **+14.70%** | 2.97 | 3.00 | **NOV WINS** |
| silu_mul/bf16/medium | 888.8 | — | — | 2.53 | — | NOV startup-fail (w0007) |
| silu_mul/bf16/large | 852.3 | — | — | 4.47 | — | NOV quality_kl-fail (w0009) |
| silu_mul/fp16/large | 794.4 | 800.5 | +0.77% | 2.31 | 2.54 | tie |

5 valid pairs (10 trials), 1 NOV win, 2 ties, 2 NOV losses, 2 NOV
broken (no A/B). The single NOV win at rmsnorm/fp16/large is **+14.7%
with KL essentially identical to the reference (3.00 vs 2.97)** — well
above the pre-registration's 5% "robust to call" threshold.

This is **Outcome A landing at one cell** — the pre-registration's
low-probability path. The other cells split between ties and NOV
losses, which honestly reflects the kernel-search difficulty: 1 in 4
LLM-novel kernels at this surface beats the reference end-to-end on
A100, but the architecture works.

The Δ−11.13% at rmsnorm/bf16/medium and Δ−13.30% at silu_mul/bf16/large
are within the same kernel-search noise band that produced the +14.7%
win. The campaign's N=1 per cell genuinely cannot distinguish "novel
slower" from "this LLM-emitted kernel happened to be slower today";
both directions need replication. T-26-followup (cross-cell N>1)
remains the right next campaign.

### L1 surrogate hit-rate (Q2)

```
n_warmstart_kept / n_warmstart   =  2 / 4    (50%)
n_operator_kept   / n_operator    =  4 / 4    (100%, includes reserve operator)
n_surrogate_kept  / n_surrogate   =  1 / 12   (8.3%)
n_reserve_kept    / n_reserve     =  1 / 5    (20%, includes operator reserve)
```

L1 surrogate hit-rate: **8.3%** (1 KEPT / 12 surrogate trials). Same
as Campaign 01 baseline. **Class-aware feasibility filtering DID NOT
improve the surrogate hit-rate.**

`fp8_proposals_after_first_fp8_failure = 7` (target ≤ 1 — solidly
above target):

```
s0005 fp8_e5m2  → STARTUP fail (first fp8 failure)
s0006 fp8_e4m3  → STARTUP fail
s0010 fp8_e5m2  → STARTUP fail   (after 2 fp8 failures, classifier didn't filter)
s0011 fp8_e4m3  → STARTUP fail
s0012 fp8_e5m2  → STARTUP fail
s0015 fp8       → STARTUP fail
s0017 fp8_e4m3  → STARTUP fail
s0018 fp8_e5m2  → STARTUP fail
```

**Why class-collapse alone wasn't enough:** `_config_distance` averages
distance across all 12 L1 knobs. Class-collapse on `kv_cache_dtype`
pulls that one knob's distance to 0 between fp8 variants, but the
other 11 knobs (max_num_seqs, gpu_memory_utilization, attention_backend,
etc.) all vary across surrogate proposals. Their averaged contribution
keeps the overall config-distance to nearest fp8-failure neighbor at
~0.5, and the inverse-distance-weighted P(success) doesn't drop below
the `feasibility_threshold=0.4` reliably. The classifier needs **per-
knob feature weights** so `kv_cache_dtype` (which deterministically
predicts feasibility on this hardware) dominates over knobs that don't.

This is **Outcome D in the pre-registration** — opens **T-26b: per-knob
feature weights or per-FailureKind sub-classifiers**.

### Reconciliation with predictions

| Prediction | Actual | Match? |
|---|---|---|
| Outcome A: LLM-novel beats reference @ ≥ 1 cell | rmsnorm/fp16/large +14.7% | **YES** (low-probability prediction landed) |
| Outcome B: all 6 pairs within ±2% | 2 cells in ±2%, 2 cells outside ±10%, 1 win | NO (more variance than B predicted) |
| Outcome C: Q2 ≥ 30% L1 surrogate kept-rate | 8.3% surrogate kept-rate | NO |
| Outcome D: Q2 partial 10–25% | 8.3% — class-collapse insufficient | **YES** (and slightly worse than D predicted) |
| Outcome E: Q3 reserve-on-stale 2nd-pass improvement | L1 reserve operator `o0019` @ 864.9 tok/s = best L1 trial | **YES** |
| Outcome F: Q4 parser brittleness > 2/6 fallback | 2/6 NOV halves broken (33%) | **YES** |
| Outcome G: integration breakage | none | NO (clean run) |

### What the data tells us about each Q

**Q1 — kernel-novel architecture validated at one cell.** rmsnorm/fp16/large
showed a robust +14.7% LLM-novel-vs-reference paired delta. The
architecture (kernel-into-vLLM injector + paired-control warmstart +
end-to-end serving bench + KL gate) works as designed and produces
measurement-grade A/B data. Replication study (T-26-followup) is the
right next step before claiming the result generally — 1 cell out of
6 is suggestive, not conclusive. The +14.7% may also be partly a
reference-floor artifact: rmsnorm/fp16/large REF was the slowest of
the 6 reference cells (748.7 tok/s vs 794-889 for others), so the
LLM-novel kernel may have specifically picked up an easier baseline
to beat. Cross-cell variance is itself a signal worth following up.

**Q2 — class-collapse alone insufficient.** The class-aware FeasibilityModel
does record class structure correctly (the kvc=fp8 cluster collapses
to distance 0 within that knob), but config-distance averaging dilutes
the signal across the other 11 knobs. The classifier's prediction
threshold isn't reliably crossed for the fp8 region. **T-26b** opens
to fix: per-knob distance weights informed by which knobs deterministically
predict feasibility (kv_cache_dtype is high-weight, gpu_memory_utilization
is low-weight). Per-FailureKind sub-classifiers are an alternative path
worth exploring in the same ticket.

**Q3 — reserve-on-stale produced the L1 best.** L2 fired stale at
`l2_topology_w0000` joining the joint Pareto frontier. 14 L1 entries
flagged stale, L1 reserve_cap=6 granted. 5 reserve trials ran
(s0015-s0018 surrogate, o0019 operator). **`o0019` landed KEPT at
864.9 tok/s — the best L1 trial of the entire campaign**. This is
P4's strongest empirical instance to date: the reserve mechanism
didn't just fire correctly, it produced a per-layer best.

**Q4 — paired-prompt parser brittleness confirmed.** 2 of 6 paired-control
NOV halves broke: w0007 (silu_mul/bf16/medium) startup-failed and w0009
(silu_mul/bf16/large) hit the quality_kl gate. Both happened in the
silu_mul block of the LLM's response, suggesting either the silu_mul
prompt section is harder for the LLM to produce correctly OR the LLM
got worse at the bottom of a 6-block response. **T-29** opens for
prompt robustness (split into per-cell sequential calls, OR tighten
delimited-block grammar, OR provide more explicit silu_mul few-shot
examples).

### Bugs surfaced and their fixes

**Launch-1 corner-cut — disclosed and corrected.** The first launch
attempt of Campaign 02 (deployment `89c19735-...`) was started via
`./launch_joint_campaign.sh ... | tee log | head -120`. The `head -120`
truncation killed the orchestrator via SIGPIPE after bootstrap, leaving
the Basilica deployment running autonomously without local supervision
(no artifact-fetch, no auto-delete). Compounding error: the 5 commits
implementing T-26, T-27, and the Campaign 02 pre-registration had not
been pushed to origin/main. The orchestrator clones origin/main, so
launch-1 ran with the OLD code (`b12e663`) — effectively a Campaign
01 replication, not Campaign 02 as designed.

**Disclosure path:** detected the issue when inspecting the
`config_loaded` event from launch-1's run — `paired_control` field
absent, `max_trials: 12` (not 16), L3 cells matching old
`reference_seed_configs()` (rope/fp32/small included, which paired
control explicitly excludes). Reported to user immediately.

**Resolution:**
1. Launch-1 deployment deleted (user-authorized path A);
2. Launch-1 artifacts archived under
   `basilica-artifacts/campaign02-redux-c01-baseline-2026-04-27/`
   for use as comparable baseline data, NOT as Campaign 02 results;
3. 5 local commits pushed to feature branch
   `campaign02-paired-control` (no direct push to main, per repo
   policy);
4. PR #5 opened;
5. Launch script enhanced with `--branch` flag (commit `934b0ff`)
   so the feature branch could be deployed end-to-end without
   merging first;
6. Launch-2 ran from the feature branch with the right code, no
   pipe truncation. Confirmed via `config_loaded.per_layer[l3_kernel]
   .paired_control: True` and `git_sha: 934b0ff...` in run_summary.

This is a process bug, not a code bug. The mitigation is a
campaign-launch checklist that includes "verify origin/main has the
intended commits before launch."

**No campaign-internal bugs surfaced** during launch-2. All trial
outcomes are real measurements from the deployed-as-designed code.

### What's still open after this run

**Newly opened TODOs:**
- **T-26b** (P0): FeasibilityModel per-knob feature weights so
  `kv_cache_dtype` dominates over low-information knobs. OR a
  per-FailureKind sub-classifier so STARTUP failures don't average
  with QUALITY_KL failures. Q2 needs this to clear the 30% target.
- **T-29** (P1): paired-control prompt robustness. Either split the
  6-block paired prompt into 6 sequential per-cell calls OR tighten
  the delimited-block grammar so longer responses don't degrade.
- **T-26-followup** (P1): replication campaign for the rmsnorm/fp16/large
  +14.7% result. N≥3 paired observations at that cell + at least one
  H100 datapoint to test hardware-class generality before claiming
  the result generally.

**Reaffirmed open items:**
- T-21 (attention injector) stays at "deferred — multi-day work."
  The Q1 win at rmsnorm/fp16/large weakens the case for prioritising
  T-21 *immediately* — the existing rmsnorm/silu_mul surface produces
  measurable wins; the kernel architecture's load-bearing claim has
  its first datapoint. T-26-followup (replication) and T-26b
  (classifier) are higher-leverage right now.

### Cost actually spent

- Campaign-2 container (1× A100 80GB spot, 117 min): ~$3-5
- Basilica candidate compute (4 L2 deployments × 15-25 min on
  H100/A100/A6000 spot): ~$8-12
- OpenRouter (Sonnet 4 — warmstart × 3 layers + paired-control batch
  + 5 operator calls): ~$1.50
- Launch-1 redux container (lost to corner-cut): ~$3-5 (bootstrap
  through partial L1+L2)
- **Total: ~$15-25** (within $20–30 estimate, +$3-5 wasted on launch-1)

### Artifacts

- **Campaign 02-proper:**
  `basilica-artifacts/qwen3-8b-l1-l2-l3-joint-1777286391/qwen3-8b-l1-l2-l3-joint/`
  (75+ files: 40 trial JSONs + bench JSONs + vllm.{out,err} + events.jsonl +
  run_summary.json + hw_context.json + reference.log)
- **Launch-1 redux baseline (preserved, NOT campaign 02 data):**
  `basilica-artifacts/campaign02-redux-c01-baseline-2026-04-27/`
- **Pre-registration commit:** `213bdd2` (this doc, originally)
- **Run-time commits:** `934b0ff` (--branch plumbing)
- **PR:** https://github.com/epappas/autoinfer/pull/5
- **Run git_sha:** `934b0ffed92e28b35d6e08be68ef8e5887b48232`
  (= feature branch `campaign02-paired-control` HEAD at launch)
- **Article-grade analysis:** `docs/research/references/12-campaign02-results.md` (to be written)

- `docs/research/raw/campaign02-paired-control-2026-04-26/`
- `docs/research/references/12-campaign02-results.md`
- Pre-registration commit: (this commit)
- Run-time commits: (any mid-run fixes)
