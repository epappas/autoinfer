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

## Outcome (filled in after the run)

**Status:** PLANNED → ?

### Headline numbers
*To fill after run.*

### Paired-cell A/B table (Q1)

| Cell | Reference tok/s | Novel tok/s | Δ% | Reference KL | Novel KL | Verdict |
|---|---|---|---|---|---|---|
| rmsnorm/bf16/medium | … | … | … | … | … | … |
| rmsnorm/bf16/large | … | … | … | … | … | … |
| rmsnorm/fp16/large | … | … | … | … | … | … |
| silu_mul/bf16/medium | … | … | … | … | … | … |
| silu_mul/bf16/large | … | … | … | … | … | … |
| silu_mul/fp16/large | … | … | … | … | … | … |

### L1 surrogate hit-rate (Q2)

```
n_warmstart_kept / n_warmstart   = … / 4
n_operator_kept   / n_operator    = … / ~3
n_surrogate_kept  / n_surrogate   = … / ~7
n_reserve_kept    / n_reserve     = … / 0–6
```

`fp8_proposals_after_first_fp8_failure = …` (target ≤ 1)

### Reconciliation with predictions

| Prediction | Actual | Match? |
|---|---|---|
| Outcome A: LLM-novel beats reference @ ≥ 1 cell | … | … |
| Outcome B: all 6 pairs within ±2% | … | … |
| Outcome C: Q2 ≥ 30% L1 kept-rate | … | … |
| Outcome D: Q2 partial 10–25% | … | … |
| Outcome E: Q3 reserve-on-stale 2nd-pass improvement | … | … |
| Outcome F: Q4 parser brittleness > 2/6 fallback | … | … |

### Bugs surfaced and their fixes
*To fill.*

### What's still open after this run
*To fill — likely T-21 escalation regardless of Q1 outcome.*

### Cost actually spent
*To fill.*

### Artifacts

- `docs/research/raw/campaign02-paired-control-2026-04-26/`
- `docs/research/references/12-campaign02-results.md`
- Pre-registration commit: (this commit)
- Run-time commits: (any mid-run fixes)
