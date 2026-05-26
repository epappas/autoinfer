# Campaign 04 — autoinfer-L1 vs vLLM `auto_tune.sh` (2026-05-26)

**Status:** PLANNED

Pre-registration written before launch. The "Outcome" section at the
bottom is filled in after the run, reconciling predictions with reality.

Per repo discipline: every "Pre-flight changes" entry below cites a
real commit on `main`. The C04 recon retraction (PR #39) explains why
this campaign covers two sequential sub-campaigns ("C04a" + "C04b")
rather than one — the recon discovered three load-bearing errors in
the original framing before T-37 was launched. Read the recon's
2026-05-26 corrective addendum + the T-37 baseline artifact + the
framing-overview note (PR #47) for the full why-this-shape narrative
before reading this pre-reg.

---

## What this campaign measures

C04 is the autoinfer thesis's **first per-layer head-to-head against a
published specialist tool**. The specialist is vLLM's
`benchmarks/auto_tune/auto_tune.sh` — a small bash grid-search over
`max_num_seqs` × `max_num_batched_tokens` that picks the highest
throughput meeting an end-to-end latency ceiling.

T-37 (PR #46) ran that script on Basilica and captured three reference
goodput numbers. C04 runs autoinfer-L1's hybrid policy on the *same
workload tuples* and asks: at the same trial budget, does the surrogate
find an engine-config that beats the grid?

The "Both, in sequence" decision (recon addendum + framing overview)
means C04 covers two sub-campaigns sharing the same harness, workload,
and reference baseline:

- **C04a — 2-knob restricted.** autoinfer-L1 is restricted to the
  same {`max_num_seqs`, `max_num_batched_tokens`} surface
  `auto_tune.sh` searches. All other L1 knobs are fixed at sensible
  defaults matching auto_tune's behaviour (chunked prefill on,
  attention backend left at vLLM's chosen default, prefix caching
  enabled, dtype=auto, no quantization, kv_cache_dtype=auto,
  enforce_eager=false). The question this sub-campaign answers:
  **does Bayesian optimisation beat a bash grid on the same search
  surface?**

- **C04b — full surface.** autoinfer-L1 with its full 12-knob
  catalog (the 10 baseline knobs + the two V1 knobs landed in T-33:
  `long_prefill_token_threshold`, `enforce_eager`). The question:
  **does the wider search find recipes the 2-knob grid literally
  cannot reach, and are they meaningfully better?**

C04a is the smaller, conservative claim — a methodology validation.
C04b is the actual L1-layer thesis test (joint search of a wider
surface wins). Running both gives two independent data points; the
four-quadrant outcome matrix (win/win, win/loss, loss/win, loss/loss)
is enumerated in the framing-overview note (PR #47).

### Methodology footnote — dummy vs real weights

`auto_tune.sh` runs `vllm serve --load-format dummy` (random
weights; no actual model download or load). autoinfer **cannot** use
`--load-format dummy` because the C9 live-reference-replica KL gate
requires real outputs from both candidate and reference; dummy
weights produce garbage outputs and the gate would fail every trial.

Implication: autoinfer measures throughput AND output quality under
real weights; auto_tune.sh measures only throughput under random
weights. Both measure the same goodput axis (requests meeting
TTFT/TPOT/E2E SLO per second). Since the per-request bottleneck on
both sides is *compute*, not weight access, the goodput comparison
is meaningful — but autoinfer is held to a strictly higher bar (real
weights + quality gate). A "tie" outcome is still a methodology win
because autoinfer matched a weaker (quality-blind) baseline.

The analysis writeup will explicitly note this asymmetry.

### Why Baseline B is the primary reference

The T-37 baselines give three independent reference points (PR #46):

| Baseline | Hardware | Workload | SLO | Goodput |
|---|---|---|---|---|
| A | A100 | INPUT=1800/OUTPUT=20 | none | 8.53 req/s |
| **B** | A100 | INPUT=256/OUTPUT=20 | 500 ms | **21.39 req/s** |
| C | H100 | INPUT=1800/OUTPUT=20 | 500 ms | 2.97 req/s |

C04's primary comparison is against **Baseline B** because:

1. **Same hardware as autoinfer's prior campaigns** (1× A100 spot;
   matches C02/C03's validated deploy path).
2. **Realistic SLO-bounded workload** (chat-shape; the way C9
   "live reference replica gate" is designed to be evaluated).
3. **Lowest physical-infeasibility risk** (the original recon's
   long-context-with-500ms-SLO was infeasible on A100; T-37 attempt
   3 burned $0.50 confirming this. Baseline B's workload is the one
   that produced clean SLO-meeting cells across all 7 non-cold-start
   configs).

Baselines A and C remain as anchor points the writeup can reference
(unconstrained-throughput sanity check; auto_tune README target
sanity check) but C04a and C04b are not run against them in the
primary pre-reg. An **optional C04c** on H100 long-context vs
Baseline C is sketched at the bottom of this doc but **not pre-
registered** — it would launch only if C04a + C04b results are
interesting enough to justify H100 spot spend.

---

## Goal — questions to answer

| # | Question | Mechanism | Success criterion |
|---|---|---|---|
| **Q1 (C04a)** | On a 2-knob surface restricted to {`max_num_seqs`, `max_num_batched_tokens`}, does autoinfer's hybrid policy find a higher-goodput engine-config than `auto_tune.sh`'s grid? | `RunConfig.layers.l1_engine.knobs_path` points at a *restricted* catalog containing only the 2 knobs. Same workload + SLO as Baseline B. autoinfer's surrogate (TPE + T-26c feasibility classifier) + warmstart + LLM operator search the 8-config surface. | Best autoinfer goodput ≥ 23.5 req/s (≥ 1.10 × Baseline B's 21.39) → Q1 affirmed. 19.3–23.5 (within ±10%) → tie. < 19.3 → Q1 negative (surrogate worse than grid — surrogate-bug indicator). |
| **Q2 (C04b)** | With the full 12-knob L1 catalog, does the wider search find a higher-goodput engine-config than `auto_tune.sh` on the same workload? | Same workload + SLO as Baseline B. Same hybrid policy. autoinfer is free to set `kv_cache_dtype`, `attention_backend`, `dtype`, `quantization`, `enable_prefix_caching`, `enable_chunked_prefill`, `block_size`, `gpu_memory_utilization`, `long_prefill_token_threshold`, `enforce_eager` in addition to the 2 auto_tune knobs. | Best autoinfer goodput ≥ 23.5 req/s (≥ 1.10 × 21.39) → Q2 affirmed (thesis support at L1). 19.3–23.5 → wider surface provides no benefit; publishable nuance. < 19.3 → wider search worse than grid (surrogate gets lost). |
| **Q3** | Does the per-FailureKind classifier (T-26c) clear the ≥30% kept-rate target that T-26b couldn't? | T-26b shipped in C03 and produced ~20% L1 kept-rate. T-26c (PR #34) adds per-FailureKind sub-classifiers + max-aggregation. The kept-rate is `n_kept / n_total` over the surrogate phase (warmstart trials excluded since they're forced through). | L1 surrogate kept-rate ≥ 30% → Q3 affirmed; surrogate is converging usefully. 20–30% → partial (better than T-26b but not at target; opens T-26d data-mined priors). < 20% → regression vs T-26b; T-26c needs another iteration. |
| **Q4** | Does the run-level `goodput` axis (T-34) correctly route the Pareto frontier when `harness.driver.slo_e2e_p99_ms=500` is set? | The L1 spec's `objective_axis` should switch from `tokens_per_sec` to `goodput_req_per_sec` automatically; the `config_loaded` event should record `objective_axis="goodput_req_per_sec"` and `goodput_slo_ms={"TTFT":..., "TPOT":..., "E2E":500}`. | Event-log inspection post-run shows `objective_axis=goodput_req_per_sec` and the goodput SLO block. Per-trial JSON's `extra["goodput_req_per_sec"]` is populated. Best-by-tokens_per_sec vs best-by-goodput in `run_summary.json` differ when SLO bites. |

Q1 is the conservative validation — methodology check. Q2 is the
load-bearing thesis test. Q3 is the surrogate-health gate (without it
Q1 and Q2 are meaningless). Q4 is the wiring sanity check on the T-34
work that landed for this campaign.

---

## Pre-flight changes

All landed on `main` before this pre-reg writes against them:

| Change | Why | Commit |
|---|---|---|
| **T-26c** — per-FailureKind sub-classifier | T-26b's shared knob_weights averaged over mixed-kind history, capping kept-rate at ~20% in C03. T-26c routes each FailureKind through its own weight vector with max-aggregation; Q3 measures whether this clears 30%. | PR #34 (`96624aa`) |
| **T-33** — L1 catalog V1 gaps | auto_tune.sh searches a 2-knob surface; autoinfer's surface needs to *contain* it AND extend beyond. T-33 adds `long_prefill_token_threshold` + `enforce_eager` so the C04b full-surface comparison is honestly wider. | PR #35 (`1961e2c`) |
| **T-34** — driver `--goodput` SLO + objective axis flip | auto_tune optimises "max throughput s.t. p99 E2E < SLO". Without the SLO threaded through, autoinfer measures raw throughput and the comparison is incommensurable. T-34 wires `--goodput TTFT:X TPOT:Y E2E:Z` and switches the optimisation axis to `goodput_req_per_sec` when `slo_e2e_p99_ms` is set. | PR #36 (`d915cc7`) |
| **T-35** — corpus sha + per-trial seed | Reproducibility: every run records the workload-corpus sha256 + the `vllm bench serve --seed N` value into `hw_context.json` + the `config_loaded` event. Required for the writeup's "reproduces on a second machine" claim. | PR #37 (`66c8d8d`) |
| **T-36** — `harness.determinism` sub-block | Typed contract for the reference replica's seed / batch_invariant / multiprocessing_v1 levers. The KL gate is only meaningful if the replica's own noise floor is bounded; T-36 makes the levers a per-campaign config knob rather than ad-hoc env vars. | PR #38 (`a50913d`) |
| **C04 recon corrective addendum** | Discovered three load-bearing recon errors (workload, knob scope, vLLM pin) by fetching auto_tune.sh from vllm-project. Reframed C04 as "Both, in sequence" + H100 anchor. | PR #39 (`a0385d4`) |
| **vLLM lock upgrade to 0.21.0** | Match Basilica image (`vllm/vllm-openai:v0.21.0`) so autoinfer and T-37 run the same vllm commit. | PR #40 (`a030316`) |
| **T-37 baseline orchestrator + 3 fixes** | SDK-orchestrated `auto_tune.sh` runs on Basilica. Three fixes landed during the run: install bc; rename cloned vllm/ to avoid import shadow; sed-patch HOSTNAME to localhost. | PRs #41 (`3991d21`), #42 (`5c5d925`), #43 (`b3c6f01`), #44 (`6dd78bc`), #45 (`2a790ca`) |
| **T-37 baseline artifact doc** | Three reference goodputs (8.53 / 21.39 / 2.97 req/s) on Llama-3.1-8B-Instruct, vllm commit `ad7125a431e176d4161099480a66f0169609a690`, with full per-cell grids. This is the doc C04 compares against. | PR #46 (`b7192bd`) |
| **C04 framing-overview note** | Plain-language explainer of what C04 is + what each four-quadrant outcome means. Context preservation for future agents. | PR #47 (`36d33ed`) |
| **L1 restricted-catalog YAML for C04a** | New `examples/c04a-l1-restricted/knobs.yaml` containing only `max_num_seqs` + `max_num_batched_tokens` so the C04a config uses a 2-knob surface without code changes. | (this commit) |
| **Joint configs for C04a + C04b** | `examples/c04a-l1-restricted/config.yaml` and `examples/c04b-l1-full/config.yaml` pin model / workload (INPUT=256/OUTPUT=20) / SLO (500 ms) / determinism (seed=0, multiprocessing_v1=False, batch_invariant=True) / hardware (1× A100 via Basilica spot). | (this commit) |

The two new YAML files land in this PR alongside the pre-reg itself
(they don't ship code — only data — so they're not a separate PR).

---

## Configuration

### Common (both C04a and C04b)

- **Model:** `meta-llama/Llama-3.1-8B-Instruct` (HF-gated; same as T-37).
- **Hardware:** 1× A100 spot via Basilica (Verda FIN-01 preferred,
  ~$0.49/hr; same hardware class as autoinfer's C02/C03 campaigns).
- **vLLM:** 0.21.0 (lock pin per PR #40; image
  `vllm/vllm-openai:v0.21.0`).
- **Workload:** synthetic random, INPUT_LEN=256, OUTPUT_LEN=20,
  MAX_MODEL_LEN=512 — matches T-37 Baseline B exactly.
- **SLO:** `harness.driver.slo_ttft_p99_ms=500`,
  `slo_tpot_p99_ms=50`, `slo_e2e_p99_ms=500` (the third one is the
  toggle for goodput-mode per T-34).
- **Determinism:** `harness.determinism.seed=0`,
  `batch_invariant=true`, `multiprocessing_v1=false`. Reference
  replica + candidate both inherit.
- **Bench seed:** `harness.driver.bench_seed=0` for the
  `vllm bench serve --seed 0` invocation (T-35).
- **Pareto axes:** `(goodput_req_per_sec, tpot_p99_ms, peak_hbm_gb)`.
- **Surrogate:** TPE with `feasibility_threshold=0.4`,
  `feasibility_k=3`, `feasibility_min_observations=4`, and T-26c's
  per-FailureKind weights derived from the catalog.
- **Operator cadence:** 8 (LLM proposes a fresh config every 8
  trials; OpenRouter Sonnet 4 as in C03).
- **Reserve cap:** `reserve_cap=4` (T-14 default; cross-layer
  stale-signal not active since C04 is L1-only).

### C04a — restricted surface

- **Catalog:** `examples/c04a-l1-restricted/knobs.yaml` (3 knobs total):
  - `max_num_seqs` ∈ {128, 256}
  - `max_num_batched_tokens` ∈ {512, 1024, 2048, 4096}
  - `enable_prefix_caching` ∈ {true, false}

  **Surface = 16 unique configs** (8 × 2). `auto_tune.sh`'s own
  surface is 8 configs (auto_tune always sets `--enable-prefix-caching`
  on; it doesn't sweep it). autoinfer's surrogate is allowed to flip
  prefix_caching as a third axis — a slight widening of the search
  space, documented here for honesty. Strictly apples-to-apples
  would pin prefix_caching=true; the YAML system's current bool
  type doesn't support a 1-value bool without a code change, so the
  surrogate gets one extra binary axis. The pre-reg's win
  condition (best autoinfer goodput ≥ 23.5 req/s) is unchanged.

  All other 10 L1 knobs are *omitted* from the restricted catalog.
  vLLM's defaults apply at startup (matches auto_tune.sh, which
  also doesn't override them):
  - `enable_chunked_prefill`: vLLM V1 default `true`.
  - `kv_cache_dtype: auto`, `dtype: auto`, `quantization: none`.
  - `attention_backend`: vLLM auto-picks (auto_tune.sh likewise).
  - `block_size: 16`, `gpu_memory_utilization: 0.9` (vLLM defaults).
  - `enforce_eager: false`, `long_prefill_token_threshold`: vLLM picks.

- **Warmstart:** `warmstart_n=4` (LLM picks a diverse corner of the
  16-config surface; hardware_notes block in the config tells the
  LLM the surface + baseline + target).
- **Max trials:** 20 (4 warmstart + 16 surrogate/operator). With 16
  unique configs, 20 trials give the surrogate ~1.25× coverage —
  enough for TPE to converge on the best corner.

### C04b — full surface

- **Catalog:** `examples/c04b-l1-full/knobs.yaml` →
  `src/autoinfer/layers/l1_engine/knobs.yaml` (12 knobs including
  T-33's two additions).
- **Warmstart:** `warmstart_n=8` (LLM warmstart proposes a
  diverse set of full-12-knob configs).
- **Max trials:** 40 (8 warmstart + 32 surrogate/operator). The
  12-knob surface has ~10^4 unique configs; 40 trials is the same
  budget C02/C03 used for L1.

### Launch commands

```bash
# C04a (no GPU until this command runs)
set -a && source .env && set +a
uv run python -m autoinfer.cli run \
    --config examples/c04a-l1-restricted/config.yaml \
    --target basilica \
    --gpu-models A100 --spot true --ttl-hours 4 \
    --artifacts-dir basilica-artifacts/c04a-<date>-<sha>

# C04b — only run AFTER C04a completes and the kept-rate is ≥30%
set -a && source .env && set +a
uv run python -m autoinfer.cli run \
    --config examples/c04b-l1-full/config.yaml \
    --target basilica \
    --gpu-models A100 --spot true --ttl-hours 6 \
    --artifacts-dir basilica-artifacts/c04b-<date>-<sha>
```

The exact CLI commands above are illustrative — `autoinfer.cli run`
is the stub at session-one and may need its CLI wired before launch.
If the CLI isn't ready by launch, fall back to
`uv run python -c "from autoinfer.builder import build_runner;
from autoinfer.config import load_config; r, _ = build_runner(
load_config(Path('...')))"`-style direct invocation per the smoke
patterns in `scripts/`.

---

## Expected timeline

| Phase | Trials | Per-trial wall | Total |
|---|---|---|---|
| C04a warmstart | 4 | ~2 min (small surface, quick startup) | ~8 min |
| C04a surrogate + operator | 16 | ~2.5 min | ~40 min |
| C04a bookkeeping (gate, summary, artifact write) | — | — | ~5 min |
| **C04a subtotal** | **20** | | **~55 min wall** |
| C04b warmstart | 8 | ~3 min (full surface, occasional fp8 infeasibles) | ~24 min |
| C04b surrogate + operator | 32 | ~3 min | ~96 min |
| C04b bookkeeping | — | — | ~10 min |
| **C04b subtotal** | **40** | | **~130 min wall** |
| **C04 total** | **60** | | **~3 h 5 min** |

Per-trial wall estimate based on C03-S's ~117 min for 60 trials
including L2/L3 (L2 trials are slower; L1-only should be faster
per trial). Each L1 trial: vllm serve startup (~30s with
`--load-format auto` for the real model) + bench (~45s for 64
prompts) + gate (~30s for 20 quality prompts) ≈ 1.5–2 min.
Adding ~30s overhead per trial for the reference-replica vllm
also running on the GPU.

---

## Expected cost

| Item | Approx. |
|---|---|
| Basilica A100 spot, C04a (~1 h) | ~$0.50 |
| Basilica A100 spot, C04b (~2.5 h) | ~$1.20 |
| OpenRouter Sonnet 4 (warmstart + operator LLM calls) | ~$0.30 |
| **Total** | **~$2.00** |

Within the recon's revised $24-47 budget for the whole proof-program
Phase 1. Cheaper than C03 (~$15) because C04 is L1-only — no L2
Basilica deploys, no L3 LLM kernel-emission calls.

---

## Expected outcomes (predictions written before data is seen)

These probabilities are honest priors. After the run, the "Outcome"
section reconciles them with the data — including cases where they
were wrong.

### Q1 / C04a — surrogate-vs-grid on the same 2-knob surface

- **Outcome A1 — surrogate wins (≥10%):** Best autoinfer goodput ≥
  23.5 req/s. *Probability:* **~30%.** Surrogate clearly beats
  grid; suggests TPE finds the right corner of the 8-config surface
  in fewer trials than auto_tune's full sweep, OR finds a config
  slightly off the grid points (e.g., max_num_batched_tokens=3072
  if the surrogate is allowed continuous values — *unlikely since
  auto_tune.sh's grid points are common-sense breakpoints*).

- **Outcome B1 — tie (±10%, i.e. 19.3–23.5 req/s):** *Probability:*
  **~50%.** Surrogate finds the same best config as the grid (auto_tune's
  256/512 = 21.39 req/s). The 8-config surface is small enough that
  any competent search method should find this point in 20 trials.
  This is the *most likely* outcome and is the methodology
  validation — surrogate works on its home turf.

- **Outcome C1 — surrogate loses (>10%, i.e. <19.3 req/s):**
  *Probability:* **~20%.** Surrogate is worse than grid on the same
  surface. **This indicates a surrogate-engine bug** — the TPE +
  feasibility classifier should never lose to a 20-trial grid sweep
  on an 8-config surface. If this happens, **do NOT launch C04b**
  until the bug is diagnosed. Most likely culprits: feasibility
  classifier over-rejecting (too tight threshold), warmstart
  picking unlucky corners, or `goodput_req_per_sec` axis routing
  broken.

### Q2 / C04b — full surface vs grid

- **Outcome A2 — wider search wins (≥10%):** Best autoinfer goodput
  ≥ 23.5 req/s. *Probability:* **~40%.** The wider 12-knob surface
  contains a recipe the 2-knob grid cannot reach. Likely
  combinations: `kv_cache_dtype=fp8` with compatible
  attention_backend × particular `block_size` × the right
  scheduling knobs. **This is the load-bearing thesis support at
  L1.** Confirms wider search beats narrower search at the
  engine-config layer.

- **Outcome B2 — wider surface no benefit (±10%):** Best autoinfer
  goodput in 19.3–23.5 req/s. *Probability:* **~40%.** Extra knobs
  produce noise, not signal, at this workload. The 2-knob surface
  was already where the goodput peak lives. *This is publishable
  nuance* — narrows the thesis claim from "joint search wins" to
  "joint search wins on workloads where the knobs interact, and
  this isn't one of them."

- **Outcome C2 — wider surface worse (>10% loss):** Best autoinfer
  goodput < 19.3 req/s. *Probability:* **~20%.** Surrogate gets
  lost in the bigger search space. Suggests insufficient trial
  budget, the feasibility classifier rejecting too aggressively
  on the wider catalog, or warmstart picks misleading the TPE.
  Diagnostic action: re-run with 2× the trial budget; if still
  losing, the surrogate's exploration policy needs work.

### Q3 — L1 surrogate kept-rate

- **Outcome A3 — ≥30%:** *Probability:* **~55%.** T-26c clears the
  target. The per-FailureKind sub-classifier sharpens each kind's
  region; the surrogate spends fewer trials on infeasible configs.
  Validates the C03-S → T-26c diagnostic.

- **Outcome B3 — 20–30%:** *Probability:* **~30%.** Better than
  T-26b's ~20% but doesn't clear the target. Opens T-26d
  (data-mined per-kind priors from accumulated history rather than
  the static `_KIND_DRIVERS` taxonomy).

- **Outcome C3 — <20%:** *Probability:* **~15%.** Regression vs
  T-26b. T-26c needs another iteration before any L1 head-to-head
  is honest. **Diagnostic priority.**

### Q4 — goodput-axis wiring

- **Outcome A4 — wired correctly:** *Probability:* **~90%.** Event
  log records `objective_axis="goodput_req_per_sec"`; per-trial
  JSON `extra["goodput_req_per_sec"]` is populated;
  `run_summary.json`'s `top_by_goodput` differs from
  `top_by_tokens_per_sec` (the SLO meaningfully bites at some
  configs).

- **Outcome B4 — partial wiring:** *Probability:* **~8%.** Event
  log records the axis but per-trial JSON is missing the field, or
  the runner's `objective_axis` doesn't propagate to the ledger.
  Bug-fix, not a thesis problem.

- **Outcome C4 — completely broken:** *Probability:* **~2%.** SLO
  isn't passed to `vllm bench serve` at all (the `--goodput` argv
  doesn't appear in the rendered command). Surprising given the
  17 T-34 tests already pass; only possible if integration with
  the basilica adapter loses the SLO somewhere.

---

## Decision tree from the data

```
Read C04a outcome first.

If C04a-Q3 says kept-rate < 20% (Outcome C3):
    → STOP. Do NOT launch C04b.
    → T-26c needs another iteration (T-26d data-mined priors,
      or a bug fix). Re-run C04a after fixing.

Else if C04a-Q1 says surrogate-loses (Outcome C1, <19.3 req/s):
    → STOP. Do NOT launch C04b.
    → Surrogate-engine bug. Diagnose: check feasibility classifier
      threshold, warmstart corner picks, goodput-axis routing.

Else if C04a is a tie or win:
    → Launch C04b.
    → After C04b, the (C04a, C04b) outcome pair selects the cell:
        (Win, Win)  → thesis support at L1; proceed to Phase 2
                       (multi-model coverage) per the proof-program
                       roadmap.
        (Win, Loss) → publishable nuance; write up as
                       "joint search wins on the same surface but
                       wider surface doesn't help at this workload."
                       Consider C04c (H100 vs Baseline C) to test
                       if hardware-class changes the wider-surface
                       payoff.
        (Tie, Win)  → wider search wins; proceed to Phase 2.
                       Note: methodology check (C04a) was a tie,
                       so the surrogate isn't *clearly* better than
                       grid on the shared surface — the thesis
                       support is conditional on wider surface.
        (Tie, Tie)  → autoinfer-L1 ties auto_tune at this workload.
                       Move to a different workload (C04c on H100
                       long-context; or Phase 2 model coverage).
        (Tie, Loss) → wider search hurts; investigate before C05.

If C04a-Q4 says goodput-axis is partial or broken:
    → Fix in a follow-up PR; do not invalidate the C04a goodput
      numbers if the per-trial extra field is the only thing missing
      (the result.txt's `best_goodput` and the bench's
      `Request goodput (req/s)` are the citable numbers).
```

The four-quadrant matrix in `c04-framing-overview-2026-05-26.md`
(PR #47) names the interpretations more concretely; this decision
tree captures the *operational* what-to-do-next branches.

---

## Optional C04c — H100 anchor (not pre-registered here)

If both C04a and C04b produce interesting results on A100, a follow-on
H100 run against Baseline C (2.97 req/s, long-context, 500ms SLO)
would add a cross-hardware data point matching the auto_tune README's
published target. **Not pre-registered in this doc** — would require
its own pre-reg PR after C04a/C04b complete, citing this campaign's
outcome. Cost estimate (per the recon addendum): ~$1.50-3 on H100
spot. Decision deferred to post-C04 analysis.

---

## What this campaign explicitly does NOT do

- **No L2 work.** C04 is single-replica L1-only. L2 topology campaigns
  belong to Phase 4 of the proof program (per the roadmap).
- **No L3 kernel work.** The v3 audit (PR-merge history before this
  session) closed out one L3 instantiation. L3 work resumes via T-21
  (attention injector) or T-32 (production-baseline kernel surface) —
  not here.
- **No speculative decoding knobs.** auto_tune.sh doesn't sweep them;
  out-of-scope for the comparable.
- **No paired-control or kernel-novelty measurement.** That was C02/C03.
- **No ShareGPT.** auto_tune.sh uses synthetic `random`; C04 follows
  (T-35 corpus pinning is unused this campaign; it stays useful for
  C05+ campaigns on real workloads).
- **No new model.** Llama-3.1-8B-Instruct only, matching T-37's
  references.

---

## Outcome (filled in after the run)

**Status:** PLANNED

### Headline numbers

(To be filled in.)

### Reconciliation with predictions

| Prediction | Actual | Match? |
|---|---|---|
| Outcome A1 (C04a surrogate wins ≥10%, P=30%) | … | yes/no |
| Outcome B1 (C04a tie ±10%, P=50%) | … | yes/no |
| Outcome C1 (C04a surrogate loses, P=20%) | … | yes/no |
| Outcome A2 (C04b wider wins, P=40%) | … | yes/no |
| Outcome B2 (C04b tie, P=40%) | … | yes/no |
| Outcome C2 (C04b wider loses, P=20%) | … | yes/no |
| Outcome A3 (kept-rate ≥30%, P=55%) | … | yes/no |
| Outcome A4 (goodput axis wired correctly, P=90%) | … | yes/no |

### What the data tells us about each Q

(To be filled in.)

### Bugs surfaced and their fixes

(To be filled in.)

### What's still open after this run

(To be filled in.)

### Cost actually spent

(To be filled in.)

### Artifacts

- `basilica-artifacts/c04a-<date>-<sha>/` (per-trial JSON,
  `events.jsonl`, `hw_context.json`, `results.tsv`,
  `run_summary.json`).
- `basilica-artifacts/c04b-<date>-<sha>/` (same shape).
- `docs/research/references/12-c04-outcome.md` (analysis writeup;
  TBD after the run).
- Closing commits: TBD.
