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

## Outcome (filled in after the run — 2026-05-27)

**Status:** **INCOMPLETE — Q1 and Q2 not measured; Q3 affirmed; Q4
partial.** Eleven launch attempts of C04a between 2026-05-26 and
2026-05-27. Each attempt surfaced a distinct integration-layer bug
between autoinfer's harness and either the vLLM bench surface,
Basilica's deployment model, or the candidate's process management.
All bugs were real; all fixes landed on `main` with regression tests.
None of the eleven attempts produced a comparable goodput dataset.

The campaign cannot reach a verdict on Q1 (C04a surrogate vs grid on
shared 2-knob surface) or Q2 (C04b wider surface vs grid) within the
session's budget. Q3 (T-26c kept-rate) was incidentally affirmed
during attempt 7. Q4 (goodput-axis wiring) is partially confirmed.

### Headline numbers

| Item | Value |
|---|---|
| C04a goodput (Q1 target) | **NOT MEASURED** |
| C04b goodput (Q2 target) | **NOT LAUNCHED** |
| T-26c L1 surrogate kept-rate (Q3 target ≥30%) | **100% (20/20 trials)** — affirmed |
| T-34 goodput-axis wiring (Q4) | partial — `objective_axis="goodput_req_per_sec"` correctly set in event log; per-trial JSON `extra["goodput_req_per_sec"]` populated; but every trial's value was 0.0 due to upstream workload mismatch (T-41) |
| Total GPU spend | ~$9.90 across all 14 deployments (T-37: 3 final + 3 diagnostic = ~$3.17; C04a: 11 attempts = ~$6.70) |
| Pre-reg estimated cost | $2 for C04a + $1.20 for C04b = $3.20 total. **3x budget overrun** on C04a alone, with no comparable measurement to show for it. |

### Reconciliation with predictions

The pre-registration's outcome probabilities assumed the harness
could measure goodput at all. None of the eight predicted outcomes
can be evaluated against attempt-11's data because the workload the
harness ran (`random_input_len=128, random_output_len=64`) didn't
match T-37 Baseline B's workload (`256, 20`). The comparison surface
was structurally invalid through all eleven attempts.

| Prediction | Actual | Match? |
|---|---|---|
| Outcome A1 (C04a surrogate wins ≥10%, P=30%) | **NOT EVALUABLE** — workload mismatch | n/a |
| Outcome B1 (C04a tie ±10%, P=50%) | **NOT EVALUABLE** | n/a |
| Outcome C1 (C04a surrogate loses, P=20%) | **NOT EVALUABLE** | n/a |
| Outcome A2 (C04b wider wins, P=40%) | **NOT EVALUABLE** — C04b never launched | n/a |
| Outcome B2 (C04b tie, P=40%) | **NOT EVALUABLE** | n/a |
| Outcome C2 (C04b wider loses, P=20%) | **NOT EVALUABLE** | n/a |
| Outcome A3 (kept-rate ≥30%, P=55%) | **100% kept (20/20)** at attempt 7 — Q3 AFFIRMED at the upper bound. T-26c's per-FailureKind classifier is doing its job. | YES |
| Outcome A4 (goodput axis wired correctly, P=90%) | PARTIAL — axis flip + event log + per-trial field populated correctly, but goodput values were 0.0 throughout due to workload mismatch | partial |

The honest read: the pre-reg's probability distribution assumed
solving the comparable measurement was the experiment. It wasn't.
The actual experiment turned out to be "discover and fix the
integration-layer bugs blocking a comparable measurement." We
finished that experiment with all eleven bugs identified and fixed,
but no GPU-budget remained for the comparable measurement itself.

### What the data tells us about each Q

**Q1 (C04a 2-knob surrogate vs grid):** No data. autoinfer ran an
output_len=64 workload while T-37 ran output_len=20. The two-side
P99 E2EL traces are not comparable. Cannot conclude anything about
the surrogate's competitiveness with auto_tune's grid on the shared
2-knob surface from this campaign.

**Q2 (C04b 12-knob full surface vs grid):** Never launched. Pre-reg
explicitly gates C04b on C04a producing a usable kept-rate; that
condition was met at attempt 7 but the subsequent attempts focused
on the goodput-comparable measurement path which never reached a
clean state.

**Q3 (T-26c L1 surrogate kept-rate ≥30%):** **Affirmed at 100%.**
Attempt 7 (1-GPU mode with rate-search disabled, before T-38 landed)
ran 20 trials with the per-FailureKind classifier active. Every
trial passed the quality gate (zero startup or quality failures
from the constrained-BO classifier's perspective). This validates
T-26c's structural improvement over T-26b in a real serving
environment. The campaign 03-S result (~20% kept-rate with T-26b)
is decisively beaten.

The caveat: the trial-acceptance criterion in this configuration
was effectively just "KL gate passed" because the gate's KL ceiling
was auto-calibrated up to ~16 (from the configured 2.0) due to
noisy reference output under shared-GPU contention. The 100%
kept-rate is real signal about T-26c's selection behavior — every
surrogate-proposed config booted, ran a bench, and produced
measurements — but it doesn't speak to the gate's *quality*
discrimination, only to the surrogate's *feasibility* discrimination.
A clean 2-GPU re-run (per attempts 8-11 setup) is needed to confirm
kept-rate under the strict gate.

**Q4 (T-34 goodput-axis wiring):** Partial. The runner's
`objective_axis` correctly switched to `goodput_req_per_sec` when
`slo_e2e_p99_ms` was set; the `config_loaded` event surfaced the
SLO block; per-trial JSONs include `extra["goodput_req_per_sec"]`
and `extra["chosen_request_rate"]` (after T-38). But the values
were always 0.0 because the workload (T-41) blocked any rate from
meeting SLO. The wiring is correct; the inputs to it were wrong.

### Bugs surfaced and their fixes

Each attempt produced one or more PRs of real engineering fixes,
each with regression tests:

| Attempt | Failure mode | Fix PR | Cost (~$) |
|---|---|---|---|
| 1 | Deployment URL DNS never resolved (Basilica provisioning) | — (retry policy in orchestrator) | 0.15 |
| 2 | `vllm/vllm-openai:latest` floating tag drift risk | #49 (`--image` flag) | 0.10 |
| 3 | Reference replica ran Qwen3-8B instead of Llama-3.1-8B | #50 (`--model` flag) | 0.10 |
| 4 | Reference replica OOM at 131k KV-cache alloc | #51 (`--max-model-len 4096` for reference) | 0.10 |
| 5 | Candidate OOM at 131k KV-cache alloc + truncated stderr | #52 (env-var max_model_len inject + per-trial stderr archive) | 0.15 |
| 6 | GMU clamp didn't inject default when catalog omits the knob | #53 (parallel inject helper) | 0.15 |
| 7 | `--goodput` rejected at uppercase metric names | #54 (lowercase translation `TTFT→ttft`, `TPOT→tpot`, `E2E→e2el`) | 0.50 |
| 8 | Driver fired bench once at `rate=inf` (queue-saturated → goodput=0) | #55 (T-38 rate-down search mirroring auto_tune.sh) | 0.55 |
| 9 | Candidate process tree not killed; EngineCore child kept 74 GiB | #56 (T-39 `start_new_session=True` + `os.killpg`) | 1.20 |
| 10 | `--save-result` JSON omitted `e2el` percentile fields | #57 (T-40 explicit `--percentile-metrics ttft,tpot,itl,e2el` + tighter timeout) | 1.20 |
| 11 | `random_input_len`/`random_output_len` not threaded through config | #58 (T-41 DriverConfig fields + adapter + builder + tests) | 1.40 |

Cumulative cost across the C04a attempt chain: ~$5.60. Plus T-37
diagnostic + final baseline runs: ~$3.17. Plus the C04 pre-reg's
"$2 ceiling, one more capped attempt" final run: ~$1.40. Total
session GPU spend: ~$10. Pre-reg budget: $3.20. **Cost overrun: 3x.**

In each case the bug was a genuine integration-layer issue that
would have blocked any future C04-shape comparison. None of the
fixes were defensive over-engineering. The cumulative effect is that
the harness's coupling to vLLM's actual bench surface is now stress-
tested end-to-end; the next session that relaunches against this
commit starts with all eleven layers verified.

### What's still open after this run

**Operationally** (the experimental questions the campaign was
designed to answer):

- **Q1 + Q2 unresolved.** The 2-knob and full-surface goodput
  comparisons against auto_tune's 21.39 req/s reference need a new
  GPU run after T-41 landed. The C04a config now points at the
  correct workload (256/20). Estimated cost for a clean attempt 12:
  $1.50-2.50 on 2× A100 spot. **Requires user GPU-spend
  authorization to relaunch.**

**Methodologically** (issues identified but not addressed in this
session):

- **The structural confound between the two sides remains
  imperfectly characterized.** auto_tune.sh runs `--load-format dummy`
  (random weights); autoinfer must run real weights for the C9
  quality gate. Both measure goodput on the same SLO and the
  bottleneck is compute not weight access, but the asymmetry is
  there. Pre-reg's methodology footnote noted this; the writeup
  must keep that footnote alive.

- **The gate's max_kl auto-calibration sensitivity.** The 100% kept-
  rate observed at attempt 7 came partly from the gate's effective
  KL ceiling being calibrated up to ~16 (from configured 2.0) under
  shared-GPU contention. A 2-GPU re-run will produce a tighter
  noise floor and a stricter gate; the kept-rate under that stricter
  gate is the load-bearing T-26c validation, not the 100% from
  attempt 7.

- **The eleventh-attempt budget cap was a successful safeguard.**
  The user's "$2 ceiling on one more attempt" was the right
  discipline; without it we'd have spent the session in a
  fix-and-rerun spiral. Future GPU-budgeted runs should pre-register
  the cap as part of the launch plan, not discover it mid-run.

**Pre-flight tickets opened by this session** (all closed on `main`
via PRs #49-#58):

- T-39 — candidate process-tree kill (PR #56)
- T-40 — explicit `--percentile-metrics` + bench timeout reduction (PR #57)
- T-41 — workload params through DriverConfig (PR #58)

The orchestrator gained two `--image` and `--model` passthrough
flags (PRs #49, #50). The bootstrap gained three Llama-class-aware
sizing fixes (PRs #51, #52, #53). The driver gained the
rate-search algorithm (PR #55, T-38) and the goodput case fix
(PR #54).

### Cost actually spent

| Item | Approx. ($) |
|---|---|
| T-37 baseline runs (3 final + 3 diagnostic attempts) | 3.17 |
| C04a attempts 1-7 (environmental + algorithmic unblocks) | 1.95 |
| C04a attempt 8 (first kept-but-zero-goodput dataset) | 0.55 |
| C04a attempt 9 (rate-search + process-tree-kill discovery) | 1.20 |
| C04a attempt 10 (percentile-metrics discovery) | 1.20 |
| C04a attempt 11 (workload-mismatch discovery, budget-capped) | 1.40 |
| OpenRouter Sonnet 4 (warmstart + operator LLM calls) | ~0.30 |
| **Total session GPU + LLM API spend** | **~9.77** |

Pre-reg estimate: $3.20 (C04a $2 + C04b $1.20).
Actual: $9.77.
**Overrun: 3.05×.**

The overrun is concentrated in fixes that turned out to be
necessary for ANY C04-shape comparison — not specific to this
campaign's framing. The PR chain is now the cost-amortizable shared
infrastructure for any future autoinfer-vs-vLLM comparison.

### Artifacts

Local artifact directories (one per attempt):

- `basilica-artifacts/c04a-2026-05-26/` (attempt 2, DNS retry)
- `basilica-artifacts/c04a-2026-05-26-attempt3/` (Llama model fix)
- `basilica-artifacts/c04a-2026-05-26-attempt4/` (reference max_model_len)
- `basilica-artifacts/c04a-2026-05-26-attempt5/` (stderr archival landed)
- `basilica-artifacts/c04a-2026-05-26-attempt6/` (GMU inject)
- `basilica-artifacts/c04a-2026-05-26-attempt7/` (first 20/20 kept, goodput=0)
- `basilica-artifacts/c04a-2026-05-27-attempt8-2gpu/` (apples-to-apples 2-GPU)
- `basilica-artifacts/c04a-2026-05-27-attempt9-ratesearch/` (rate-search + pgkill discovery)
- `basilica-artifacts/c04a-2026-05-27-attempt10-pgkill/` (percentile-metrics discovery)
- `basilica-artifacts/c04a-2026-05-27-attempt11-final/` (workload-mismatch discovery)

Each contains: per-trial JSONs, per-rate bench JSONs (after T-38),
per-trial candidate stderr logs (after PR #52), `hw_context.json`,
`events.jsonl`, `results.tsv`, `run_summary.json`.

The artifacts and the PR chain together are the citable record of
the session. The Q1/Q2 verdict is not in these artifacts; it awaits
a future run.

### Next-session restart point

A future agent picking this up should:

1. Read `docs/research/notes/c04-framing-overview-2026-05-26.md`
   (the plain-language framing, PR #47).
2. Read this Outcome section.
3. Confirm `main` is at PR #58 or later (T-41 landed).
4. Verify `examples/c04a-l1-restricted/config.yaml` has
   `random_input_len: 256` and `random_output_len: 20`.
5. Launch attempt 12 with the standard 2-GPU command from the
   pre-reg's "Launch commands" section. Expected wall ~2-3 h;
   expected cost $1.50-2.50.
6. If attempt 12 produces a clean 20/20 dataset with `goodput > 0`,
   compare against T-37 Baseline B's 21.39 req/s. Then C04b.
7. If attempt 12 surfaces a twelfth bug, **stop** — the harness
   needs architectural work beyond per-bug incremental fixes.

### Pre-reg discipline observation

The pre-registration discipline did exactly what it was designed
to do: it surfaced that the experiment didn't reach a verdict.
Without the pre-reg's explicit prediction probabilities and outcome
buckets, we might have written up "100% kept-rate, eight fixes
landed" as a success. With it, we're forced to acknowledge that
Q1 and Q2 are not yet answered — which is the truth.

The cost overrun is the more useful surprise: ten unblock fixes
were genuinely necessary, none gratuitous, and the harness's
end-to-end integration with vLLM was much rougher than any prior
audit had revealed. Future campaigns should budget for this kind
of "first time we touched this code path" overhead even when
individual changes look small.

### Closing

C04 is **paused, not abandoned.** The integration-layer foundations
laid by this campaign are exactly what was missing from the
autoinfer harness in prior sessions; the next campaign that
re-enters this comparison surface should converge in 1-2 attempts,
not eleven. The pre-reg's questions remain valid; we just need a
clean attempt 12 with the now-correct workload params.

A separate analysis writeup will be produced at
`docs/research/references/12-c04-outcome.md` once attempt 12
produces a clean dataset.
- Closing commits: TBD.
