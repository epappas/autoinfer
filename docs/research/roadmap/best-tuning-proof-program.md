# Proof program — autoinfer as the BEST inference-tuning solution

**Status:** ADOPTED 2026-04-30 — strategic anchor for all campaigns from C04 forward.

This document operationalises the claim "autoinfer is the best
open-source inference-tuning solution" into a six-phase empirical
proof program with measurable success gates at every phase.

The program exists because the load-bearing project claim — *given a
model + hardware + SLO + traffic shape, autoinfer produces a deployable
recipe (engine config + topology + optional kernels) that is
Pareto-optimal at the joint metric* — cannot be made on assertion. It
must be earned by evidence at every (model × hardware × workload)
cell that real production traffic uses.

Every campaign from C04 forward references this document for its
strategic justification. Every campaign produces evidence for one of
the six phases below. Every phase has a decision gate that honestly
answers "is autoinfer still on track to BEST?"

---

## Operationalising "BEST"

Three claims must be measurable for the framework to credibly carry
the BEST label:

| Claim | Measurable definition | How we test it |
|---|---|---|
| **Optimality** | For every (model, hardware, SLO, traffic-shape) tuple in the supported matrix, autoinfer's recipe achieves goodput ≥ every comparable specialist tool's recipe at the same SLO with the same trial budget | Head-to-head campaigns vs `vllm/benchmarks/auto_tune`, NVIDIA Triton Model Analyzer, AutoKernel, hand-tuned managed-service public recipes |
| **Coverage** | The supported matrix spans the model families + hardware classes + workload shapes that account for ≥80% of real production inference traffic | Llama / Mistral / DeepSeek / Qwen × H100 / A100 / B200 × chat / batch / code / reasoning |
| **Reproducibility** | Every recipe carries enough metadata (git SHA, vLLM version, hardware ID, ShareGPT shard sha, seed) that a second machine reproduces the same Pareto point within ε | Phase 5 reproducibility audit; Phase 6 public-leaderboard re-runs by external contributors |

If any of the three fails, the BEST claim narrows. The phases below
trace explicit paths for each.

---

## Phase 1 — First specialist beat (C04)

**Goal.** autoinfer-L1 beats `vllm/benchmarks/auto_tune` on
(Llama-3.1-8B / 1× H100 or A100 / ShareGPT / p99 E2E < 500 ms).

**Why this first.** It's the cheapest, most-comparable test of the
joint-search-vs-line-search hypothesis. vLLM's auto_tune is the
incumbent open-source baseline at L1; if autoinfer-L1 cannot beat it,
the joint-search thesis has no L1 footing.

**Scope already pinned** in
`docs/research/notes/c04-l1-autotune-comparable-recon.md`. Pre-flight
tickets T-26c (surrogate kept-rate ≥30%), T-33–T-37 (L1 catalog gaps
+ harness hardening + baseline run) listed in `TODO.md`.

**Success gate.** autoinfer-L1 finds a recipe with **≥10% higher
goodput than auto_tune at the same SLO**, with comparable trial
budget. (10% is the threshold for "win is not measurement noise at
p99 E2E < 500 ms"; 5% would be a tie at this hardware.)

**Decision after Phase 1.**

| Result | Next action |
|---|---|
| ≥10% goodput win | Phase 2 starts. Stage-gate cleared. |
| 5–10% win | Tie. Investigate whether surrogate noise (T-26c not strong enough), catalog gap, or operator-prompt quality. One re-run after fix. |
| <5% / loss | Stop. Diagnose root cause — surrogate, catalog, hybrid policy, or thesis. No Phase 2 until C04 lands. |

**Cost / time.** ~$25-40 / ~2-3 weeks.

---

## Phase 2 — Multi-model coverage (C05-C08)

**Goal.** Prove the joint-search advantage isn't a Qwen3-8B fluke.
Same comparable harness, swept across 4 production-relevant model
families.

| Campaign | Model | Hardware | Why this one |
|---|---|---|---|
| C05 | Llama-3.1-70B | 4× H100 (TP=4) | Biggest deployed open-source model class. Exercises L2 (TP) seriously. |
| C06 | Mistral-7B + Mixtral-8x7B | 1× H100 / 2× H100 | Different attention + MLP shape; MoE adds EP knobs. |
| C07 | DeepSeek-V3 | 8× H100 (TP=8, EP=8) | MoE at scale. Tests L2's EP axis under real load. |
| C08 | Qwen-2.5-32B | 2× H100 | Dense mid-size; tests cross-Qwen-generation generalisation. |

**Success gate.** autoinfer-L1 (or L1+L2 where the model demands it)
beats `auto_tune` (or runs head-to-head against the model family's
published-baseline configs) on **≥3 of 4** model families.

**Failure semantics.**

| Result | Diagnosis | Mitigation |
|---|---|---|
| 4/4 wins | Joint search generalises across families | Proceed Phase 3 |
| 3/4 wins | One family has unusual structure (probably MoE) | Per-family surrogate prior; re-test that family |
| ≤2/4 wins | Surrogate doesn't generalise across model families | Open T-26d for per-family priors; pause Phase 3 |

**Pre-flight before C05.** Close T-12 (`max_model_len` hardcoded to
Qwen3-8B). Per-model knob catalogs for Llama-3.1, Mistral, DeepSeek,
Qwen-2.5.

**Cost / time.** ~$80-150 / ~6-8 weeks (~2 weeks per campaign,
sequential).

---

## Phase 3 — Multi-workload coverage (C09)

**Goal.** Prove the recipe shifts with workload shape — autoinfer is
not one-recipe-for-all.

Single model + hardware (e.g., Llama-3.1-8B / 1× H100), four
workload shapes:

| Workload | Distinguishing axis | Pareto-optimal target |
|---|---|---|
| Chat | Low TTFT + low TPOT, short prompts (~200 tok), short outputs (~150 tok) | Low TPOT, high request rate |
| Batch | Max throughput, no TTFT constraint, long prompts (~1k), long outputs (~500) | Max tok/s |
| Code | Long-context (8k-32k), prefix-cache-hit-heavy | Max prefix-hit-tok/s |
| Reasoning | Very long output (5k+ tokens), TPOT-dominant | Low TPOT at long output |

**Success gate.** autoinfer produces **4 distinct Pareto-optimal
recipes** that each beat a single "default" recipe on its workload's
primary metric. If recipes converge to the same config across
workloads, the framework isn't doing workload-aware tuning.

**Cost / time.** ~$50-80 / ~3-4 weeks.

---

## Phase 4 — Vs every credible alternative (C10-C12)

The adversarial phase. Direct head-to-head against every named
alternative at the layer they specialize in.

| Campaign | Comparison | Layer focus |
|---|---|---|
| C10 | autoinfer-L1+L2 vs **NVIDIA Triton Model Analyzer** on NVIDIA-stack | L1 + L2 |
| C11 | autoinfer-L3 (with T-21 attention + T-32 autotune sweep) vs **AutoKernel** on the same model | L3 |
| C12 | autoinfer-joint vs hand-tuned configs from **Together / Anyscale / Fireworks public benchmarks** | Whole-stack |

**C12 mechanics.** We can't run inside hyperscaler stacks but we can
reverse-engineer their public-benchmark recipes (Together publishes
throughput + hardware spec for popular models). Run their recipe and
autoinfer's recipe on the same hardware in the same campaign window.

**Success gate.** autoinfer wins or ties at the **joint-stack level
(C12)** — which is the structural thesis. Per-layer (C10, C11)
losses are *allowed* because specialists go deeper at single layers;
the thesis only requires joint search dominates at the joint metric.

**Pre-flight before Phase 4.**

- T-21 (attention-layer injector) implemented. Multi-day work; see
  `docs/research/notes/t-21-attention-injector-recon.md`.
- T-32 (post-emission Triton autotune sweep) implemented.
- Issue #3 (vllm-project/router) integrated — gates C10 + C12.

**Cost / time.** ~$300-500 / ~6-8 weeks.

---

## Phase 5 — Operational maturity (parallel with Phases 2-4)

Engineering work to make the framework self-hosted-product-ready.
Not a campaign; doesn't compete for GPU time.

| Item | Severity | Effort |
|---|---|---|
| Per-model knob catalogs (Llama, Mistral, DeepSeek, Qwen) — close T-12 hardcode | P0 (Phase 2 gate) | ~3-4 weeks total |
| Service API: `POST /tune {model, hardware, slo, traffic_shape, budget}` → recipe.yaml | P0 | ~2 weeks |
| CLI: `autoinfer tune --model X --slo Y --budget $Z` | P0 | ~1 week |
| Cost-bounded mode (budget cap + pre-trial cost estimate) | P1 | ~1 week |
| Deterministic reproducibility audit (same input → same recipe within ε) | P1 (Phase 5 gate for "Reproducibility" claim) | ~1 week |
| Docker container + helm chart for self-hosted deployment | P1 | ~1 week |
| Issue #3 router integration (multi-replica + PD-disagg) | P0 (gates Phase 4 C12) | ~2-3 weeks |
| **Total** | | **~10-13 weeks** but largely parallel with campaign work |

---

## Phase 6 — Public proof

The community-facing artefact. After Phases 1-5 have produced ≥6
campaigns of evidence, open the work for adversarial review.

- Open all campaign artefacts (run JSONs + Pareto plots + recipes) at
  `github.com/epappas/autoinfer-benchmarks` (or equivalent).
- Public leaderboard showing autoinfer vs alternatives across the
  (model × hardware × workload) matrix.
- Continuous re-runs against latest vLLM / new model releases (CI on
  cron).
- Reproducibility receipts — every recipe ships with the metadata
  needed for a reader to re-run the campaign on their own GPU and
  verify within ε.

**Cost / time.** ~$0 GPU + ~2 weeks engineering for the leaderboard.

---

## Risk register

| Risk | Phase | Mitigation |
|---|---|---|
| C04 doesn't beat auto_tune | 1 | Stop. Diagnose surrogate (T-26c more aggressive) vs catalog (more knobs) vs hybrid policy (different LLM operator). Re-run before any Phase 2 work. |
| Multi-model doesn't generalise | 2 | Per-family surrogate priors; LLM warmstart that takes model-family as input. Adds ~2 weeks. |
| L3 stays null even with T-21 / T-32 | 4 | Drop "joint includes L3 wins" claim; narrow value-prop to "L1+L2 joint with L3 as opt-in safety-checkable kernel rewrites." Still defensible but narrower. |
| Hyperscaler public recipes are unbeatable on the same hardware | 4 (C12) | Honest publishable result. Position autoinfer as "BEST self-hosted open-source alternative" — managed services have proprietary kernels we can't access. |
| Reproducibility breaks under hardware-class drift | 5 | Pin per-recipe `gpu_pci_id + driver_version + cuda_version`; emit hardware-equivalence-class warnings when a recipe is applied to a slightly different SKU. |
| Cost overruns | All | Per-phase cost gate; if Phase 1+2 spend exceeds 2× estimate, re-scope before Phase 3. |
| Workload generalisation fails (Phase 3) | 3 | Per-workload surrogate prior. Adds ~1 week per workload. |
| Service API takes longer than expected | 5 | Self-hosted CLI alone is acceptable for v1 product; SaaS API as v2. |

---

## Decision gates — the four moments where we honestly answer "is autoinfer BEST?"

1. **After Phase 1 (C04 result):** Is autoinfer-L1 beating vLLM
   `auto_tune` on the same target? **YES → continue.** NO → debug,
   do not proceed.
2. **After Phase 2 (C05-C08):** Does the joint-search advantage
   generalise across model families? **YES → autoinfer is BEST at
   L1 across the production model surface, citable.** NO →
   per-family priors, then re-test.
3. **After Phase 4 (C10-C12):** Does autoinfer-joint beat the best
   joint-stack alternative? **YES → autoinfer is BEST at joint-stack
   tuning, citable.** NO → narrow scope honestly (e.g., "BEST
   self-hosted joint L1+L2 tuner" without the L3 / managed-stack
   claim).
4. **After Phase 6 (public benchmarks running):** Has the community
   run autoinfer on configurations we didn't ship? **YES → market
   validation, citable.** NO → keep iterating on documentation +
   onboarding.

The four gates are sequential; each unlocks the next phase's
work. They are NOT internal-only — every gate result publishes (PR
on `main`, campaign artefact in `basilica-artifacts/`, retraction
addendum if a prior gate's reading needs correction, per the v3
audit pattern).

---

## Total budget + timeline

| Phase | Cost | Weeks (sequential) | Weeks (parallel) |
|---|---|---|---|
| 1 (C04) | $25-40 | 2-3 | 2-3 |
| 2 (C05-C08) | $80-150 | 6-8 | 6-8 |
| 3 (C09) | $50-80 | 3-4 | 3 (overlaps Phase 2) |
| 4 (C10-C12) | $300-500 | 6-8 | 6-8 |
| 5 (operational) | $0 GPU | 10-13 | 10-13 (parallel with 2-4) |
| 6 (public) | $0 GPU | 2 | 2 |
| **Total** | **~$500-800 GPU + $300-500 LLM API** | ~30 weeks sequential | **~24-26 weeks parallel-overlapped** |

That's the program. **~6-7 months focused work, ~$1000-1500 total**,
four decision gates at which we honestly answer "are we still on
track to BEST."

---

## How this document is used

Every campaign pre-registration (`docs/research/campaigns/NN-…md`)
**must** cite the Phase number from this document under "Strategic
context." Every TODO ticket that opens a campaign-prereq must cite
the Phase it serves.

This document is updated when a decision gate is reached. Updates
are commits on `main` per the same campaign-discipline rules as
campaign docs — retraction-style addendums, not rewrites of prior
text.

Cross-references:

- `docs/research/references/00-hypothesis-seed.md` — thesis, C1–C9
  claims, P1–P12 principles
- `docs/research/notes/c04-l1-autotune-comparable-recon.md` — Phase 1
  detailed recon
- `docs/research/notes/t-21-attention-injector-recon.md` — Phase 4
  L3 pre-flight
- `docs/research/raw/03-autokernel.md` — Phase 4 C11 baseline
- `docs/research/raw/07-vllm-v1-architecture.md` — Phase 1 catalog
  reference
- `TODO.md` — open tickets, including the Phase mapping in each
  ticket's description from this commit forward
