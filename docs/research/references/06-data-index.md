# Autoinfer iteration-zero data index

One-stop index to every campaign we ran, what it proves, what it
doesn't, and where the raw data lives. This is the "notes gathered"
reference. The detail lives in the per-run docs (`01`–`05`); this doc
just maps what we have so we can decide what to build next.

## Campaigns

| # | Date | Layer | Trials | Kept | Pareto | Raw dir | Analysis | Headline result |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-04-23 | L1 | 5 | 1 | 1 | `iteration-zero-smoke-run-2026-04-23/` | `01-iteration-zero-results.md` | Substrate validated; 239 tok/s on random workload |
| 2 | 2026-04-23 | L1 | 10 | 3 | 2 | `iteration-zero-10trial-run-2026-04-23/` | `02-iteration-zero-10trial-results.md` | **870 tok/s** — FLASHINFER vLLM defaults (highest single number across iter-0) |
| 3 | 2026-04-23 | L1 | 20 | 3 | 2 | `iteration-zero-llm-20trial-2026-04-24/` | `03-iteration-zero-llm-20trial-results.md` | LLM warmstart undershot without hardware context (735 tok/s) |
| 4 | 2026-04-24 | L1 | 50 | 9 | 4 | `iteration-zero-llm-50trial-2026-04-24/` | `04-iteration-zero-50trial-final-analysis.md` | Operator hit rate 75% vs Optuna 0%; best 821 tok/s |
| 5 | 2026-04-24 | L2 (partial) | 3 | 1 | 1 | `iteration-zero-l2-3trial-2026-04-24/` | (partial) | First A100 datapoint (744 tok/s) |
| 6 | 2026-04-24 | L2 | 5 | 3 | 2 | `iteration-zero-l2-5trial-2026-04-24/` | `05-iteration-zero-l2-first-data.md` | **H100 895 tok/s / 16 ms TPOT** — first real cross-GPU data |

85 L1 trials + 8 L2 trials + 5 Pareto entries across campaigns.

## What each campaign empirically shows

- L1 knob ablations on RTX A6000: FLASHINFER ~2× FLASH_ATTN baseline;
  prefix caching recovers most of the FLASHINFER gain on FLASH_ATTN;
  bfloat16 + larger block_size is a valid novel region.
- L1 policy comparison: deterministic peak (870) > LLM peak (821) on
  absolute tok/s at iter-zero scope; LLM-operator produces diverse
  Pareto frontier; LLM warmstart needs explicit hardware context.
- L2 cross-GPU: H100 +20% tok/s and 3.4× lower TPOT vs A100; RTX
  A6000 spot availability flaky at gpu_count=1. L2 best (H100, 895)
  beats L1 best (A6000, 870) — hardware class swamps knob tuning.
- Self-kl calibration: eliminates quality_kl false positives from
  concurrent-query scheduling noise. Zero false positives in the
  50-trial and L2 campaigns after enabling.

## What the data does NOT yet show

- **L1 × L2 joint search:** no campaign has varied hardware AND
  engine-config knobs together. The best L1 config on A6000 may not
  be the best L1 config on H100 (FLASHINFER on Hopper might gain more
  from FP8 KV, unreachable on Ampere).
- **L3 kernel search:** unimplemented. The knob surface for kernel
  replacement inside vLLM custom ops doesn't exist yet.
- **Cross-layer stale-signal invalidation actually firing:** the code
  exists (`Ledger.mark_stale`, `LayerScheduler.propagate_finding`) but
  no run has triggered it. We haven't measured whether a finding at
  one layer actually invalidates a cached finding at another —
  thesis's load-bearing claim remains untested.
- **Replication across models:** all campaigns use Qwen/Qwen3-8B. No
  data on whether findings transfer to Llama-3.1-8B or Covenant-72B.
- **Workload diversity:** ShareGPT only. RAG, code-completion, agent
  scratchpads would shift the frontier.

## What can be claimed today vs what requires more build

### Today (replication / tooling)

- "Open-source three-layer search framework for LLM inference."
- "Replicated H100 > A100 > A6000 ordering on Qwen3-8B ShareGPT with
  specific multipliers."
- "Self-kl reference calibration as a denoising technique for
  concurrent-query quality gates."
- "Hybrid LLM-operator + Optuna surrogate: operator produces 75% hit
  rate, surrogate 0% when infeasible regions dominate."

### Requires L3 + joint campaign (potential novelty)

- **"First demonstration of cross-layer stale-signal invalidation in
  inference-engine optimization"** — requires a run where a kernel
  win invalidates a cached engine config and the re-search produces
  a measurable Pareto improvement.
- **"Joint L1×L2×L3 search finds configurations that no single-layer
  search explores"** — requires the joint campaign to produce a
  Pareto entry that dominates every single-layer winner, with a
  config that combines knobs across all three layers.
- **"Hardware-class-specific optimal engine configs"** — requires
  showing the best L1 config differs materially between H100 and
  A6000. (L2 run touched this; joint run would confirm it with
  enough samples.)

## Data files reference

All trial-level data is in `docs/research/raw/iteration-zero-*/`
directories. Each has:

- `l1_engine_*.json` or `l2_topology_*.json` — per-trial JSONs with
  full config + measurement xor typed FailureRecord + any extras.
- `*_bench.json` — vLLM bench raw output per trial.
- `events.jsonl` (when fetched successfully) — streaming event log.
- `results.tsv` — one-row-per-trial flat schema for plotting.
- `run_summary.json` — aggregate (kept/failed/Pareto/phase counts).
- `hw_context.json` — GPU/driver/version snapshot at run start.
- `orchestrator.log` — dev-side orchestrator log including stream'd
  deployment logs.

Reproduce any plot or number via `scripts/analyze_run.py <dir>`.

## Build plan going forward (thesis-grade)

Three pieces to land, in order of cost:

1. **Cross-layer scheduler wiring** (smallest). Currently
   `LayerScheduler.propagate_finding` exists but is never called by
   `ContinuousRunner`. Wire it so that on every new measurement that
   enters the Pareto frontier, `propagate_finding` is called for
   layers below (kernel findings invalidate engine-config findings;
   topology findings invalidate engine-config findings). Add an
   `L2-triggers-L1-stale` test and then empirically observe it in a
   joint run.

2. **L3 adapter** (biggest). LLM-driven kernel search inside vLLM
   custom ops. Minimum viable: swap ONE kernel (probably RMSNorm or
   RoPE — smallest blast radius) via a Triton rewrite proposed by
   the LLM, compile with vLLM's custom-op registry, run through the
   L1/L2 harness. Reuses most of the existing code. Hard parts:
   (a) kernel-level correctness test, (b) compile-on-the-fly for
   vLLM, (c) harness that isolates the kernel-level win from
   engine-level confounders.

3. **Joint L1×L2×L3 campaign** (depends on #1 and #2). 50-100 trials
   with all three layers enabled. Budget ~$100-200 on Basilica plus
   LLM costs. Goal: at least one Pareto entry that combines knobs
   from multiple layers + at least one observed stale-signal
   propagation that improves the frontier.

Success criterion for the thesis claim: the joint run produces a
Pareto entry that strictly dominates every single-layer run's best,
AND the cross-layer stale-signal fires at least once with an
observable effect on the ledger.
