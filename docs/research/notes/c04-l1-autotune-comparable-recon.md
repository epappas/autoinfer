# Campaign 04 reconnaissance — autoinfer-L1 vs vLLM `benchmarks/auto_tune`

**Status:** in progress (live notes; reconciled into a campaign-design proposal at the end).

**Question to answer:** can autoinfer's L1 (engine-config) layer beat
vLLM's published `benchmarks/auto_tune` line-search on a directly
comparable (model, GPU, workload) tuple? If yes, by how much, and what
infrastructure has to land before the head-to-head is honest?

This document is a recon log — the pre-flight scope. End of doc has
the **decision** (write pre-reg now / write pre-reg after pre-flight
work / abandon) plus the explicit pre-flight ticket list.

**Why now (and what this campaign is NOT):** The autoinfer thesis is
**joint three-layer search Pareto-dominates per-layer specialist tools**
(L1 = engine config, L2 = topology, L3 = kernel; cf. hypothesis-seed
section "AutoKernel solves only the bottom layer. This is where
autoinfer can claim novelty — joint search across layers, not just
kernel tuning"). All three layers stay in scope from day one (P1
invariant). C04 is **the next-most-tractable per-layer comparable**, not
a redefinition of the project.

What C03 v3 actually closed out: **one specific L3 instantiation** —
single-shot Sonnet 4 Triton, rmsnorm surface, Qwen3-8B, A100, no
post-emission autotune. That's one op × one model × one hardware × one
code-emission strategy. **The L3 axis is not falsified**; T-21
(attention surface, ~70% of compute) and T-32 (stronger code model /
post-emission autotune sweep / less-optimised kernel surfaces) are open
paths with structurally different win ceilings.

What's still alive in each layer after C03:
- **L1** — direction validated (surrogate kept-rate moved 8% → ~20%
  with T-26b); needs auto_tune head-to-head (this campaign) to be
  citable against a published baseline.
- **L2** — architecture works (C03-S has L2 entries on the joint
  Pareto at 663.5 tok/s); cross-hardware + multi-replica +
  PD-disagg untested (Q3 deferred; Issue #3 router prereq).
- **L3** — one instantiation closed; multiple paths open via T-21 +
  T-32. Joint Pareto frontier from C03-S still has L3 entries
  dominating the high-throughput regime (1200-1211 tok/s).

C04 is the cheapest, most-comparable next campaign. Issue #2 already
pins the work; this recon maps it to concrete tickets. **The choice of
C04 is tactical** — it's the campaign with the lowest cost-to-evidence
ratio given the current state of the harness — not strategic. The
three-layer thesis remains the north star; C05+ is where T-21 / T-32 /
Issue #3 / Issue #1 work plays out.

---

## What I'm looking for

Two artefacts the campaign needs:

1. **A published comparable.** vLLM's `benchmarks/auto_tune/README.md`
   describes a config search that "maximises throughput s.t. p99 E2E
   < 500 ms" on a fixed (model, GPU, workload) tuple. If we run on
   the same tuple, the comparison is direct.
2. **An L1 search surface that covers the same knobs vLLM auto_tune
   searches over.** If our catalog is missing knobs the baseline
   uses, we're not comparing like-for-like.

I also want to verify our harness can:

- Drive `vllm bench serve` against ShareGPT (not just synthetic random).
- Apply a configurable `goodput` SLO threshold (`p99 E2E < 500 ms` per
  the auto_tune spec).
- Run with deterministic settings (seeded sampling, batch-invariant
  kernels, `VLLM_ENABLE_V1_MULTIPROCESSING=0`) for the reference
  replica.

---

## Recon log

### Harness state (NOT stubs)

`CLAUDE.md` lists `harness/driver.py`, `harness/gate.py`,
`harness/replica.py` as stubs. They are not. Current state:

| File | Lines | What it does |
|---|---|---|
| `harness/driver.py` | 161 | Wraps `vllm bench serve` via `build_bench_command` + `run_driver`. Returns a typed `DriverResult` with TTFT / TPOT percentiles + `goodput_req_per_sec` (falls through to `request_throughput` when no SLO supplied). Supports `random` (default), `sharegpt`, `sonnet`, `custom` datasets. |
| `harness/gate.py` | 292 | Live reference replica + per-prompt KL + batch-invariance check. P8/P9 honoured; httpx client to a real OpenAI-compatible endpoint. |
| `harness/replica.py` | 100 | FP16 `vllm serve` subprocess lifecycle. Real subprocess; no mocks. |

So the driver-side and gate-side infrastructure is in place.

### L1 catalog vs vLLM V1 live API

Our `layers/l1_engine/knobs.yaml` has 10 knobs:

```
max_num_batched_tokens, max_num_seqs, enable_chunked_prefill,
block_size, kv_cache_dtype, gpu_memory_utilization,
enable_prefix_caching, attention_backend, dtype, quantization
```

Issue #2 (and the raw note `docs/research/raw/07-vllm-v1-architecture.md`)
pins V1's live engine surface as:

| Subsystem | V1 knob | In our catalog? |
|---|---|---|
| scheduling | `max_num_seqs` | ✓ |
| scheduling | `max_num_batched_tokens` | ✓ |
| scheduling | `long_prefill_token_threshold` | **✗ MISSING** |
| scheduling | `enable_prefix_caching` | ✓ |
| memory | `gpu_memory_utilization` | ✓ |
| memory | `block_size` | ✓ |
| compile | `--enforce-eager` (disables CUDA graphs) | **✗ MISSING** |
| determinism | `VLLM_ENABLE_V1_MULTIPROCESSING=0` | partial (env var, not catalog'd) |
| determinism | batch-invariant kernels | partial (gate side, not catalog'd) |
| speculation | `speculative_config.method ∈ {ngram, eagle, medusa}` | ✗ (out-of-scope for v1) |
| speculation | `num_speculative_tokens`, `prompt_lookup_min/max` | ✗ (out-of-scope for v1) |

Three concrete gaps to close before C04 launches:

1. Add `long_prefill_token_threshold` (int knob, scheduling axis,
   coupled with `enable_chunked_prefill` + `max_num_batched_tokens`).
2. Add `enforce_eager` (bool knob, compile axis — toggles CUDA graph
   capture). Compatible with `attention_backend` selection.
3. Add a `determinism_mode` config-level sub-block (not a search
   axis — pinned for the reference replica + optionally for trials)
   that flips `VLLM_ENABLE_V1_MULTIPROCESSING=0` + sets
   `--seed N` + enables batch-invariant kernel paths if available.
   Held constant across trials; documented in `harness/gate.py` as
   the determinism configuration contract.

Speculation knobs (`speculative_config.*`) are deliberately
out-of-scope for the C04 v1 head-to-head — they're a separate axis,
the `benchmarks/auto_tune` baseline doesn't sweep them, and SMC-SD
work (Issue #1) is the appropriate place to land speculation
knobs end-to-end.

### Driver gaps for the auto_tune comparable

Driver already supports ShareGPT (`dataset_name="sharegpt"`).
Two gaps:

1. **Goodput SLO threshold not flowing through.** `DriverResult`
   exposes `goodput_req_per_sec`, but `build_bench_command` doesn't
   add `--goodput TPOT:80 TTFT:800 E2E:500` (vLLM's CLI for SLO
   thresholds). Without that, `vllm bench serve` returns
   `request_throughput` and the result's "goodput" is just the
   throughput. To match auto_tune's "max throughput s.t. p99 E2E <
   500ms" objective honestly, the driver must pass the SLO and the
   adapter must select on `goodput_req_per_sec` (which then
   accurately reflects "requests that met SLO per second").
2. **Per-trial seed + ShareGPT corpus pinning.** Auto_tune runs are
   reproducible because the ShareGPT subset is fixed and the request
   stream is deterministic. We need to pin the ShareGPT shard
   (file + sha) used in the trial and seed the request generator.
   Currently the driver is at the mercy of vLLM's internal sampling
   for ShareGPT; needs a concrete corpus pinning step.

### Surrogate kept-rate gap

C03-S's L1 surrogate kept-rate was ~20% (Q2 partial). Target for any
"we beat auto_tune" claim is **≥30% kept-rate so the surrogate
actually drives convergence within the trial budget**. T-26c
(per-FailureKind sub-classifier) is already in TODO P0 — it must
land before C04, not after.

If we ship C04 with a 20% kept-rate against vLLM's auto_tune (which
deterministically explores along its grid and never wastes a trial),
we'd lose by construction — auto_tune burns N trials, autoinfer's
surrogate burns 5N to find feasible candidates. The fair-comparison
prerequisite is the surrogate clearing 30%.

### What `benchmarks/auto_tune` actually does

Per Issue #2 and the V1 raw note: a bash script that line-sweeps
configs, running `vllm bench serve` with each, applying a
"max throughput s.t. p99 E2E < SLO" filter. The script is in
`vllm/benchmarks/auto_tune/auto_tune.sh` (mainline).

Exact comparable shape:

- Model: Llama-3.1-8B (the auto_tune README target — has to be
  this one or we're choosing a different model than the published
  baseline).
- Hardware: 1× H100 (auto_tune's published target, per Issue #2).
- Workload: ShareGPT subset (auto_tune's default).
- SLO: `p99 E2E < 500 ms`.
- Budget: comparable trial count to auto_tune's grid (estimate from
  the script — probably ~30–60 configs based on common knob
  cardinalities).

If H100 is unavailable (likely, given C03's H100 stalls), the
fallback is **1× A100 80GB** with the same model + workload + SLO
and we publish "comparable on A100" as the claim, not "comparable on
H100." Auto_tune doesn't have an H100-only requirement; the README
just shows H100 numbers because that's what the team ran.

### What we have to NOT do for the comparable to be honest

- **Don't change the workload** between the auto_tune baseline run
  and the autoinfer-L1 run. Same ShareGPT shard, same request rate,
  same num_prompts.
- **Don't tune the trial budget after seeing data.** Pre-register
  the budget in the C04 doc.
- **Don't compare against auto_tune's *headline* numbers** — they
  may have been measured on different vLLM commits, different driver
  flags. Run auto_tune ourselves on the exact same machine in the
  exact same campaign window, then compare to autoinfer-L1 from the
  same campaign window. Two numbers, one disk, one day.
- **Don't run autoinfer-L1 with paired-control L3 enabled.** C04 is
  L1-only — we already know L3 kernel-novelty at small ops is null
  (v3 audit). Including L3 just adds noise to the L1 comparison.

---

## Pre-flight ticket map (Issue #2 → TODO entries)

What must land on `main` before the C04 pre-reg can be written:

| New ticket | Description | Estimate | Blocks |
|---|---|---|---|
| **T-33** | Extend L1 catalog with V1 gaps: `long_prefill_token_threshold`, `enforce_eager`. Catalog-rule entries for both. Update `derive_knob_classes` / `derive_knob_weights` to handle the new knobs without regression. | ~1.5 h | C04 pre-reg |
| **T-34** | Driver wires `--goodput TTFT:X TPOT:Y E2E:Z` from `harness.slo_*_ms` config; adapter selects on `goodput_req_per_sec` (currently it falls through to throughput). 4-5 unit tests pin the CLI assembly + parse. | ~1 h | C04 pre-reg |
| **T-35** | ShareGPT corpus pinning (file path + sha) + per-trial seed plumbing. Driver passes `--seed` and reads ShareGPT from a campaign-pinned path. | ~1 h | C04 pre-reg |
| **T-36** | Determinism config sub-block: `harness.determinism = {seed: int, batch_invariant: bool, multiprocessing_v1: bool}`. Reference-replica-side; flowed into `replica.py` + gate. Documented in `gate.py` as the determinism contract. | ~1 h | C04 pre-reg |
| **T-37** | Run vLLM `benchmarks/auto_tune` standalone on the C04 target tuple. Record the suggested config + achieved goodput in `docs/research/raw/auto_tune-baseline-<date>.md`. This is the baseline number autoinfer-L1 will be measured against. | ~1.5 h wall + ~$3-5 GPU | C04 pre-reg |
| **T-26c** (already P0) | Per-FailureKind sub-classifier OR finer surrogate-feedback loop. Surrogate kept-rate must clear ≥30% on the C04 target before the head-to-head is honest. | open already | C04 pre-reg |

After T-33–T-37 + T-26c land, the pre-reg is writable. Pre-reg
itself is ~1 h.

### Explicitly NOT pre-flight for C04

- **Issue #3 router work.** C04 is single-replica; no router needed.
- **Issue #1 SMC-SD work.** Speculation knobs are out-of-scope for the
  L1 head-to-head; SMC-SD is its own multi-engine work.
- **Issue #4 Cloudflare Omni.** Co-tenancy axis; orthogonal.
- **T-21 (attention injector).** Multi-day; not on the C04 path.

---

## Estimated work

| Step | Effort | Cost |
|---|---|---|
| T-33 (L1 catalog gaps + tests) | ~1.5 h | $0 |
| T-34 (driver SLO threshold + selection) | ~1 h | $0 |
| T-35 (ShareGPT pinning + seed) | ~1 h | $0 |
| T-36 (determinism sub-block) | ~1 h | $0 |
| T-26c (surrogate kept-rate to ≥30%) | ~3-4 h | $0 (CPU tests) |
| T-37 (auto_tune baseline run) | ~1.5 h wall + ~30 min hands-on | ~$3-5 |
| **Pre-flight subtotal** | **~10 h hands-on** | **~$3-5** |
| C04 pre-reg writeup | ~1 h | $0 |
| C04 launch (~3-5 h wall, ~30 trials) | ~30 min hands-on | ~$15-30 |
| Outcome reconciliation + writeup | ~2 h | $0 |
| **Total to first L1 head-to-head verdict** | **~13 h hands-on** | **~$20-35** |

Comparable to the C02 + C03 envelopes individually. Multi-session;
not a same-session add-on.

### Why the impact ceiling looks promising (unlike T-21)

- vLLM's `benchmarks/auto_tune` is a **bash line-search**, not a
  surrogate or BO-driven search. autoinfer's hybrid policy
  (warmstart + Optuna TPE + Hyperband + LLM operator) has multiple
  structural advantages: warm-start prior, surrogate generalisation
  across knob combinations, fail-fast on infeasible regions.
- The L1 surface has ~10 knobs with categorical + bool + int axes —
  the kind of search where TPE is well-known to beat grid-search
  with the same trial budget by 1.2-2×.
- Even a modest +5% goodput win at the same SLO is publishable
  ("autoinfer-L1 finds a Pareto-better engine config than vLLM's
  auto_tune in N trials on Llama-3.1-8B / H100 / ShareGPT / p99
  E2E < 500 ms"). +20% would be article-grade.

This is genuinely different from T-21's outlook, where the reference
kernel is heavily optimised and the LLM-rewrite has high failure-mode
tail. Here the reference is a bash grid-search; surrogate-driven
search has clear theoretical advantages and the empirical question is
"by how much does it win," not "does it win at all."

### Why the impact ceiling has a real lower bound

If the surrogate kept-rate stays at C03-S's 20% even after T-26c, the
trial budget is largely wasted on infeasible candidates and
auto_tune's grid (which never spawns infeasible configs because it
respects catalog rules implicitly) wins by sheer hit-rate. T-26c is
the single most load-bearing pre-flight item. Without it C04 is a
coin-flip; with it, the structural advantages above carry through.

---

## Decision

**C04 is tractable, multi-session, fully scoped. Pre-reg is NOT
writable today** — pre-flight engineering (~10 h hands-on across
T-33–T-37 + T-26c) must land first so the pre-reg's "Pre-flight
changes" table cites real commits, not promises.

### Recommended sequence

1. **Open T-33 through T-37 in TODO.md as P0 (this commit).** They
   block the C04 pre-reg explicitly.
2. **Land T-26c first** (already P0; the surrogate kept-rate is
   the load-bearing pre-flight item — without ≥30% kept-rate the
   head-to-head is by-construction lost).
3. **Land T-33 + T-34 + T-35 + T-36** in two PRs — catalog gaps
   in one, harness/determinism wiring in another. Both small,
   should each take ~2 h end-to-end.
4. **Run T-37** (vLLM auto_tune baseline) on whatever H100 / A100
   capacity is available. Pin the result file's sha into the C04
   pre-reg.
5. **Write C04 pre-reg** at
   `docs/research/campaigns/04-l1-autotune-comparable-<date>.md`
   per `TEMPLATE.md`. Predictions before launch.
6. **Launch C04**, write outcome.

### What NOT to do

- **Don't write the C04 pre-reg until pre-flight engineering is on
  `main`.** The pre-reg's "Pre-flight changes" table must point to
  real commits or the discipline is broken (per repo policy
  documented in `CLAUDE.md`'s Campaign discipline section).
- **Don't run vLLM auto_tune *and* autoinfer-L1 in different
  weeks / different commits.** Same machine, same window, same
  vLLM commit, same workload — back-to-back. Otherwise drift in
  GPU-clock state, kernel cache, ShareGPT shard contents, etc.,
  contaminates the comparison.
- **Don't include L3 paired-control in C04.** It's L1-only. L3 at
  small ops is closed-out by C03 v3.
- **Don't sweep speculation knobs in C04.** They belong to Issue #1
  / SMC-SD work; auto_tune doesn't sweep them either, so out-of-scope
  for fair comparison.

---

## TODO updates this recon implies

Tickets to open as P0, blocking the C04 pre-reg:

- **T-33** — extend L1 catalog with V1 gaps (`long_prefill_token_threshold`, `enforce_eager`)
- **T-34** — driver passes `--goodput` SLO; adapter selects on `goodput_req_per_sec`
- **T-35** — ShareGPT corpus pinning + per-trial seed
- **T-36** — `harness.determinism` sub-block (seed, batch-invariant, V1-multiprocessing)
- **T-37** — vLLM `benchmarks/auto_tune` baseline run on the C04 target

T-26c stays P0 (already there). It's the single most important pre-flight
ticket — the structural-advantage argument depends on the
surrogate clearing 30% kept-rate.

T-32 (production-baseline gap from v3 audit) stays P0; orthogonal to
C04 (which is L1, not L3) but reminds us C04 must NOT silently slip
in any kernel-level claim.

### Time spent on this recon

~30 minutes. Within the 30-min reconnaissance budget; mirrors the
T-21 recon's discipline — recon doc, decision, ticket list, no
half-built campaign.
