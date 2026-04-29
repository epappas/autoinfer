# autoinfer — open backlog

**Single source of truth** for open tasks, tracked corner-cuts, and
open research questions. Updated as part of every commit that opens
or closes an item. Commit references in `[…]` brackets.

Bands by priority:
- **P0** — blocking the next thesis-grade campaign or the project's
  load-bearing claim. Fix before running anything new.
- **P1** — solid corner-cuts that affect data quality or
  reproducibility but don't gate the next campaign. Fix before any
  article-style writeup.
- **P2** — research extensions and architecture work that would
  change what we *can* measure, not what we *do* measure today.

---

## Open

### P0 — blocks next campaign

| ID | Item | Why it blocks | Reference |
|---|---|---|---|
| T-26c | Per-FailureKind sub-classifiers OR finer-grained surrogate feedback | Campaign 03-S confirmed T-26b's per-knob weights raised L1 surrogate kept-rate from C02's 8% to ~20%, but didn't clear the Q2 30% target. Same diagnosis as C02: surrogate keeps proposing fp8 variants on A100 even after seeing fp8 failures because the failure-region geometry isn't captured by per-knob weights alone. | `policy/feasibility.py` |

### P1 — corner-cuts; must fix before article

| ID | Item | Why it matters | Reference |
|---|---|---|---|
| T-26-followup | Replication study for the rmsnorm/fp16/large +14.7% NOV win | Campaign 02 had N=1 paired observation per cell. The +14.7% NOV win at rmsnorm/fp16/large is robust enough to call (above the 5% pre-registered threshold) but needs N≥3 paired obs at that same cell + at least one H100 datapoint for hardware-class generality before the result publishes. | next campaign pre-registration |
| T-29 | Paired-control prompt robustness — split into per-cell sequential calls OR tighten delimited-block grammar | Campaign 02: 2 of 6 paired-control NOV halves broke in the silu_mul block (w0007 startup-fail, w0009 quality_kl-fail) = 33% breakage vs target ≤ 17%. Both failures clustered in silu_mul, suggesting either silu_mul prompts are harder for the LLM or response quality degrades after the rmsnorm blocks. | `layers/l3_kernel/proposer.py:propose_for_cells` |
| T-10 | Top-K Jensen-Shannon (`topk_js_divergence`) wired into `run_gate` in place of top-K KL | Three calibration band-aids ride on the broken-metric stack (median-cap → noise-floor → one-way valve). JS is bounded, no floor amplification, would collapse all three band-aids into one principled metric | `harness/gate.py:topk_kl_divergence` shipped, JS shipped at [`803c1f9`] but never wired |
| T-12 | Catalog constraint hardcoded to Qwen3-8B `max_model_len=32768` | Switching models silently breaks the `chunked_prefill_batched_tokens_bound` rule. Should be parameterised over the configured model's actual max length | `layers/l1_engine/knobs.yaml:chunked_prefill_batched_tokens_bound` |
| T-14 | `reserve_cap=4` picked arbitrarily, not informed by data | No principled argument for 4 vs 2 vs 8. Should be informed by observed surrogate-improvement rate per re-explored trial after a few campaigns generate data | `controller/stale.py:LayerSpec.reserve_cap` |
| T-15 | Hardware notes hand-authored as prose (FP8-on-A100, chunked_prefill rules) | Exactly the manual-rule-authoring anti-pattern that the FeasibilityModel was built to retire. Once the classifier accumulates ~50 trials of failure data, the prose notes should be progressively trimmed and the data-driven path validated end-to-end | `examples/qwen3-8b-l1-l2-l3-joint/config.yaml` `hardware_notes:` |
| T-16 | `OptunaSurrogate._penalty_for_failure` ignores FailureKind | All typed failures earn the same scalar penalty. The `ConstrainedOptunaSurrogate` uses FailureKind via the FeasibilityModel, but the perf model still treats OOM and QUALITY_KL as identical signals to TPE's KDE | `policy/surrogate.py:_penalty_for_failure` |

### P2 — research extensions

| ID | Item | Description | Reference |
|---|---|---|---|
| T-20 | RoPE support in injector | vLLM's `RotaryEmbedding` API has many shape-varying paths (positions, query/key fused, kv-cache aware). v1 injector explicitly skips it; v2 needs to either fan out per shape variant or accept a kernel signature richer than the rmsnorm/silu_mul cases | `layers/l3_kernel/injector.py:_TARGET_BINDINGS` |
| T-21 | Attention-layer injector (**escalated from P2 → P1 after campaign 01**). Multi-day work, not a same-session task — see recon for scope. | RMSNorm + SiluAndMul are tiny fractions of total compute on Qwen3-8B; campaign 01 confirmed all L3 KEPT trials cluster in 515-621 tok/s regardless of LLM-novel-vs-reference. The actual kernel-level wins live in attention — without an attention injector, L3 can't show end-to-end wins on this model | recon: `docs/research/notes/t-21-attention-injector-recon.md` |
| T-28 | Pre-registration of T-21 attention-injector campaign (per TEMPLATE.md) | T-21 implementation must not start without a pre-registered campaign doc that pins the config subset (no FP8 / sliding window / sinks for v1) and Q1/Q2/Q3 questions. Avoids the half-built injector failure mode flagged in the recon. | depends on T-21 |
| T-23 | L2 `peak_hbm_gb` reads campaign-container nvidia-smi for remote candidates | The L2 adapter reports local GPU memory even when the candidate is on a remote H100. Real per-trial HBM needs to be queried inside the remote deployment | `layers/l2_topology/adapter.py` |
| T-24 | Stale Basilica deployments from prior sessions | 11 deployments going back to 2026-03-16 listed during the first joint campaign. May be the user's other projects; not deleted unilaterally. Needs user triage | flagged in joint-run analyses |

---

## Closed

| ID | Item | Closed by | Notes |
|---|---|---|---|
| T-01 | Verify smoke 4 KernelProposer used novel source | (this commit) | Audit confirmed: smoke 4 sources sha `034fc4b4...` (rmsnorm) and `d1407aa1...` (silu_mul) differ from reference shas `3e8a82bf...` and `69cf8186...`. Both sources contain `@triton.jit` decorators with names `*_triton_kernel`. The LLM proposer IS generating Triton source. |
| T-02 | Kernel source hash on every L3 trial Measurement | (this commit) | `_kernel_source_metadata()` in `layers/l3_kernel/vllm_adapter.py` + same in standalone `adapter.py`. Records `kernel_source_sha_int` (int of first 12 hex chars) and `kernel_is_reference` (1.0/0.0) in `Measurement.extra` so post-run analysis can group novel vs fallback. |
| T-17 | KernelProposer fallback transparency | (this commit) | `[autoinfer.l3.proposer] fallback to reference seeds` marker prints to stdout when LLM returns unparseable blocks; captured in basilica logs + per-trial `_vllm.out` files alongside the injector's `[autoinfer.l3.injector]` marker. |
| T-11 | L3 standalone perf timing uses `time.perf_counter` without `torch.cuda.synchronize()` | (this commit) | `_best_elapsed_cuda` uses `torch.cuda.Event` + `synchronize()` when inputs are CUDA tensors; `_best_elapsed_wall` keeps the wall-clock path for CPU. The vLLM-mode L3 adapter uses `vllm bench serve` which has its own correct timing — this fix matters for the standalone L3 path used in dev workflows and on a real GPU. |
| T-13 | `compile_candidate` temp-file leak | (this commit) | `_l3_temp_root()` returns a process-shared temp dir created on first use; `atexit.register` deletes it on process exit. All compiled candidates write into this shared dir, so `/tmp` no longer accumulates stale `_l3_*.py` files across trials. Test added in `test_l3_surface.py` to pin the shared-dir behavior. |
| T-22 | per-layer Pareto + best in `run_summary.json` | (this commit) | Adds `best_by_layer`, `pareto_frontier` (serialised), `n_kept_by_layer`, `n_failed_by_layer` to the summary so downstream tools (plots, articles, the analyzer) read these directly without re-loading trial JSONs. 2 unit tests pin the structure. |
| T-25 | Per-prompt KL distribution shape in trial JSON | (this commit) | `_kl_percentiles` adds `kl_min`, `kl_p50`, `kl_p90`, `kl_p95`, `kl_p99` to `Measurement.extra` for both L1 (compose_measurement) and L3-vLLM. Lets post-run analysis distinguish "low-mean, no outliers" from "low-mean, one bad prompt" — the gate's scalar masks the latter. |
| T-26 | FeasibilityModel feature engineering: knob-class one-hot for "fp8 variants" / "Hopper-only KV" / "attention-backend × KV-format compat" | (this commit) | `_knob_distance` and `_config_distance` now accept a `class_map` / `knob_classes` arg; values mapped to the same class collapse to distance 0 within that knob. `derive_knob_classes(catalog)` builds the taxonomy from compatibility rules' `when_values` lists, so a single fp8 failure generalises across `{fp8, fp8_e4m3, fp8_e5m2}`. Tests pin the campaign-01 counterfactual: legacy classifier predicts 0.41 for an unseen fp8 variant; class-aware predicts ~0.0. Wired through `_build_surrogate(knob_classes=...)` for L1 only — L2/L3 keep the legacy fallback until they have constraint rules of their own. |
| T-27 | Same-config L3 control trials: pin (op, regime, dtype, hardware) and run reference + novel back-to-back | (this commit) | New `PairedControlProposer` wraps `KernelProposer` and emits 2N interleaved (reference, llm-novel) configs at identical (target_op, dtype, shape_regime) cells. `KernelProposer.propose_for_cells(cells)` asks the LLM to produce one kernel per pinned cell and force-overrides the cell triple onto the returned config so any LLM-side drift is normalised — invariant: the two configs in each pair always share the same (op, dtype, regime). Wired via new `cfg.layers.l3_kernel.paired_control: bool = False` flag (default off, backwards-compat); when on, the builder bumps warmstart_n to a multiple of 2 so each ref has its novel partner inside the warmstart batch. 12 new tests pin: order, drift override, op-only fallback, reference-twice degeneracy when LLM produces no block, odd-n truncation, seed cycling. |
| Campaign 01 | Full L1×L2×L3 with mode='vllm', constrained surrogate, kernel-source telemetry — first production run of the full stack | run completed `a4cf34e..`; results at `docs/research/references/11-campaign01-results.md` | 36 trials in 115 min, ~$15-20. L2 H100 wins joint Pareto (757.6 tok/s, 17.4ms TPOT). Q1 inconclusive at rmsnorm/silu_mul surface (Outcome B); Q2 partial (8% vs 0% baseline); Q3 negative (gated by Q2). Opened T-26, T-27; escalated T-21. |
| Campaign 02 | Paired-control L3 + class-aware feasibility (T-26 + T-27 deployed via PR #5) | run completed `934b0ff..`; pre-registration at `docs/research/campaigns/02-paired-control-feasibility-2026-04-27.md` | 40 trials in 117 min, ~$15-25. **Q1 AFFIRMED at 1 cell** — rmsnorm/fp16/large NOV +14.7% (KL parity vs REF). Q2 NEGATIVE — class-collapse insufficient, opens T-26b. Q3 AFFIRMED — L1 reserve operator o0019 @ 864.9 tok/s = best L1 trial. Q4 parser brittle — 2/6 NOV halves broke silu_mul side, opens T-29. Disclosed launch-1 corner-cut (head-pipe SIGPIPE + un-pushed commits → ran old code; archived as `campaign02-redux-c01-baseline-2026-04-27`, relaunched proper from feature branch). |
| T-26b | FeasibilityModel per-knob feature weights | (this commit) | `_config_distance` accepts a `knob_weights` kwarg; weighted average instead of plain average per knob. `derive_knob_weights(catalog)` emits `{knob_name: 10.0}` for any knob appearing as `when_knob` or `requires_knob` in a compat rule (rest fall back to weight=1.0). Wired through `_build_surrogate(knob_weights=...)` for L1. Counterfactual test pinned: campaign-02 setup (2 fp8 fails far from query on other knobs + 2 auto KEPTs near query) gives unweighted P(success)=0.91 → ACCEPT (the bug), weighted P=0.24 → REJECT. 5 new tests in test_feasibility.py + 2 in test_l1_surface.py. 326 CPU tests pass. |
| T-29 | Paired-control prompt robustness — split into per-cell sequential calls | (PR #17) | `KernelProposer.propose_for_cells` now issues N separate per-cell LLM calls via `build_single_cell_kernel_prompt` instead of one batched 6-block paired prompt. C03-S validated: NOV-half failure rate dropped from C02's 2/6 (33%) to 1/9 (11%), at the pre-registered Outcome H threshold. |
| Campaign 03 | A100 narrow-replication paired-control (1× A100 spot, OpenRouter Sonnet 4, T-26b + T-29 + 1-GPU mode) | run completed `8c2ef41`; pre-registration + outcome at `docs/research/campaigns/03-h100-replication-2026-04-27.md` | 60 trials in 144 min, ~$15 for the final S run + ~$30-40 in earlier failed attempts. **Q1 NEGATIVE / Outcome C** — across 9 paired observations at 3 cells × N=3, 0/9 NOV wins, mean Δ%=−0.11%, range [−0.44, +0.15]; **the C02 +14.7% rmsnorm/fp16/large NOV win was lucky single-shot LLM emission noise, not robust kernel novelty**. Q2 partial (~20% L1 surrogate kept-rate vs C02's 8% baseline; below 30% target — opens T-26c). Q3 deferred (ran on A100, not H100; cross-hardware opens for a future campaign). Q4 AFFIRMED — T-29 dropped NOV-half failure rate to 1/9 (Outcome H). 7 PRs of latent bugs surfaced + fixed pre-completion (#14-#21); see Outcome section for full list. |
