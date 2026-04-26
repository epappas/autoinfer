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

(none currently open — T-01, T-02, T-17 closed in pre-flight for
campaign 01)

### P1 — corner-cuts; must fix before article

| ID | Item | Why it matters | Reference |
|---|---|---|---|
| T-10 | Top-K Jensen-Shannon (`topk_js_divergence`) wired into `run_gate` in place of top-K KL | Three calibration band-aids ride on the broken-metric stack (median-cap → noise-floor → one-way valve). JS is bounded, no floor amplification, would collapse all three band-aids into one principled metric | `harness/gate.py:topk_kl_divergence` shipped, JS shipped at [`803c1f9`] but never wired |
| T-12 | Catalog constraint hardcoded to Qwen3-8B `max_model_len=32768` | Switching models silently breaks the `chunked_prefill_batched_tokens_bound` rule. Should be parameterised over the configured model's actual max length | `layers/l1_engine/knobs.yaml:chunked_prefill_batched_tokens_bound` |
| T-14 | `reserve_cap=4` picked arbitrarily, not informed by data | No principled argument for 4 vs 2 vs 8. Should be informed by observed surrogate-improvement rate per re-explored trial after a few campaigns generate data | `controller/stale.py:LayerSpec.reserve_cap` |
| T-15 | Hardware notes hand-authored as prose (FP8-on-A100, chunked_prefill rules) | Exactly the manual-rule-authoring anti-pattern that the FeasibilityModel was built to retire. Once the classifier accumulates ~50 trials of failure data, the prose notes should be progressively trimmed and the data-driven path validated end-to-end | `examples/qwen3-8b-l1-l2-l3-joint/config.yaml` `hardware_notes:` |
| T-16 | `OptunaSurrogate._penalty_for_failure` ignores FailureKind | All typed failures earn the same scalar penalty. The `ConstrainedOptunaSurrogate` uses FailureKind via the FeasibilityModel, but the perf model still treats OOM and QUALITY_KL as identical signals to TPE's KDE | `policy/surrogate.py:_penalty_for_failure` |

### P2 — research extensions

| ID | Item | Description | Reference |
|---|---|---|---|
| T-20 | RoPE support in injector | vLLM's `RotaryEmbedding` API has many shape-varying paths (positions, query/key fused, kv-cache aware). v1 injector explicitly skips it; v2 needs to either fan out per shape variant or accept a kernel signature richer than the rmsnorm/silu_mul cases | `layers/l3_kernel/injector.py:_TARGET_BINDINGS` |
| T-21 | Attention-layer injector | RMSNorm and SiluAndMul are tiny fractions of total compute. Attention is the real bottleneck. Injecting LLM-proposed attention kernels (FLASHINFER/FLASH_ATTN replacements) is where end-to-end wins live | not started |
| T-22 | `pareto_front_by_layer` not surfaced in `run_summary.json` | Analyzer prints it; summary persists only the joint frontier. Per-layer best in the summary would let downstream tools (plots, articles) skip re-running the analyzer | `harness/ledger.py:pareto_front_by_layer`, `telemetry/summary.py` |
| T-23 | L2 `peak_hbm_gb` reads campaign-container nvidia-smi for remote candidates | The L2 adapter reports local GPU memory even when the candidate is on a remote H100. Real per-trial HBM needs to be queried inside the remote deployment | `layers/l2_topology/adapter.py` |
| T-24 | Stale Basilica deployments from prior sessions | 11 deployments going back to 2026-03-16 listed during the first joint campaign. May be the user's other projects; not deleted unilaterally. Needs user triage | flagged in joint-run analyses |
| T-25 | Per-trial gate KL distribution captured to artifact | Currently only mean / max land in the trial JSON's `extra`. Full per-prompt KL histogram would let post-hoc analyses identify quality drift patterns the gate scalar masks | `harness/gate.py:GateResult` |

---

## Closed

| ID | Item | Closed by | Notes |
|---|---|---|---|
| T-01 | Verify smoke 4 KernelProposer used novel source | (this commit) | Audit confirmed: smoke 4 sources sha `034fc4b4...` (rmsnorm) and `d1407aa1...` (silu_mul) differ from reference shas `3e8a82bf...` and `69cf8186...`. Both sources contain `@triton.jit` decorators with names `*_triton_kernel`. The LLM proposer IS generating Triton source. |
| T-02 | Kernel source hash on every L3 trial Measurement | (this commit) | `_kernel_source_metadata()` in `layers/l3_kernel/vllm_adapter.py` + same in standalone `adapter.py`. Records `kernel_source_sha_int` (int of first 12 hex chars) and `kernel_is_reference` (1.0/0.0) in `Measurement.extra` so post-run analysis can group novel vs fallback. |
| T-17 | KernelProposer fallback transparency | (this commit) | `[autoinfer.l3.proposer] fallback to reference seeds` marker prints to stdout when LLM returns unparseable blocks; captured in basilica logs + per-trial `_vllm.out` files alongside the injector's `[autoinfer.l3.injector]` marker. |
| T-11 | L3 standalone perf timing uses `time.perf_counter` without `torch.cuda.synchronize()` | (this commit) | `_best_elapsed_cuda` uses `torch.cuda.Event` + `synchronize()` when inputs are CUDA tensors; `_best_elapsed_wall` keeps the wall-clock path for CPU. The vLLM-mode L3 adapter uses `vllm bench serve` which has its own correct timing — this fix matters for the standalone L3 path used in dev workflows and on a real GPU. |
| T-13 | `compile_candidate` temp-file leak | (this commit) | `_l3_temp_root()` returns a process-shared temp dir created on first use; `atexit.register` deletes it on process exit. All compiled candidates write into this shared dir, so `/tmp` no longer accumulates stale `_l3_*.py` files across trials. Test added in `test_l3_surface.py` to pin the shared-dir behavior. |
