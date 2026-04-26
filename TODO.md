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
| T-01 | Verify KernelProposer fired with novel source in smoke 4 (vs falling back to reference seeds) | If the LLM never actually generated a kernel, the "kernel-into-vLLM works" result is a stub and the planned campaign tests nothing useful | smoke `8924421` artifacts in `docs/research/raw/smoke_l3vllm_validated-2026-04-26/` |
| T-02 | KernelProposer fallback event-emit + kernel-source-hash on every L3 trial | Post-run analysis can't currently distinguish "LLM generated" from "fallback to reference" trials. Without this, novel-vs-reference comparisons are guesswork | `proposer.py:fallback_when_empty` |

### P1 — corner-cuts; must fix before article

| ID | Item | Why it matters | Reference |
|---|---|---|---|
| T-10 | Top-K Jensen-Shannon (`topk_js_divergence`) wired into `run_gate` in place of top-K KL | Three calibration band-aids ride on the broken-metric stack (median-cap → noise-floor → one-way valve). JS is bounded, no floor amplification, would collapse all three band-aids into one principled metric | `harness/gate.py:topk_kl_divergence` shipped, JS shipped at [`803c1f9`] but never wired |
| T-11 | L3 standalone adapter perf timing uses `time.perf_counter` without `torch.cuda.synchronize()` | Asynchronous CUDA can end the timer before the kernel completes. Not in the production L3-vLLM path (vllm bench has correct timing), but the standalone CPU adapter is still wrong | `layers/l3_kernel/adapter.py:_measure_perf` |
| T-12 | Catalog constraint hardcoded to Qwen3-8B `max_model_len=32768` | Switching models silently breaks the `chunked_prefill_batched_tokens_bound` rule. Should be parameterised over the configured model's actual max length | `layers/l1_engine/knobs.yaml:chunked_prefill_batched_tokens_bound` |
| T-13 | `compile_candidate` writes temp `.py` files with no cleanup (standalone path) | Production L3-vLLM adapter cleans up via `_stop_candidate`. Standalone L3KernelAdapter doesn't, and after many trials `/tmp` accumulates `_l3_*.py` files | `layers/l3_kernel/surface.py:compile_candidate` |
| T-14 | `reserve_cap=4` picked arbitrarily, not informed by data | No principled argument for 4 vs 2 vs 8. Should be informed by observed surrogate-improvement rate per re-explored trial after a few campaigns generate data | `controller/stale.py:LayerSpec.reserve_cap` |
| T-15 | Hardware notes hand-authored as prose (FP8-on-A100, chunked_prefill rules) | Exactly the manual-rule-authoring anti-pattern that the FeasibilityModel was built to retire. Once the classifier accumulates ~50 trials of failure data, the prose notes should be progressively trimmed and the data-driven path validated end-to-end | `examples/qwen3-8b-l1-l2-l3-joint/config.yaml` `hardware_notes:` |
| T-16 | `OptunaSurrogate._penalty_for_failure` ignores FailureKind | All typed failures earn the same scalar penalty. The `ConstrainedOptunaSurrogate` uses FailureKind via the FeasibilityModel, but the perf model still treats OOM and QUALITY_KL as identical signals to TPE's KDE | `policy/surrogate.py:_penalty_for_failure` |
| T-17 | `KernelProposer.fallback_when_empty=True` silently uses reference seeds when LLM output is unparseable | Hides LLM-side failures from telemetry. Should event-emit a `kernel_proposer_fallback` event on every fallback so artifacts make this visible | `layers/l3_kernel/proposer.py` |

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

| ID | Item | Closed by |
|---|---|---|
| (closed items added per commit; first iteration of this file ships with no closed items so the open list is the canonical view) | | |
