# T-21 reconnaissance — attention-layer injector for L3

**Status:** in progress (live notes; reconciled into a design proposal at the end).

**Question to answer:** can autoinfer's L3 kernel-injection
mechanism (currently rmsnorm + silu_mul) extend to vLLM's attention
layer? If yes, how — and how big is the work?

This document is a recon log. Sections grow as I read the source.
End of doc has the **decision** (tractable in 2-4 hours / multi-day /
intractable) plus the design proposal for whichever applies.

---

## What I'm looking for

The simpler ops (RMSNorm, SiluAndMul) have these properties:

1. **Single class with a clear `forward_cuda` method** that vLLM's
   `CustomOp.dispatch_forward` rebinds at instance `__init__` time.
2. **Stable signature** — `forward_cuda(self, x, residual=None) ->
   Tensor` for RMSNorm, `forward_cuda(self, x) -> Tensor` for
   SiluAndMul. The LLM kernel proposer fills one signature.
3. **Stateless math** — no scheduler interaction, no KV cache, no
   block table.

Attention is none of these. So the question is what we can carve out
of vLLM's attention path that does have those properties.

---

## Recon log

### vLLM attention layout

Top-level: `vllm/model_executor/layers/attention/attention.py` —
`Attention(nn.Module, AttentionLayerBase)`. Constructor takes
`num_heads`, `head_size`, `scale`, `num_kv_heads`, `alibi_slopes`,
`logits_soft_cap`, `per_layer_sliding_window`, etc. — the per-model
attention shape.

`Attention.forward(query, key, value, output_shape=None)`:
- Reshapes inputs to `(num_tokens, num_heads, head_size)` etc.
- Allocates `output` tensor.
- Dispatches to `torch.ops.vllm.unified_attention_with_output(...)`
  (a registered custom op via `direct_register_custom_op`) — this is
  what shows up in the Inductor compilation graph.
- Returns `output.view(-1, hidden_size)`.

The actual compute is **NOT** in `Attention.forward`. It's in
`self.impl`, an `AttentionImpl` selected at construction by
`get_attn_backend(...)`.

### Backend impls (vllm/v1/attention/backends/)

```
flash_attn.py        FlashAttentionImpl       (FA2/FA3 wrapper)
flashinfer.py        FlashInferImpl           (FlashInfer kernels)
triton_attn.py       TritonAttentionImpl      (in-tree Triton — best for L3)
flex_attention.py    ...
xformers / cpu / mla / mamba / sparse / ...   (specialised paths)
```

For Qwen3-8B (causal LM, paged KV, no MLA), the practical backends
are FLASHINFER (default), FLASH_ATTN, and TRITON_ATTN. **TRITON_ATTN
is the right injection point for L3 because it's already pure
Python + Triton kernels** — no C++ binary swap needed.

### TritonAttentionImpl.forward — the candidate patch site

`vllm/v1/attention/backends/triton_attn.py:506`. Signature:

```python
def forward(self, layer, query, key, value, kv_cache,
            attn_metadata, output, output_scale=None,
            output_block_scale=None) -> Tensor:
```

The body is mostly metadata wrangling (cu_seqlens, descale tensors,
quant mode, scale caches, cascade flag, sliding window, sinks)
followed by a **single** call to:

```python
unified_attention(
    q=query[:num_actual_tokens],
    k=key_cache, v=value_cache, out=output[:num_actual_tokens],
    cu_seqlens_q=..., seqused_k=..., max_seqlen_q=..., max_seqlen_k=...,
    softmax_scale=self.scale, causal=True,
    alibi_slopes=..., window_size=..., block_table=..., softcap=...,
    q_descale=None, k_descale=..., v_descale=...,
    seq_threshold_3D=..., num_par_softmax_segments=...,
    softmax_segm_output=..., softmax_segm_max=..., softmax_segm_expsum=...,
    sinks=..., output_scale=..., mm_prefix_range=...,
    kv_quant_mode=..., k_scale_cache=..., v_scale_cache=...,
    chunk_lookback=...,
)
```

That's the actual kernel call — defined in
`vllm/v1/attention/ops/triton_unified_attention.py:505`.

### unified_attention (Python wrapper) and kernel_unified_attention (Triton kernel)

The Python wrapper at `triton_unified_attention.py:505` does:
1. Compute strides for q/k/v/output tensors.
2. Pick `BLOCK_M`, `TILE_SIZE`, `NUM_SEGMENTS_PER_SEQ` via `_get_tile_size`.
3. Allocate the 3D segment-sum buffers if 3D mode active.
4. Launch grid: `(num_seqs, num_query_heads, 1 or NUM_SEGMENTS)`.
5. Call `kernel_unified_attention[grid](...)` with **35+ parameters
   (pointers + scalars + ~25 constexprs)**.

`kernel_unified_attention` is the actual `@triton.jit` function at
`triton_unified_attention.py:57`. It implements:
- Paged KV cache lookup via `block_tables_ptr` indirection.
- Q@K^T matmul with optional ALiBi / sinks / sliding-window / softcap masks.
- Online softmax (Flash-style) with optional 3D segmentation
  (`num_par_softmax_segments` for parallel-tile softmax reduction).
- Optional FP8 KV cache (4 quant modes via `KV_QUANT_MODE` constexpr).
- Optional Gemma3-style chunked local attention (`CHUNK_LOOKBACK`).
- Per-(token, head) descale paths.

Constexprs alone include: `BLOCK_M`, `TILE_SIZE`, `BLOCK_SIZE` (KV
cache page size), `HEAD_SIZE`, `HEAD_SIZE_PADDED`, `BLOCK_Q`,
`NUM_SEGMENTS_PER_SEQ`, `IS_3D`, `KV_QUANT_MODE`, `USE_ALIBI_SLOPES`,
`USE_ALIBI_SQRT`, `USE_QQ_BIAS`, `USE_SOFTCAP`, `USE_SINKS`,
`SLIDING_WINDOW`, `USE_MM_PREFIX`, `MAX_MM_RANGES`, `USE_FP8`,
`CHUNK_LOOKBACK`, `FP8_MIN`, `FP8_MAX`. Each combination is a
recompile.

### Comparison: rmsnorm/silu_mul vs unified_attention as L3 targets

| Aspect | RMSNorm | SiluAndMul | unified_attention |
|---|---|---|---|
| Function signature | `(x, w, eps)` | `(x)` | 30+ params, 25+ constexprs |
| Body work | normalise last dim | `silu(x[:d]) * x[d:]` | Q@K^T + paged KV lookup + online softmax + masks |
| LLM prompt fits | yes (one screen) | yes | requires multi-shot examples + KV cache layout doc |
| Correctness gate | atol/rtol vs PyTorch ref on synthetic tensors | same | needs synthetic block_table, KV cache pages, cu_seqlens, RoPE'd Q/K — non-trivial setup |
| Likely LLM-novel speedup | small (op is tiny fraction of compute) | small | could be meaningful (~70% of Qwen3-8B compute) but hard to actually achieve vs the heavily-tuned reference |
| Code-paths to handle | residual=None vs residual=tensor | none | sliding window, sinks, alibi, softcap, FP8 KV (4 modes), chunked-local, 2D vs 3D softmax, cascade |
| Time to land injector | done | done | substantial (see below) |

### Estimated work for a meaningful T-21

Two viable scopes:

**Scope A — patch the Python wrapper `unified_attention(...)`.**
LLM proposer writes its own Triton kernel call inside a function with
the same outer signature. Pros: simpler signature for the LLM. Cons:
LLM has to invent a `@triton.jit` kernel from scratch that handles
all the metadata vLLM passes (or the L3 trial restricts to a config
subset).

**Scope B — patch the inner `kernel_unified_attention` Triton kernel
directly.** LLM proposer writes a `@triton.jit` function with the
exact signature vLLM expects. Pros: no signature mismatch risk.
Cons: 35+ params; LLM has to handle every constexpr branch
correctly or pin a config subset.

Either way, the time profile:

| Step | Effort |
|---|---|
| Extend `_TARGET_BINDINGS` in `injector.py` to support attention | ~30 min |
| Build a correctness gate with realistic q/k/v/kv_cache/block_table inputs | ~2-3 hours (the hard part — needs RoPE'd inputs, block-table indirection, cu_seqlens conventions matching vLLM's) |
| Update `KernelProposer` prompt with attention-specific reference + KV-cache-layout doc + few-shot examples | ~1-2 hours |
| Force vLLM to use TRITON_ATTN in L3 trials (config knob in `extra_vllm_args` or env var) | ~15 min |
| Sanity-check end-to-end with reference kernel re-injected (smoke run) | ~30 min wall + ~$5 |
| First real LLM-novel attention smoke campaign | ~60 min wall + ~$10 |
| Iterate on correctness gate based on first smoke failures | likely ~2-4 hours |

**Realistic total: 6-10 hours of focused work plus 2-3 smoke
iterations. Multi-day, not 2-4 hours.**

### Why the impact ceiling is also limited

`kernel_unified_attention` is already heavily-optimised: Triton-tuned
tile sizes via `_get_tile_size`, multi-segment softmax for long
sequences, cascade-attention support, FP8 fused descale paths. An
LLM-proposed rewrite from scratch is **extremely unlikely** to
produce a correct kernel that's also faster than this. More likely
outcomes:

1. LLM produces a kernel that fails correctness on at least one of
   {sliding window, sinks, alibi, softcap, FP8, chunked-local}.
2. LLM produces a kernel that mirrors the reference structurally —
   no measurable speedup.
3. LLM produces something that passes correctness on the L3 test
   matrix but fails on real serving inputs the gate doesn't cover.

A more tractable framing is **autotuning the existing kernel's tile
sizes** (BLOCK_M, TILE_SIZE, BLOCK_Q, etc.) per (head_size, seq_len,
batch_size) combination. That's actually a configuration-search
problem similar to L1, not a kernel-rewrite problem. vLLM's
`_get_tile_size` already encodes a heuristic; we could empirically
beat it via TPE search over the constexprs.

But that's a different research question ("can we beat vLLM's tile
heuristic") than the L3 thesis claim ("LLM-proposed kernel beats
reference end-to-end"). The two roles aren't interchangeable.

---

## Decision

**T-21 is multi-day, not 2-4 hours.**

The minimum viable LLM-proposed attention injector — even with the
simpler Scope A (Python-wrapper patch) and a tightly-pinned
config-subset (no FP8, no sliding window, no sinks) — needs:

- A non-trivial correctness gate (~2-3 h alone)
- A specialised kernel-proposer prompt (~1-2 h)
- 2-3 smoke iterations to debug
- An end-to-end campaign to validate

Realistic envelope: **6-10 hours of focused work + ~$15-25 of smoke
+ campaign budget**. Not appropriate as a "while the campaign runs"
side task. Not appropriate as a same-session add-on.

It also has a meaningful **probability of producing flat results**
even when implemented correctly — vLLM's reference is heavily
optimised; LLM kernels writing their own paged-KV attention from
scratch are unlikely to beat it.

### Recommended path forward

**Defer T-21 to its own dedicated session.** Before spending the
hours on the implementation, do the cheaper Campaign 02 first:

- **T-26** (FeasibilityModel feature engineering, ~1-2 h)
- **T-27** (same-config L3 control trials, ~1 h)
- **Campaign 02** with rmsnorm/silu_mul + the new feasibility
  classifier + paired-control L3 trials (~$15-20)

That campaign answers Q2 cleanly and produces a clean negative on Q1
at the rmsnorm/silu_mul surface (or a positive result if there's
residual signal we missed). Either outcome is publishable evidence;
either gives us a clean baseline to compare against when T-21 lands.

When T-21 then lands as its own session, we have:
- A working FeasibilityModel that's been tuned on real failure data
- Clean rmsnorm/silu_mul baseline numbers to compare attention-level
  results against
- A pre-registration document for the T-21 campaign that pins the
  config subset (no FP8, etc.) explicitly

### What NOT to do

- **Don't start T-21 implementation in this session.** Multi-day
  task with high failure-mode tail; doing it as a "let me try"
  while sitting at the keyboard would burn budget on a half-done
  injector that produces no useful campaign.
- **Don't conflate T-21 with kernel-tile-size autotuning.** They
  answer different research questions; doing the latter while
  marketing it as the former is the kind of corner-cut the user
  pushed back on earlier.

### TODO updates this recon implies

- T-21 stays open as P1 with an explicit "multi-day, see
  `docs/research/notes/t-21-attention-injector-recon.md` for design
  notes" pointer in the description.
- Open T-28: "Pre-registration of T-21 attention-injector campaign"
  — when the time comes to start T-21, the first commit is a
  campaign doc per `TEMPLATE.md` that pins the config subset.
- T-26 and T-27 stay P0 (already there) — they're the actually-do-
  next items.

### Time spent on this recon

~30 minutes. Within the 30-min reconnaissance budget. Decision is
"defer" rather than "implement", which respects the budget by not
turning the recon itself into a half-built attention injector.
