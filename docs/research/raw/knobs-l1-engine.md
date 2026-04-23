# L1 Engine Knob Catalog (vLLM)

Scope: vLLM engine-level (L1) search surface for autoinfer, organized by the eight clusters from thesis §4.1. Each knob lists CLI flag, Python field, type, default, the practical search range autoinfer should explore (narrower than the CLI-legal range), coupled knobs, hard compatibility constraints, the primary performance axis affected, and a one-line description.

Sources (fetched 2026-04-22):
- `vllm/engine/arg_utils.py` (source of truth for CLI parser wiring).
- `vllm/config/cache.py`, `vllm/config/scheduler.py`, `vllm/config/parallel.py`, `vllm/config/compilation.py`, `vllm/config/speculative.py`, `vllm/config/model.py` (dataclass defaults).
- `vllm/v1/attention/backends/registry.py` (attention backend enum).
- `vllm/model_executor/layers/quantization/__init__.py` (quantization methods).
- `docs/configuration/optimization.md` (typical good values, tradeoffs).
- `docs/configuration/engine_args.html` returned HTTP 403; see "Under-documented areas".

All defaults below are quoted from the dataclass definitions in vLLM main as of fetch time. vLLM V1 is assumed (chunked prefill on by default, RECOMPUTE preemption default, torch.compile piecewise default).

---

## 1. Scheduling

### `--max-num-seqs`
- **Python field**: `max_num_seqs`
- **Type**: `int`
- **Default**: `128` (`SchedulerConfig.max_num_seqs`; `EngineArgs` passes `None` and resolves via `_set_default_max_num_seqs_and_batched_tokens_args()` which is hardware/usage aware).
- **Autoinfer search range**: `{32, 64, 128, 256, 512, 1024}`. Bounded by KV-cache capacity; >1024 rarely helps on a single GPU once KV-cache saturates.
- **Coupled_with**: `max_num_batched_tokens`, `gpu_memory_utilization`, `kv_cache_dtype`, `block_size`, `tensor_parallel_size` (each raises KV headroom).
- **Compat_constraints**: No hard rule, but must fit in `num_gpu_blocks` after profiling; too large triggers preemption.
- **Axis**: throughput (primary), latency (secondary; more concurrency raises ITL).
- **Description**: Cap on concurrent sequences processed in one scheduler iteration.

### `--max-num-batched-tokens`
- **Python field**: `max_num_batched_tokens`
- **Type**: `int`
- **Default**: `2048` (`SchedulerConfig`; `EngineArgs` resolves per context).
- **Autoinfer search range**: `{1024, 2048, 4096, 8192, 16384}`. The optimization guide states small values (2048) favor ITL while large values (>8192) favor TTFT.
- **Coupled_with**: `max_num_seqs`, `max_model_len`, `enable_chunked_prefill`.
- **Compat_constraints**: Must be `>= max_model_len` when `enable_chunked_prefill=False`.
- **Axis**: latency (TTFT vs ITL trade) and throughput.
- **Description**: Per-iteration token budget across prefill + decode.

### `--scheduling-policy`
- **Python field**: `scheduling_policy` (→ `SchedulerConfig.policy`)
- **Type**: categorical
- **Default**: `"fcfs"`
- **Autoinfer search range**: `{"fcfs", "priority"}`.
- **Coupled_with**: client-side `priority` on requests; `max_num_seqs` (higher concurrency makes ordering more visible).
- **Compat_constraints**: `"priority"` only meaningful when requests carry a priority value.
- **Axis**: latency (tail/p99), quality of service.
- **Description**: FCFS order vs request-priority ordering for admission and preemption.

### `--enable-chunked-prefill` / `--no-enable-chunked-prefill`
- **Python field**: `enable_chunked_prefill`
- **Type**: `bool`
- **Default**: `True` in V1 (`SchedulerConfig.enable_chunked_prefill=True`); `EngineArgs` default `None` resolves per model.
- **Autoinfer search range**: `{True, False}`. Prefer `True` for mixed workloads.
- **Coupled_with**: `max_num_batched_tokens`, `max_num_seqs`, `max_model_len`.
- **Compat_constraints**: Disabling chunked prefill forces `max_num_batched_tokens >= max_model_len`. Some attention backends do not support chunked prefill (TREE_ATTN, some MLA variants).
- **Axis**: ITL / throughput.
- **Description**: Split long prefills into chunks and co-batch with decode.

### `--preemption-mode`
- **Python field**: `preemption_mode` (part of scheduler; not exposed as CLI in all builds, V1 default is RECOMPUTE)
- **Type**: categorical
- **Default**: `"recompute"` (V1 default per optimization guide).
- **Autoinfer search range**: `{"recompute", "swap"}`.
- **Coupled_with**: `swap_space`, `gpu_memory_utilization`, `max_num_seqs`.
- **Compat_constraints**: `"swap"` requires `swap_space > 0`. V1 architecture is tuned for RECOMPUTE; SWAP may regress.
- **Axis**: latency tail under contention, memory.
- **Description**: Strategy when KV cache is full — recompute prompt or swap KV blocks to CPU.

---

## 2. KV Cache

### `--block-size`
- **Python field**: `block_size` (→ `CacheConfig.block_size`)
- **Type**: `int` (categorical set)
- **Default**: `None` → resolves to `DEFAULT_BLOCK_SIZE = 16`.
- **Autoinfer search range**: `{8, 16, 32, 64, 128}`. Larger blocks reduce metadata overhead; smaller improve fragmentation on short sequences.
- **Coupled_with**: `attention_backend`, `kv_cache_dtype`, `enable_prefix_caching`.
- **Compat_constraints**: Some backends restrict block size — FLASHMLA and MLA variants typically require specific block sizes (commonly 64); FLASHINFER has its own constraints. Autoinfer should restrict per backend.
- **Axis**: memory fragmentation, throughput.
- **Description**: KV cache block granularity in tokens.

### `--kv-cache-dtype`
- **Python field**: `kv_cache_dtype` (→ `CacheConfig.cache_dtype`)
- **Type**: categorical
- **Default**: `"auto"` (use model dtype).
- **Autoinfer search range**: `{"auto", "fp8", "fp8_e4m3", "fp8_e5m2"}`. Full enum also includes `float16`, `bfloat16`, `fp8_inc`, `fp8_ds_mla`, `turboquant_k8v4`, `turboquant_4bit_nc`, `turboquant_k3v4_nc`, `turboquant_3bit_nc`, `int8_per_token_head`, `fp8_per_token_head`, `nvfp4` — treat non-fp8 quant options as opt-in since they trade quality for memory.
- **Coupled_with**: `attention_backend`, `quantization`, `block_size`.
- **Compat_constraints**: `fp8_e4m3` / `fp8_e5m2` require FP8-capable hardware (Hopper+, or ROCm MI300). Many older backends (TORCH_SDPA) do not accept fp8 KV. `fp8_ds_mla` and `nvfp4` are MLA/Blackwell specific.
- **Axis**: memory, quality (small regression).
- **Description**: Storage dtype for KV cache.

### `--gpu-memory-utilization`
- **Python field**: `gpu_memory_utilization` (→ `CacheConfig.gpu_memory_utilization`)
- **Type**: `float`
- **Default**: `0.92` (constraint `gt=0, le=1`).
- **Autoinfer search range**: `{0.85, 0.90, 0.92, 0.95, 0.97}`. Do not go above 0.97 in production to keep OOM headroom.
- **Coupled_with**: `max_num_seqs`, `max_num_batched_tokens`, `cpu_offload_gb`, `swap_space`.
- **Compat_constraints**: Must leave enough VRAM for activations; interacts with `cudagraph_capture_sizes` memory.
- **Axis**: memory, throughput.
- **Description**: Fraction of GPU memory reservable for model + KV cache.

### `--swap-space`
- **Python field**: `swap_space`
- **Type**: `float` (GiB per worker)
- **Default**: `4` (standard vLLM default; not directly in `cache.py` excerpt — see Under-documented areas).
- **Autoinfer search range**: `{0, 4, 8, 16}`. Only relevant when `preemption_mode="swap"`.
- **Coupled_with**: `preemption_mode`, system RAM.
- **Compat_constraints**: Values >0 require pinned CPU memory; V1 favors recompute, so swap is often inert.
- **Axis**: memory, latency tail.
- **Description**: CPU swap budget for preempted KV blocks per worker.

### `--num-gpu-blocks-override`
- **Python field**: `num_gpu_blocks_override`
- **Type**: `int | None`
- **Default**: `None` (profiler-chosen).
- **Autoinfer search range**: leave as `None` unless pinning for reproducibility; if pinned, vary in ±10% band around the profiler output.
- **Coupled_with**: `gpu_memory_utilization`, `block_size`.
- **Compat_constraints**: Must not exceed profiled memory; otherwise OOM.
- **Axis**: memory (deterministic cap).
- **Description**: Override the profiler's computed KV-block count.

### `--enable-prefix-caching` / `--no-enable-prefix-caching`
- **Python field**: `enable_prefix_caching`
- **Type**: `bool`
- **Default**: `True` in `CacheConfig`; `EngineArgs` default `None` resolves per model.
- **Autoinfer search range**: `{True, False}`. Turn off only if workload has essentially zero prefix reuse and caching overhead hurts throughput.
- **Coupled_with**: `block_size`, `kv_cache_dtype`, `attention_backend`.
- **Compat_constraints**: Some attention backends and some encoder-decoder models do not support it; autoinfer should probe backend compatibility before enabling.
- **Axis**: latency (TTFT on repeat prefixes), throughput on prefix-heavy workloads.
- **Description**: Automatic reuse of KV cache for shared prompt prefixes.

---

## 3. Attention Backend

### `--attention-backend` (env `VLLM_ATTENTION_BACKEND`)
- **Python field**: `attention_backend` (→ `AttentionConfig.backend`)
- **Type**: categorical (`AttentionBackendEnum`)
- **Default**: `None` (platform auto-selects — typically FLASH_ATTN on Hopper/Ampere with non-MLA models, FLASHMLA/FLASH_ATTN_MLA for MLA models on Hopper).
- **Autoinfer search range**: primary dense-model set `{None (auto), FLASH_ATTN, FLASHINFER, TRITON_ATTN, FLEX_ATTENTION}`; MLA-model set `{FLASHMLA, FLASH_ATTN_MLA, TRITON_MLA, CUTLASS_MLA, FLASHINFER_MLA}`; ROCm set `{ROCM_ATTN, ROCM_AITER_FA, ROCM_AITER_UNIFIED_ATTN, ROCM_AITER_MLA, ROCM_AITER_TRITON_MLA}`; CPU `{CPU_ATTN, TORCH_SDPA}`.
- **Full legal enum**: `FLASH_ATTN`, `FLASH_ATTN_DIFFKV`, `TRITON_ATTN`, `ROCM_ATTN`, `ROCM_AITER_MLA`, `ROCM_AITER_TRITON_MLA`, `ROCM_AITER_FA`, `ROCM_AITER_MLA_SPARSE`, `XPU_MLA_SPARSE`, `TORCH_SDPA`, `FLASHINFER`, `FLASHINFER_MLA`, `FLASHINFER_MLA_SPARSE`, `TRITON_MLA`, `CUTLASS_MLA`, `FLASHMLA`, `FLASHMLA_SPARSE`, `FLASH_ATTN_MLA`, `NO_ATTENTION`, `FLEX_ATTENTION`, `TREE_ATTN`, `ROCM_AITER_UNIFIED_ATTN`, `CPU_ATTN`, `TURBOQUANT`, `CUSTOM`.
- **Coupled_with**: `block_size`, `kv_cache_dtype`, `quantization`, `enforce_eager`, `speculative_config` (`TREE_ATTN` is for eagle-style tree decoding).
- **Compat_constraints**: MLA backends only for MLA architectures (DeepSeek, etc.). `*_SPARSE` variants require sparse models. `TREE_ATTN` intended for speculative tree decoding. `CPU_ATTN`/`TORCH_SDPA` needed on CPU targets. `FLASHINFER` requires the flashinfer package; `FLASHMLA` / `CUTLASS_MLA` require Hopper/Blackwell.
- **Axis**: latency and throughput (primary), sometimes quality on sparse variants.
- **Description**: Attention kernel implementation used by all layers.

---

## 4. Parallelism

### `--tensor-parallel-size` / `-tp`
- **Python field**: `tensor_parallel_size`
- **Type**: `int`
- **Default**: `1`.
- **Autoinfer search range**: divisors of `num_attention_heads` within `{1, 2, 4, 8}`. Limit to physical GPU count.
- **Coupled_with**: `pipeline_parallel_size`, `data_parallel_size`, `enable_expert_parallel`, `disable_custom_all_reduce`, `distributed_executor_backend`.
- **Compat_constraints**: Must divide the model's head count (and KV-head count for GQA). Higher TP needs fast intra-node interconnect (NVLink); TP across PCIe/Ethernet is usually a net loss.
- **Axis**: memory (enables larger models), latency on large models, throughput (moderate loss from all-reduce).
- **Description**: Shard each layer across N GPUs via tensor parallelism.

### `--pipeline-parallel-size` / `-pp`
- **Python field**: `pipeline_parallel_size`
- **Type**: `int`
- **Default**: `1`.
- **Autoinfer search range**: `{1, 2, 4}`; only above 1 when TP has been maxed on intra-node and model still does not fit, or cross-node.
- **Coupled_with**: `tensor_parallel_size`, `max_num_seqs` (needs enough in-flight micro-batches to hide bubbles).
- **Compat_constraints**: Best on deep/narrow models and multi-node. Interacts poorly with low request concurrency (pipeline bubbles dominate).
- **Axis**: memory (enables very large models), latency penalty.
- **Description**: Split model across layer stages.

### `--data-parallel-size` / `-dp`
- **Python field**: `data_parallel_size`
- **Type**: `int`
- **Default**: `1`.
- **Autoinfer search range**: `{1, 2, 4, 8}` bounded by available replicas (GPUs / (TP*PP)).
- **Coupled_with**: `tensor_parallel_size`, `pipeline_parallel_size`, `enable_expert_parallel` (DP degree also shards MoE).
- **Compat_constraints**: `DP * TP * PP <= total GPUs`. DP raises throughput but duplicates model weights.
- **Axis**: throughput (primary).
- **Description**: Replicate the engine across independent request streams.

### `--enable-expert-parallel` / `-ep`
- **Python field**: `enable_expert_parallel`
- **Type**: `bool`
- **Default**: `False`.
- **Autoinfer search range**: `{True, False}` only for MoE models.
- **Coupled_with**: `tensor_parallel_size`, `data_parallel_size`.
- **Compat_constraints**: Only valid for MoE architectures (DeepSeek-MoE, Mixtral, Qwen-MoE, etc.). EP degree is set by `TP * DP`.
- **Axis**: throughput / memory on MoE.
- **Description**: Route experts across ranks instead of sharding each expert via TP.

### `--disable-custom-all-reduce` / `--no-...`
- **Python field**: `disable_custom_all_reduce`
- **Type**: `bool`
- **Default**: `False`.
- **Autoinfer search range**: `{False, True}`. Default is fine; toggle only on suspected correctness/perf regressions.
- **Coupled_with**: `tensor_parallel_size` (only relevant when TP>1).
- **Compat_constraints**: Custom all-reduce requires NVLink peer access; on PCIe-only systems it may be auto-disabled.
- **Axis**: latency.
- **Description**: Force NCCL instead of vLLM's custom all-reduce kernel.

---

## 5. Quantization

### `--quantization` / `-q`
- **Python field**: `quantization`
- **Type**: categorical
- **Default**: `None` (or the model's own `quantization_config`).
- **Autoinfer search range (weight quant only)**: `{None, "fp8", "awq_marlin", "gptq_marlin", "compressed-tensors", "bitsandbytes"}`. For newer Blackwell hardware add `{"modelopt_fp4", "mxfp4"}`.
- **Full legal set** (from `quantization/__init__.py`): `awq`, `gptq`, `gptq_marlin`, `awq_marlin`, `fp8`, `fbgemm_fp8`, `fp_quant`, `modelopt`, `modelopt_fp4`, `modelopt_mxfp8`, `modelopt_mixed`, `gguf`, `compressed-tensors`, `bitsandbytes`, `experts_int8`, `quark`, `moe_wna16`, `torchao`, `inc`, `mxfp4`, `gpt_oss_mxfp4`, `cpu_awq`, `online`, plus online shortcuts `fp8_per_tensor`, `fp8_per_block`, `int8_per_channel_weight_only`, `mxfp8`. Deprecated: `tpu_int8`, `fbgemm_fp8`, `fp_quant`.
- **Coupled_with**: `kv_cache_dtype`, `attention_backend`, `dtype`, weights on disk (quant must match checkpoint).
- **Compat_constraints**:
  - `awq_marlin`, `gptq_marlin`, `marlin` — Ampere+ (sm80+), require Marlin-compatible weights. Marlin kernels commonly require `kv_cache_dtype ∈ {"auto", "fp8_e4m3"}`.
  - `fp8`, `fp8_per_tensor`, `fp8_per_block` — Hopper+ (sm90+); some paths work on Ada.
  - `modelopt_fp4`, `mxfp4`, `gpt_oss_mxfp4` — Blackwell (sm100+).
  - `bitsandbytes` — works broadly but slower; incompatible with some MoE paths.
  - `gguf` — checkpoint must be GGUF; single-GPU mostly.
  - `compressed-tensors` — schema must match the checkpoint.
- **Axis**: memory (primary), throughput (Marlin/FP8 often faster), quality (small regression).
- **Description**: Weight quantization format applied at load time.

### `--dtype`
- **Python field**: `dtype` (→ `ModelConfig.dtype`)
- **Type**: categorical
- **Default**: `"auto"` (FP16 for FP32/FP16 models, BF16 for BF16).
- **Autoinfer search range**: `{"auto", "bfloat16", "float16"}`.
- **Coupled_with**: `quantization`, `kv_cache_dtype`, `attention_backend`.
- **Compat_constraints**: `float16` not advised on Hopper BF16-native models; `float32` only for debugging.
- **Axis**: memory, quality.
- **Description**: Activation/weight compute dtype when not quantized.

### `--load-format`
- **Python field**: `load_format`
- **Type**: categorical
- **Default**: `"auto"`.
- **Autoinfer search range**: leave at `"auto"`; toggle only for `"bitsandbytes"`, `"gguf"`, `"runai_streamer"`, `"sharded_state"`, `"tensorizer"` when operationally needed.
- **Coupled_with**: `quantization`.
- **Compat_constraints**: Must match the on-disk checkpoint encoding.
- **Axis**: startup time; no runtime perf change.
- **Description**: Weight loader backend.

---

## 6. Speculative Decoding

### `--speculative-config` / `-sc`
- **Python field**: `speculative_config` (JSON → `SpeculativeConfig`)
- **Type**: JSON object (`dict[str, Any] | None`)
- **Default**: `None` (off).
- **Autoinfer search range**: four candidate configs:
  1. off (`None`)
  2. `{"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 5, "prompt_lookup_min": 2}` — no draft model needed.
  3. `{"method": "eagle", "model": <eagle head>, "num_speculative_tokens": 3}` — requires trained eagle head.
  4. `{"method": "draft_model", "model": <small model>, "num_speculative_tokens": 5, "draft_tensor_parallel_size": 1}`.
- **Coupled_with**: `attention_backend` (TREE_ATTN pairs with eagle-style), `max_num_seqs` (spec decoding typically helps more at low batch), `tensor_parallel_size` (`draft_tensor_parallel_size ∈ {1, target_tp}`).
- **Compat_constraints**: Verified methods: `ngram`, `medusa`, `mlp_speculator`, `draft_model`, `suffix`, `eagle`, `eagle3`, `extract_hidden_states`, `ngram_gpu`, `dflash`, MTP variants. `num_speculative_tokens > 0` required (unless draft config provides `n_predict`). `draft_tensor_parallel_size ∈ {1, target_tp}`. Eagle requires an eagle head checkpoint matching the target model. MoE + speculative decoding has known corner-case issues in some versions.
- **Axis**: latency (primary, especially ITL at low concurrency), throughput (can hurt at high batch).
- **Description**: Generate and verify K draft tokens per step.

### `num_speculative_tokens` (inside `--speculative-config`)
- **Type**: `int`
- **Default**: `None` (required unless the draft model's config exposes `n_predict`).
- **Autoinfer search range**: `{2, 3, 4, 5, 7}`. Larger values help only when acceptance rate is high.
- **Coupled_with**: `method`, `model`, batch size.
- **Compat_constraints**: `gt=0`.
- **Axis**: latency vs throughput trade.
- **Description**: How many draft tokens to propose per verification step.

### `draft_tensor_parallel_size`
- **Type**: `int`
- **Default**: `None` (auto).
- **Autoinfer search range**: `{1, target_tp}`.
- **Coupled_with**: `tensor_parallel_size`.
- **Compat_constraints**: Must be `1` or equal to target TP.
- **Axis**: latency (draft overhead), memory (draft weights).
- **Description**: TP degree for the draft model.

---

## 7. Compile Graphs (torch.compile / CUDA graphs)

### `--enforce-eager` / `--no-enforce-eager`
- **Python field**: `enforce_eager` (→ `ModelConfig.enforce_eager`)
- **Type**: `bool`
- **Default**: `False`.
- **Autoinfer search range**: `{False, True}`. `True` only as a fallback (slower; useful for warm-up or debugging).
- **Coupled_with**: `compilation_config`, `cuda_graph_sizes`, `max_seq_len_to_capture`.
- **Compat_constraints**: When `True`, CUDA graphs and `torch.compile` are both disabled regardless of `compilation_config`.
- **Axis**: latency (slower when True), startup (faster when True).
- **Description**: Force eager PyTorch execution — bypass CUDA graphs and compile.

### `--compilation-config` / `-cc` / `-O<level>`
- **Python field**: `compilation_config` (→ `CompilationConfig`)
- **Type**: JSON object or integer mode
- **Default**: V1 auto-selects `mode=3` (`VLLM_COMPILE`). Mode values:
  - `0` NONE — pure eager.
  - `1` STOCK_TORCH_COMPILE — standard `torch.compile`.
  - `2` DYNAMO_TRACE_ONCE — single Dynamo trace, no recompile.
  - `3` VLLM_COMPILE — vLLM's Inductor backend with piecewise compile, caching, fusions, shape specialization.
- **Autoinfer search range**: primary `{mode: 3}` (default); fallbacks `{mode: 0, mode: 1}` for debugging or compile-failure cases. `{mode: 2}` for faster startup when deploying short-lived jobs.
- **Coupled_with**: `enforce_eager`, `cudagraph_mode`, `cuda_graph_sizes`, `max_seq_len_to_capture`, `splitting_ops`, `custom_ops`.
- **Compat_constraints**: Mode 3 only on V1. Mode 0 effectively equals `enforce_eager=True`. Some custom ops and some quantization kernels disable certain fusions.
- **Axis**: startup time vs runtime performance.
- **Description**: Torch.compile pipeline and optimization level.

### `--cuda-graph-sizes` (field alias `cudagraph_capture_sizes`)
- **Python field**: `cuda_graph_sizes` → `CompilationConfig.cudagraph_capture_sizes`
- **Type**: `list[int] | None`
- **Default**: `None` → auto-pattern `[1, 2, 4] + range(8, 256, 8) + range(256, max_size+1, 16)`, capped at `max_cudagraph_capture_size`.
- **Autoinfer search range**: leave auto; if tuning, candidate sets `{auto, [1,2,4,8,16,32,64,128,256], [1,8,32,128]}` trading coverage vs startup.
- **Coupled_with**: `max_cudagraph_capture_size`, `max_num_seqs`, `cudagraph_mode`, `gpu_memory_utilization` (graphs cost VRAM).
- **Compat_constraints**: Empty list disables graphs. Graphs require same tensor shapes across runs; conflicts with highly variable batch sizes.
- **Axis**: latency (primary — graph hits reduce dispatch overhead), memory, startup.
- **Description**: Batch sizes to capture as CUDA graphs.

### `--max-seq-len-to-capture` (→ `max_cudagraph_capture_size`)
- **Python field**: `max_seq_len_to_capture` → `CompilationConfig.max_cudagraph_capture_size`
- **Type**: `int`
- **Default**: `min(max_num_seqs * 2, 512)` (or `max(cudagraph_capture_sizes)` if explicitly set).
- **Autoinfer search range**: `{256, 512, 1024, 2048, 8192}`. Higher values cover more graph hits but cost memory and startup.
- **Coupled_with**: `max_num_seqs`, `cuda_graph_sizes`, `gpu_memory_utilization`.
- **Compat_constraints**: Sequences longer than this fall through to eager path.
- **Axis**: latency (wider graph coverage) vs memory / startup.
- **Description**: Upper bound on captured CUDA-graph size.

### `cudagraph_mode` (inside `--compilation-config`)
- **Type**: categorical
- **Default**: `None` (auto-selected for V1).
- **Autoinfer search range**: `{"PIECEWISE", "FULL_AND_PIECEWISE", "FULL_DECODE_ONLY", "FULL", "NONE"}`. Prefer `PIECEWISE` (default). `FULL_DECODE_ONLY` helps pure decode workloads.
- **Coupled_with**: `splitting_ops`, `attention_backend`, `enable_chunked_prefill`.
- **Compat_constraints**: `FULL` incompatible with many attention backends that require host-side dynamism; `FULL_DECODE_ONLY` requires chunked prefill so prefill runs outside graph.
- **Axis**: latency.
- **Description**: Whether CUDA graph captures whole model, decode-only, or piecewise segments.

---

## 8. Offload

### `--cpu-offload-gb`
- **Python field**: `cpu_offload_gb` (→ `UVAOffloadConfig.cpu_offload_gb`)
- **Type**: `float` (GiB)
- **Default**: `0.0`.
- **Autoinfer search range**: `{0, 4, 8, 16, 32}`. Only when weights do not fit in VRAM and TP/PP are not options.
- **Coupled_with**: `gpu_memory_utilization`, `tensor_parallel_size`, pinned host memory.
- **Compat_constraints**: Needs page-locked host memory and a CUDA UVA-capable setup. Not all models support offload of all layers. Large values incur heavy PCIe traffic — expect large throughput loss.
- **Axis**: memory (enables larger models at severe latency cost).
- **Description**: GiB of weights to offload to CPU pinned memory per worker.

### `--swap-space` (also relevant here — covered in §2).

---

## Compatibility Matrix (hard rules)

| Rule | Source / reason |
|---|---|
| `enable_chunked_prefill=False` ⇒ `max_num_batched_tokens >= max_model_len` | scheduler invariant |
| `preemption_mode="swap"` ⇒ `swap_space > 0` | needs CPU swap buffer |
| FP8 KV cache (`fp8`, `fp8_e4m3`, `fp8_e5m2`) ⇒ Hopper+ (sm90+) or ROCm MI300 | kernel support |
| `modelopt_fp4`, `mxfp4`, `gpt_oss_mxfp4` ⇒ Blackwell (sm100+) | kernel support |
| `awq_marlin`, `gptq_marlin`, `marlin` ⇒ Ampere+ and typically `kv_cache_dtype ∈ {auto, fp8_e4m3}` | Marlin kernel constraints |
| MLA attention backends (`FLASHMLA`, `CUTLASS_MLA`, `TRITON_MLA`, `FLASH_ATTN_MLA`, `FLASHINFER_MLA`) ⇒ MLA architecture (DeepSeek etc.) | architecture-specific |
| `ROCM_*` backends ⇒ ROCm platform only | platform |
| `CPU_ATTN`, `TORCH_SDPA` ⇒ CPU or fallback target | platform |
| `enable_expert_parallel=True` ⇒ MoE model | architecture |
| `tensor_parallel_size` must divide `num_kv_heads` (and `num_attention_heads`) | sharding |
| `data_parallel_size * tensor_parallel_size * pipeline_parallel_size <= available GPUs` | placement |
| `draft_tensor_parallel_size ∈ {1, target_tp}` | speculative decoding constraint (`SpeculativeConfig`) |
| `num_speculative_tokens > 0` when speculative decoding is enabled | `gt=0` on field |
| `enforce_eager=True` disables both CUDA graphs and torch.compile | runtime guard |
| `CompilationConfig.mode=3` (VLLM_COMPILE) ⇒ V1 engine | engine gating |
| `cudagraph_mode="FULL"` incompatible with backends needing host-side dynamism (e.g. some prefill paths) | kernel constraint |
| `cudagraph_mode="FULL_DECODE_ONLY"` practically requires `enable_chunked_prefill=True` | prefill runs outside graph |
| `attention_backend=TREE_ATTN` pairs with eagle/medusa tree speculative decoding | design intent |
| `attention_backend=FLASHMLA` and some MLA backends typically require `block_size=64` | kernel constraint (verify per build) |
| `quantization="bitsandbytes"` not compatible with all MoE paths | loader limitation |
| `gpu_memory_utilization ∈ (0, 1]` | dataclass `gt=0, le=1` |
| `load_format` must match on-disk checkpoint encoding | loader |

---

## Known Under-Documented Areas

- **`docs/configuration/engine_args.html` and `docs/configuration/optimization.html`** returned HTTP 403 from WebFetch against `docs.vllm.ai`. Defaults and help strings were recovered from the GitHub markdown source and dataclass files. Rendered CLI help strings (argparse `help=...` metadata in `arg_utils.py`) were only partially visible in the fetched slice — exact CLI descriptions per flag were not captured verbatim.
- **`swap_space`**: not present in the `vllm/config/cache.py` excerpt that was returned. vLLM's long-standing default is `4` GiB per worker, but this catalog treats the exact default as needs-verification.
- **`preemption_mode` CLI exposure**: the scheduler dataclass excerpt did not show the field; V1's default is RECOMPUTE per the optimization guide. Whether this is currently exposed as a top-level CLI flag versus only settable through internal config needs verification against the latest `arg_utils.py` help block.
- **`num_speculative_tokens` as a top-level CLI flag**: source inspection shows it is carried inside the `--speculative-config` JSON. Any standalone `--num-speculative-tokens` flag, if present, may be a legacy shim — treat the JSON path as canonical.
- **`max_seq_len_to_capture` vs `max_cudagraph_capture_size`**: the dataclass field was renamed. The CLI flag name exposed by the current argparse in `arg_utils.py` was not fully visible in the fetched slice; both names are present in recent vLLM code.
- **Attention backend × block size**: exact allowed block sizes per backend (especially MLA and FLASHINFER) depend on the vLLM build. The catalog flags "commonly 64" for MLA but autoinfer should probe per backend at startup rather than hardcode.
- **Quantization × backend × dtype matrix**: the full cross-product of (quantization method, attention backend, kv_cache_dtype) has kernel-level constraints that are not exhaustively documented in any single upstream page. A runtime probe is the safer source of truth than a static table.
- **`compilation_config` fine-grained knobs** (`splitting_ops`, `custom_ops`, `use_inductor_graph_partition`): defaults documented, but the performance impact of non-default values is not quantified in the public optimization guide; treat as L1-advanced and leave at defaults unless a workload regression is observed.
- **Deprecated quantization methods** (`tpu_int8`, `fbgemm_fp8`, `fp_quant`): listed as deprecated in source but still callable. Autoinfer should exclude them from its search space.
- **`max_num_seqs` / `max_num_batched_tokens` "smart defaults"**: `EngineArgs` passes `None` and resolves via `_set_default_max_num_seqs_and_batched_tokens_args()`, which inspects hardware and usage context. Autoinfer should record the resolved values for reproducibility rather than assume the dataclass defaults (128 / 2048) always apply.
