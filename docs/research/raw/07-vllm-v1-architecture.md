# vLLM V1 architecture — full transcription (raw, 2026-04-23)

Primary source: [Aleksa Gordić — "Inside vLLM: Anatomy of a High-Throughput
LLM Inference System"](https://www.aleksagordic.com/blog/vllm)
Published 2025-08-29. Base commit `42172ad` (2025-08-09).
Cross-post: [blog.vllm.ai/2025/09/05/anatomy-of-vllm.html](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html)
(already cited under C8 in `references-L1-engine-config.md`).
Repo: [vllm-project/vllm](https://github.com/vllm-project/vllm).

This note is a **scholastic transcription** of the article for the autoinfer
research corpus — class names, call graphs, formulas, env vars, flag
surfaces preserved verbatim where possible, with WHY / WHAT / HOW pulled
out per subsystem and autoinfer implications tagged at the end.

---

## 0. Why this article matters to autoinfer

It is the most complete public description of the V1 engine at the level
of **named classes, methods, CLI flags, and env vars**. Every L1 knob the
autoinfer engine-config layer searches is named here in the live API.
vLLM V1 is also the published baseline the project targets (per
`CLAUDE.md` and `references-L1-engine-config.md`). Nothing in this post
contradicts C1/C5/C8/C9 — it strengthens them by pinning the knob names
to current mainline code.

---

## 1. Engine anatomy

### 1.1 What — top-level shape
`LLM` is the user-facing class (offline, single-process). It wraps an
`LLMEngine`, which contains four elements:

1. **Configuration** — `ModelConfig`, `CacheConfig`, `ParallelConfig`, etc.
2. **Input processor** — tokenizes + validates, emits `EngineCoreRequest`.
3. **Engine core client** — `InprocClient` (≈ synchronous, single-process
   `EngineCore`) scales up to `DPLBAsyncMPClient` (data-parallel,
   load-balanced, async, multi-process).
4. **Output processor** — converts raw engine outputs to `RequestOutput`.

### 1.2 What — `EngineCore` internals
Inside the core:

- **Model executor** — `UniProcExecutor` (1 GPU) or `MultiProcExecutor`
  (TP/PP, one daemon process per rank).
- **Scheduler** — FCFS or priority; maintains `waiting` / `running` queues.
- **KV-cache manager** — `KVCacheManager` + `KVCacheCoordinator`;
  paged attention blocks.
- **Structured output manager** — `StructuredOutputManager`, FSM / grammar
  compilation via backends (primary: `xgrammar`).

Request object state machine: `WAITING`, `RUNNING`, `WAITING_FOR_FSM`.

### 1.3 How — worker init (three procedures, in order)

**(a) Init device**
- Assign CUDA device (e.g. `cuda:0`).
- Verify dtype support (bf16, fp32, …).
- Check VRAM budget vs `gpu_memory_utilization` (default 0.9 → reserve 90%
  for weights + KV cache + activations).
- Set up distributed settings (DP / TP / PP / EP).
- Instantiate `ModelRunner` (sampler, KV cache ref, forward buffers).
- Instantiate `InputBatch` (CPU-side `input_ids`, `positions`, block
  tables, sampling metadata).

**(b) Load model**
- Instantiate model architecture.
- Load weights.
- `model.eval()` (PyTorch inference mode).
- Optional: `torch.compile()` unless `--enforce-eager`.

**(c) Init KV cache**
- Retrieve per-layer KV-cache spec (`FullAttentionSpec` or hybrid via **Jenga**
  for heterogeneous layers).
- Run dummy forward pass to profile activation memory; snapshot GPU memory.
- Allocate + reshape + bind KV tensors to attention layers.
- Prepare attention metadata (FlashAttention backend etc.).
- Unless `--enforce-eager`: **capture CUDA graphs** for each warmup
  batch size.

Key env vars:
- `VLLM_USE_V1=1` — force V1 engine.
- `VLLM_ENABLE_V1_MULTIPROCESSING=0` — single-process mode
  (also the determinism lever cited under C3).
- `CUDA_VISIBLE_DEVICES` — GPU assignment.

### 1.4 How — `generate` loop (three stages per step)

Stage 1 — **Schedule**:
1. Walk `running` queue, pick decodes, compute new tokens.
2. Call `allocate_slots()` for each chosen request; deduct from token budget.
3. Walk `waiting` queue for prefills; `get_computed_blocks()` consults
   prefix cache; pop to `running` with `status=RUNNING`.

Stage 2 — **Forward pass**:
1. Update request states; prune finished.
2. Copy CPU→GPU buffers; compute positions; build `slot_mapping`.
3. Execute model in eager or captured-CUDA-graph mode.
4. Gather last-token hidden states; compute logits.
5. Sample (greedy / temperature / top-p / top-k / beam).

Stage 3 — **Postprocess**:
1. Append token IDs; detokenize.
2. Check stop conditions: length limit, EOS, stop IDs, stop string.
3. On finish: return blocks to `free_block_queue`; emit output.

---

## 2. Paged attention and KV cache

### 2.1 What — block layout
Per-block bytes, non-MLA path:
```
block_bytes = 2 * block_size_tokens * num_kv_heads * head_size * dtype_bytes
```
Default `block_size_tokens = 16`. `2` = one K + one V tensor.

### 2.2 What — data structures
- `free_block_queue` — doubly-linked list of available blocks (scales to
  hundreds of thousands).
- `req_to_blocks` — dict `request_id → list[Block]`.
- `cached_block_hash_to_block` — dict `BlockHash → Block` for prefix cache.

### 2.3 How — `allocate_slots(request, new_tokens)`
1. `n = ceil(new_tokens / block_size_tokens)`.
2. If `free_block_queue` has < n: attempt **recompute preemption** (evict
   low-priority running requests).
3. Pop first n blocks from `free_block_queue`; append to
   `req_to_blocks[request_id]`.

### 2.4 Why
Blocks decouple physical KV memory from logical sequence contiguity.
Enables (a) non-fragmenting reuse across sequences, (b) prefix sharing
via hash-keyed block lookup, (c) preemption by evicting whole blocks
rather than whole requests.

---

## 3. Chunked prefill

**Flag:** `long_prefill_token_threshold` (positive integer, tokens per step).

**What:** splits long prompts into chunks ≤ the token budget. Example:
```
prompt = 24 tokens, chunk = 8
step 1: prefill 8 tokens
step 2: prefill 8 tokens
step 3: prefill 8 tokens + sample 1 token  (>= 3 steps to TTFT)
```

**Why:** stops a single long prompt from starving all decodes in the
same iteration. V1 additionally allows **mixed prefill + decode in one
batch** (V0 could not). Scheduling becomes a superposition, not a regime
selector.

**Autoinfer implication:** an L1 surrogate that models prefill and decode
as disjoint regimes is wrong by construction. The step-time model must
account for `chunked_prefill_tokens + decoded_requests` in one forward.

---

## 4. Prefix caching

**Enabled by default.** Disable with `enable_prefix_caching=False`.

### 4.1 How — hashing
`hash_request_tokens()` walks the token stream in `block_size_tokens`
chunks. Each `BlockHash` is:
```
BlockHash_i = hash(BlockHash_{i-1}, tokens_i, metadata)
metadata = (MM_hash, LoRA_ID, cache_salt)
```
Multimodal hash, LoRA ID, and cache salt are mixed in when present →
cache entries are isolated by modality / adapter / tenant.

### 4.2 How — lookup and lifecycle
1. `find_longest_cache_hit()` — linear search in `cached_block_hash_to_block`.
2. On hit: reuse block; increment ref-count.
3. On miss: `allocate_slots()`; compute; `coordinator.cache_blocks()`
   installs the `BlockHash → Block` mapping.
4. On request finish or eviction: decrement ref-count; when zero and the
   block is repurposed from `free_block_queue`, clear its hash entry.

### 4.3 Why
First-request cost is unchanged, subsequent requests with matching
prefix skip prefill. Autoinfer-relevant axes: `block_size_tokens`
(hash granularity vs memory waste), whether MM/LoRA/salt are in the hash
(tenant isolation), admission policy (always-cache vs budgeted).

---

## 5. Guided decoding (FSM + grammar)

### 5.1 What
Constrains token generation to a regular or context-free grammar using
a finite-state machine. Primary backend: **xgrammar**.

### 5.2 How
1. New request → status `WAITING_FOR_FSM`.
2. `grammar_init()` selects backend; grammar compiles **asynchronously**.
3. On compile done → status `WAITING` (schedulable).
4. Scheduler adds request to `structured_output_request_ids`.
5. After forward pass:
   a. `StructuredOutputManager` asks backend for `_grammar_bitmask`.
   b. `xgr_torch_compile` expands the bitmask 32× (one bit per allowed
      token) to vocabulary size.
   c. Disallowed logits set to `-inf`.
   d. Sample.
   e. `rejection_sampler.accept_tokens()` advances FSM.

### 5.3 Why
Bitmask expansion matters: 32-bit integers encode 32 allowed/disallowed
token positions each. For `vocab_size > 32` this concatenates across
many 32-bit words, then expands to a `[vocab_size]` mask. Overhead is
sampler-side, not model-side — it is part of the tail of step latency,
not FLOPs.

---

## 6. Speculative decoding

### 6.1 What
Three V1-native backends:

1. **N-gram** — match last `prompt_lookup_max` tokens against history
   in the same request; if match found, propose the k tokens that
   followed. Falls back to `prompt_lookup_min`. Current implementation
   returns k tokens **after the first match**.
2. **EAGLE** — replace the transformer stack of the drafter with a
   lightweight MLP; reuse the target model's embeddings and LM head.
3. **Medusa** — train auxiliary linear heads on top of the target model
   to predict k tokens in parallel.

### 6.2 How — verify + rejection sample
Given k draft tokens and the target model forward on `context + k`:
```
for i in 0..k-1:
  if p_large(tok_i) >= p_draft(tok_i):
      accept
  else:
      accept with probability p_large(tok_i) / p_draft(tok_i)
  stop at first rejection
if all k accepted:
  sample (k+1)th token "for free" from p_large at position k
else at rejection index r:
  resample from (p_large - p_draft).clip(min=0).normalized()
```
Statistically equivalent to vanilla autoregressive sampling — the
distribution is preserved by construction (this is the "lossless"
property cited under C2).

### 6.3 Config surface
```python
speculative_config = {
    "method": "ngram",                 # "ngram" | "eagle" | "medusa"
    "prompt_lookup_max": 5,
    "prompt_lookup_min": 3,
    "num_speculative_tokens": 3,
}
llm = LLM(model=..., speculative_config=speculative_config)
```

### 6.4 Why / where autoinfer should care
Acceptance rate is workload-dependent (see C2 evidence). Speedup ≠
throughput gain under load (TurboSpec result). Backend choice is a
**categorical L1 knob with high reward variance** — good surrogate
stress test. No guidance in the post about when to pick which backend;
empirical gap.

---

## 7. Disaggregated prefill / decode (P/D)

### 7.1 What
Separate instances for prefill and decode because their bottlenecks
differ (prefill compute-bound, decode bandwidth-bound). KV is shipped
between them via connectors.

### 7.2 Connectors
- `SharedStorageConnector` — local filesystem (debug).
- `LMCache` — production; uses **NVIDIA NIXL** backend.
- Interface is **not yet stable**; breaking changes expected.

### 7.3 How — config
```python
kv_transfer_config = KVTransferConfig(
    kv_connector="SharedStorageConnector",
    kv_role="kv_both",     # or "kv_prefill", "kv_decode"
    kv_connector_extra_config={"shared_storage_path": "local_storage"},
)
```

### 7.4 How — flow
Scheduler stage:
- `connector.get_num_new_matched_tokens()` — check external cache.
- `connector.update_state_after_alloc()` — record requests with hits.
- `connector.build_connector_meta()` — mark prefill requests
  `is_store=True`, decode requests `is_store=False`.

Forward-pass context manager:
- `__enter__`: `connector.start_load_kv()` — decode fetches from store.
- `__exit__`: `connector.wait_for_save()` — prefill uploads to store.

Layer-by-layer transfer is optional (KV can move before/after each
attention layer, overlapping transfer with compute).

### 7.5 Why this is L2, not L1
The choice "one engine per role vs co-located" is a **topology**
decision — it changes the process graph and network path, not the
engine config. So P/D disagg belongs to the L2 layer of the autoinfer
architecture, with the prefill:decode instance ratio and connector
choice as L2 axes.

---

## 8. `MultiProcExecutor` (TP / PP launch)

### 8.1 How — spawn
1. `MultiProcExecutor.__init__` creates `rpc_broadcast_mq` (shared-memory
   queue).
2. Loop `world_size` times (e.g. TP=8):
   a. Create reader/writer pipe for this worker.
   b. Spawn daemon process via `WorkerProc.make_worker_process()`.
   c. Child runs `WorkerProc.worker_main()`.
3. Each worker:
   a. Determine rank (`0` = driver, else regular).
   b. Create `rpc_broadcast_mq` handle (shared) + `worker_response_mq`
      (local).
   c. Send `worker_response_mq` handle back to parent via pipe.
4. Parent blocks until all workers have responded, then unblocks.
5. Workers enter busy loop: block on `rpc_broadcast_mq.dequeue()`, run
   the work, enqueue result to `worker_response_mq`.

### 8.2 How — runtime
- Engine calls `executor.execute_model(...)`.
- Executor enqueues work to `rpc_broadcast_mq` (non-blocking; all
  workers see it).
- Executor waits on the **output rank**'s `worker_response_mq` only.
- From the engine's perspective, the interface is identical to
  `UniProcExecutor`.

### 8.3 Why
Daemon-per-rank + shared-memory queues keeps the engine loop
single-threaded and deterministic while scaling compute. The driver-rank
(rank 0) pattern is borrowed from Megatron-LM / torchrun conventions.

---

## 9. Distributed serving (multi-node)

### 9.1 What — deployment
Two H100 nodes, TP=4, DP=4:

Headless (backend) node:
```bash
vllm serve <model> \
  --tensor-parallel-size 4 \
  --data-parallel-size 4 \
  --data-parallel-size-local 2 \
  --data-parallel-start-rank 0 \
  --data-parallel-address <master-ip> \
  --data-parallel-rpc-port 13345 \
  --headless
```

API server node:
```bash
vllm serve <model> \
  --tensor-parallel-size 4 \
  --data-parallel-size 4 \
  --data-parallel-size-local 2 \
  --data-parallel-start-rank 2 \
  --data-parallel-address <master-ip> \
  --data-parallel-rpc-port 13345
```

### 9.2 How — headless bring-up (`CoreEngineProcManager`)
Spawns `data-parallel-size-local` processes, each running
`EngineCoreProc.run_engine_core` (DP variant: `DPEngineCoreProc`):

1. Create `input_queue` + `output_queue` (`queue.Queue`).
2. ZMQ **DEALER** socket handshake with frontend; receive DP-coord
   address.
3. Init DP group on **NCCL** backend.
4. Instantiate `EngineCore` with `MultiProcExecutor(TP=4)`.
5. Create `ready_event` (`threading.Event`).
6. Start input daemon thread: `process_input_sockets(..., ready_event)`.
7. Start output daemon thread.
8. Main thread waits on `ready_event` until coordination done.
9. Send `"ready"` with `num_gpu_blocks` metadata to frontend.
10. All three threads enter steady-state loops.

Steady state:
- Input thread — block on input socket; on request decode and
  `input_queue.put_nowait(...)`.
- Main thread — block on `input_queue.get()`; feed engine; drive
  `engine_core.step()`; push results to `output_queue`.
- Output thread — block on `output_queue.get()`; send via output
  socket.

Extra mechanics:
- **DP wave counter** — increments when engines quiesce and resume.
  Used for cross-DP lockstep bookkeeping.
- **Dummy steps for lockstep** — if any DP replica has work, *all*
  replicas execute a step; MoE all-to-alls require matched
  participation.

### 9.3 How — API server (`AsyncLLM` + `DPLBAsyncMPClient`)
1. Instantiate `AsyncLLM` (asyncio wrapper over the client).
2. `DPLBAsyncMPClient` spawns:
   a. `DPCoordinator` process.
   b. `CoreEngineProcManager`.
   c. `outputs_queue` (`asyncio.Queue`).
   d. Asyncio tasks: `process_outputs_socket`, `output_handler`,
      `run_engine_stats_update_task`.

`DPCoordinator` duties:
- Periodically broadcast LB state (queue sizes, waiting/running counts).
- Handle `SCALE_ELASTIC_EP` commands (Ray backend only).
- Send `START_DP_WAVE` events; report wave-state transitions.

FastAPI routing: `OpenAIServingCompletion` for `/completion` and
`/chat/completion`, served via Uvicorn.

### 9.4 How — load balancing score
```
score = len(waiting) * 4 + len(running)
```
Pick the engine with lowest score for the next request. Multi-API-server
LB is at the OS socket layer — transparent to the application.

### 9.5 How — request lifecycle (happy path)
```
1. POST /v1/completions              (curl)
2. FastAPI route create_completion   (Uvicorn)
3. tokenize; build metadata (request_id, sampling_params, timestamp)
4. AsyncLLM.generate → DPAsyncMPClient.add_request_async
5. get_core_engine_for_request       (applies LB score)
6. ADD → chosen engine's input_socket
   6a. input thread decodes; input_queue.put_nowait(...)
   6b. main thread: add to engine; loop engine_core.step(...)
       until stop condition; push to output_queue
   6c. output thread: send over output_socket
7. AsyncLLM output tasks propagate tokens back to create_completion
8. FastAPI wraps + returns JSONResponse
```

---

## 10. Benchmarking and metrics

### 10.1 Metric definitions (quote-level)
| Metric | Definition |
|--------|-----------|
| **TTFT** | Time from submission until first output token received. |
| **ITL** | Time between consecutive tokens (e.g. tok_{i-1} → tok_i). |
| **TPOT** | Average ITL across all output tokens of a request. |
| **E2E latency** | TTFT + Σ ITL; total request time. |
| **Throughput** | Tokens/sec or requests/sec. |
| **Goodput** | Throughput of requests that meet SLOs (e.g. TTFT ≤ X ms, p99 E2E ≤ Y ms). |

### 10.2 Roofline framing
```
t_step = FLOPs_step / P_kernel
```
- Below **saturation batch** `B_sat`: HBM-bound; step time nearly flat
  in batch size (1 token vs 10 tokens per step ≈ same time).
- Above `B_sat`: compute-bound; step time ~linear in batch size; each
  token added to a batch adds to its own ITL.

Practical consequence: latency–throughput is **piecewise linear around
B_sat**, not smooth. Any L1 surrogate should encode this.

### 10.3 `vllm bench` modes
- **latency** — default 32-token input, 128 output, batch 8, report E2E.
  ```bash
  vllm bench latency --model <m> --input-tokens 32 \
    --output-tokens 128 --batch-size 8
  ```
- **throughput** — fixed 1000 ShareGPT samples, QPS=∞, report
  input/output/total tokens-per-second and requests-per-second.
- **serve** — launches server, Poisson (or Gamma) arrivals, measures
  all metrics; optional server-side max-concurrency semaphore
  (e.g. 64 concurrent).

### 10.4 Auto-tune script
In-tree. Drives the `serve` benchmark to find configs meeting SLO
targets (e.g. "maximize throughput under p99 E2E < 500 ms"). Returns
a suggested config. This is **the natural public baseline for
autoinfer's L1 loop** — the honest comparable is "did we beat vLLM's
shipped auto-tune at its own game on the same model/hardware?"

CI benchmark config: `.buildkite/nightly-benchmarks/tests`.

---

## 11. Features orthogonal to the main flow

The post catalogs features that layer on top without fundamentally
changing the step loop:

- **Hardware:** TPUs, AWS Neuron (Trainium / Inferentia).
- **Attention variants:** MLA, sliding-window, attention-free, ALiBi.
- **Model classes:** MoE (with EPLB — expert parallel load balancing),
  encoder-decoder (Whisper), pooling/embedding, multimodal (m-RoPE),
  state-space (Mamba, Mamba-2, Jamba).
- **Adapters:** LoRA.
- **KV cache:** **Jenga** — hybrid KV memory management for
  heterogeneous layers (attention + SSM + linear etc.).
- **Sampling:** beam search (beyond temperature / top-p / top-k).
- **Parallelism:** TP, PP, **SP** (sequence parallelism).
- **Experimental:** async scheduling (overlapping scheduler with
  forward pass).

---

## 12. Autoinfer-specific implications

1. **L1 knob catalog is pinned.** Every axis the L1 adapter targets is
   named here in live API:
   - scheduling: `max_num_seqs`, `max_num_batched_tokens`,
     `long_prefill_token_threshold`, `enable_prefix_caching`.
   - memory: `gpu_memory_utilization`, `block_size` (tokens).
   - parallelism: `tensor_parallel_size`, `pipeline_parallel_size`,
     `data_parallel_size` (+ `-local`, `-start-rank`, `-address`,
     `-rpc-port`, `--headless`).
   - speculation: `method` ∈ {ngram, eagle, medusa},
     `num_speculative_tokens`, `prompt_lookup_{min,max}`.
   - compile: `--enforce-eager` (disables CUDA graphs),
     implicit `torch.compile` toggle at load.
   - determinism lever: `VLLM_ENABLE_V1_MULTIPROCESSING=0`, seeded
     sampling (connects to C3).

2. **L2 surface is also pinned.** Disaggregation:
   `kv_connector` ∈ {SharedStorageConnector, LMCache(+NIXL), …},
   `kv_role` ∈ {kv_both, kv_prefill, kv_decode}, prefill:decode
   instance ratio, per-layer vs end-of-pass transfer.

3. **Harness shape matches.** `vllm bench serve` with Poisson arrivals
   and a concurrency cap is the right default workload driver for
   the shared substrate (P2). Metric definitions (TTFT/ITL/TPOT/E2E/
   throughput/goodput) should be the reporting contract.

4. **Auto-tune is the honest baseline.** Our L1 loop should be measured
   head-to-head against `benchmarks/auto_tune` on a fixed
   (model, GPU, workload) tuple.

5. **Mixed P+D batches** invalidate disjoint-regime surrogates. Step-time
   model must take (`chunked_prefill_tokens`, `num_decoded_requests`)
   jointly.

6. **Piecewise-linear step time around `B_sat`** is a structural prior
   the surrogate can exploit. Uninformed BO will rediscover it slowly.

7. **Categorical spec-dec backend is a high-variance reward.** Good
   surrogate stress test; also candidate for the LLM-operator
   proposal step (P7).

8. **Hash-granularity interaction** — `block_size_tokens` controls
   both KV fragmentation and prefix-cache hit granularity. Not
   independent; pair-tune.

9. **Determinism trail for C3.** The post explicitly names the knobs
   that enable/disable deterministic inference
   (`VLLM_ENABLE_V1_MULTIPROCESSING=0` + batch-invariant kernels +
   seeded sampling). The quality gate (P8) needs these on for the
   reference replica.

10. **Jenga hybrid KV** is already production for Mamba/Jamba-class
    models — worth a separate raw note when we extend L1 beyond
    pure-transformer configs.

---

## 13. Open questions for us

- **Effective L1 dimensionality after pinning.** The post lists the
  knobs but doesn't quantify how many are dead given
  (model, GPU, workload). Measuring the effective rank of the search
  surface is an autoinfer deliverable the post doesn't pre-empt.
- **LB score `4 * waiting + running` under skewed prompt lengths.**
  The constant 4 is hardcoded; with long-prompt workloads waiting
  requests cost more than the score implies. Candidate L2 axis.
- **`block_size_tokens` × prefix-cache hit × chunked prefill**
  3-way interaction — post describes each in isolation.
- **Spec-dec backend selection rules.** Post lists n-gram/EAGLE/Medusa;
  gives no heuristics for when to pick which. Autoinfer's job.
- **DP dummy-step cost.** For MoE models the lockstep forces idle DP
  ranks to do dummy work; how much this caps the useful DP size on
  bursty workloads is not quantified in the post.

---

## 14. References (from the post)

Primary:
1. vLLM repo: https://github.com/vllm-project/vllm
2. V1 engine guide: https://docs.vllm.ai/en/latest/usage/v1_guide.html
3. V0 deprecation: https://github.com/vllm-project/vllm/issues/18571

Foundational:
4. Vaswani et al., "Attention Is All You Need" (2017):
   https://arxiv.org/abs/1706.03762
5. Kwon et al., "PagedAttention" (2023):
   https://arxiv.org/abs/2309.06180
6. Yu et al., "Orca: distributed serving" (OSDI 2022):
   https://www.usenix.org/conference/osdi22/presentation/yu

Model-architecture:
7. DeepSeek-V2 (MLA): https://arxiv.org/abs/2405.04434
8. Jenga (hybrid KV, 2025): https://arxiv.org/abs/2503.18292

Structured / speculative:
9. XGrammar (Lin et al., 2024): https://arxiv.org/abs/2411.15100
10. Leviathan et al., Speculative sampling (2023):
    https://arxiv.org/abs/2302.01318
11. EAGLE (Li et al., 2024): https://arxiv.org/abs/2401.15077
12. Medusa (Cai et al., 2024): https://arxiv.org/abs/2401.10774

Ecosystem:
13. LMCache: https://github.com/LMCache/LMCache
14. gpt-fast (Meta / PyTorch): https://github.com/meta-pytorch/gpt-fast

---

## 15. Claim / principle traceability

- **C1** (engine-config surface is large and real) — direct evidence;
  knob surface enumerated.
- **C2** (spec-dec wins are acceptance-rate × load-dependent) — backend
  catalog but no heuristics; gap.
- **C3** (batch-variance and determinism) — `VLLM_ENABLE_V1_MULTIPROCESSING`
  and seeded sampling levers confirmed.
- **C8** (minimal rigs don't reproduce V1 feature surface) —
  direct reinforcement; this post is the feature-surface description
  that C8 contrasts against nano-vLLM.
- **C9** (reference-replica gate must cover mixed batches) — V1
  mixes prefill + decode by default, so the gate cannot be
  regime-specific.
- **P1** (three layers, day one) — P/D disagg is L2, engine config is
  L1; the post pins the split cleanly.
- **P3** (layers are adapters onto a shared harness) — `vllm bench`
  is the harness shape.
- **P4** (cross-layer stale-signal invalidation) — DP wave counter is
  a ready-made signal for the L2→L1 stale-mark path.
- **P7** (hybrid policy) — spec-dec backend categorical is a good
  LLM-operator target.
