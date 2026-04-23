# SGLang (raw, 2026-04-22)

Primary: arxiv [2312.07104](https://arxiv.org/abs/2312.07104), NeurIPS 2024.
Blog: [LMSYS 2024-01-17](https://www.lmsys.org/blog/2024-01-17-sglang/).
Repo: [sgl-project/sglang](https://github.com/sgl-project/sglang).

## What it is
Two-part system: a frontend embedded DSL (Python) for expressing structured
LLM programs with control flow, multiple generation calls, and parallelism,
and a runtime with RadixAttention + compressed-FSM constrained decoding +
API-aware scheduler.

## Core idea — RadixAttention
Persist KV cache across generation calls, keyed by token prefix. All active
sequences become leaves of a radix tree whose internal edges are labeled by
token sequences of variable length. LRU eviction at the node level. At prefill,
the longest matching prefix is reused without recomputation.

Implications:
- Prefix-sharing workloads (few-shot prompting, agent scratchpads, RLHF
  rollouts with identical system prompts) see O(unique-suffix) prefill.
- Memory is bounded by the usual paged-attention budget; radix structure is
  metadata overhead.
- Hit-rate depends on request ordering — the scheduler sorts to maximize reuse.

## Claimed wins
Up to 6.4× throughput vs prior SOTA on mixed workloads; 5× faster with
RadixAttention on specific prefix-heavy benchmarks (LMSYS blog).

## Why it matters for autoinfer
Shows that **caching policy** (not just kernel speed) is a first-class lever.
An autoinfer search loop that tunes only kernels will miss the RadixAttention-class
of gains. The policy surface must expose caching strategy, scheduler ordering,
and request batching together.

## Open questions for us
- How does RadixAttention hit-rate change across workloads (chat, code, RAG,
  agents)? Is there a workload fingerprint that predicts cache value?
- Interaction with speculative decoding — does reusing prefix KV invalidate
  draft-model speculation when the draft differs?
- Can we auto-discover the cache policy per-tenant? (This is an autoinfer
  candidate axis.)
