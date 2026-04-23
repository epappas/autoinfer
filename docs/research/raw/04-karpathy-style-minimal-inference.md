# Karpathy-style minimal inference (raw, 2026-04-22)

The equivalent of nanoGPT/llm.c for *inference*. These are the "substrates you
can hold in your head" — essential for a hypothesis-forming phase because they
make the engine's invariants legible.

## Karpathy's own line
- [karpathy/llm.c](https://github.com/karpathy/llm.c): LLMs in raw C/CUDA. Not
  an inference engine per se; `dev/cuda` has a documented ladder of attention /
  GEMM kernels from naïve to fused, useful as a pedagogical reference for
  kernel shapes.
- [karpathy/nanochat](https://github.com/karpathy/nanochat): successor to
  nanoGPT; claims "minimal, from scratch, full-stack training/inference
  pipeline" in a single dependency-minimal codebase. Implements KV-cache'd
  inference with simple pre-fill/decode. **This is the closest Karpathy has
  come to a nano-inference-engine.**
- [microgpt](http://karpathy.github.io/2026/02/12/microgpt/): Feb 2026 post
  (need to fetch).

## nano-vLLM — GeeeekExplorer
Repo: [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).

- ~1200 lines of Python.
- Reimplements: prefix caching, tensor parallelism, torch.compile, CUDA graphs.
- Modular: `LLMEngine`, `Scheduler`, `ModelRunner`, `Sequence`.
- Measured on RTX 4070 Laptop w/ Qwen3-0.6B: **outperforms full vLLM by ~5%**
  (1434 vs 1362 tok/s) on a synthetic 256-seq benchmark — likely due to
  reduced Python overhead, not fundamental algorithmic wins.
- HN discussion: [id=46855447](https://news.ycombinator.com/item?id=46855447).
- HuggingFace writeup: [Nano-vLLM meets Inference Endpoints](https://huggingface.co/blog/angt/nano-vllm-meets-inference-endpoints),
  [Introduction to nano-vLLM](https://huggingface.co/blog/zamal/introduction-to-nano-vllm).

## Forks / siblings
- [Wenyueh/MinivLLM](https://github.com/Wenyueh/MinivLLM): self-contained paged
  attention + flash attention implementations on top of nano-vllm.
- [changjonathanc/flex-nano-vllm](https://github.com/changjonathanc/flex-nano-vllm):
  FlexAttention-based for Gemma 2.

## Why this matters for autoinfer
Three reasons:

1. **Search needs a legible substrate.** Optimizing vLLM directly is possible
   but the search surface is huge and changes fast. nano-vLLM gives us a
   stable, readable baseline where we can enumerate the engine's axes
   (scheduler, KV policy, batch policy, kernel choice) and bound the search.

2. **The 5%-over-vLLM number is suspicious and useful.** If a 1200-line
   engine beats the production engine on a narrow benchmark, the delta is
   almost certainly in Python overhead or scheduling — not in kernel speed.
   That tells us where naive autoinfer gains will come from first.

3. **Karpathy's thesis — that you understand something by stripping it to
   its minimum — is the same thesis this project should adopt** for inference.
   Before we search over vLLM, we search over nano-vLLM and measure what the
   search actually moves.

## Open questions
- What does nano-vLLM drop compared to vLLM V1? (paged attention variants,
  speculative decoding, quantization, multimodal?) → read the source.
- How does nano-vLLM's scheduler compare to vLLM's V1 continuous-batching
  scheduler on long-running workloads with mixed sequence lengths?
- Is there a nano-SGLang? (None found. Could be worth building.)
