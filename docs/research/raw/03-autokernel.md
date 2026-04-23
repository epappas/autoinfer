# AutoKernel and LLM-driven kernel generation (raw, 2026-04-22)

## The pattern
A tight loop: LLM writes a kernel, fixed harness measures correctness + speed,
keep improvements, discard regressions, iterate. 300–400 experiments per
overnight run. Mirrors `autoresearch-rl`'s LLM-proposes / eval / keep-discard
loop, applied to GPU kernels.

## Active projects

### AutoKernel — RightNow AI
Repo: [RightNow-AI/autokernel](https://github.com/RightNow-AI/autokernel). Blog:
[AutoKernel: Automating GPU Kernel Optimization with LLM Agent Loops (Apr 2026)](https://earezki.com/ai-news/2026-04-06-rightnow-ai-releases-autokernel-an-open-source-framework-that-applies-an-autonomous-agent-loop-to-gpu-kernel-optimization-for-arbitrary-pytorch-models/).
Agent modifies a single `kernel.py`; benchmark harness is frozen. Profiles the
model first, then targets kernels by contribution to runtime. Supports Triton
and CUDA C++. Paper: arxiv [2603.21331](https://arxiv.org/html/2603.21331).

### KernelAgent — meta-pytorch
Repo: [meta-pytorch/KernelAgent](https://github.com/meta-pytorch/KernelAgent).
"Autonomous GPU Kernel Generation & Optimization via Deep Agents". Official
PyTorch-affiliated effort.

### KernelFalcon — PyTorch blog
[PyTorch blog: KernelFalcon](https://pytorch.org/blog/kernelfalcon-autonomous-gpu-kernel-generation-via-deep-agents/).
Deep-agent architecture for kernel generation.

### Survey
[awesome-LLM-driven-kernel-generation](https://github.com/flagos-ai/awesome-LLM-driven-kernel-generation).
Reference paper: arxiv [2601.15727](https://arxiv.org/abs/2601.15727) "Towards
Automated Kernel Generation in the Era of LLMs".

## Why it matters for autoinfer
Two non-obvious things:

1. **The frozen/mutable boundary already exists** for kernels. autoinfer doesn't
   need to invent its own harness here; it can reuse AutoKernel's for the
   kernel-level search layer and add engine-level and serving-level search layers
   on top.

2. **Kernel-search is a strict subset of inference-engine-search.** A complete
   autoinfer loop spans at least three layers: kernel (Triton/CUDA code),
   engine config (paging, batching, scheduling, caching policy), and deployment
   (hardware pairing, PD-disaggregation topology, replica count). AutoKernel
   solves only the bottom layer. This is where autoinfer can claim novelty —
   **joint search across layers**, not just kernel tuning.

## Open questions
- Can a kernel proposed by AutoKernel ever regress **quality** (e.g. numeric
  precision, batch-invariance)? How does the harness guard against that?
  (vLLM's `batch_invariance.md` is relevant.)
- Does the agent converge to a local optimum for a single shape, or does it
  discover shape-polymorphic kernels that cover the full attention workload?
- Cross-layer signal: if the engine changes batch size, does the kernel need
  re-optimizing? What's the coupling?
