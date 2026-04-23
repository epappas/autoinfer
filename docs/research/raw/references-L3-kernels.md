# L3 kernel-search evidence (2026-04-22)

## C5 — LLM-driven kernel search vs hand-tuned vLLM kernels

Claim: LLM-driven kernel search can match or beat hand-tuned vLLM kernels on
shape regions that vLLM's specialist authors have under-optimized; the
AutoKernel-family pattern (frozen benchmark harness + mutable `kernel.py` +
agent loop) is a viable alternative or complement to classical autotuning.

### LLM-driven kernel generation

- **AutoKernel: Autonomous GPU Kernel Optimization via Iterative Agent-Driven Search** (RightNow AI, 2026) — [arxiv 2603.21331](https://arxiv.org/abs/2603.21331) / [html](https://arxiv.org/html/2603.21331) / [github](https://github.com/RightNow-AI/autokernel)
  Agent loop: LLM mutates a single `kernel.py`, frozen five-stage harness validates (smoke, shape sweep, numerical stability, determinism, edge cases), then measures runtime. `torch.profiler` + Amdahl-ranking picks targets. Reported on H100: Triton kernels beat PyTorch eager by **5.29x on RMSNorm, 2.82x on softmax, 2.21x on cross-entropy**, and beat `torch.compile(max-autotune)` by **2.83x / 3.44x / 2.94x** respectively. This is the canonical "frozen harness + mutable kernel + agent" pattern C5 refers to.

- **KernelFalcon / KernelAgent** (Meta PyTorch, 2025–2026) — [PyTorch blog](https://pytorch.org/blog/kernelfalcon-autonomous-gpu-kernel-generation-via-deep-agents/) / [github](https://github.com/meta-pytorch/KernelAgent)
  Deep-agent architecture: orchestrator delegates to sub-agents (fusion-boundary extractor, Triton codegen, composer, validator). Claims **first open agentic system to hit 100% correctness on all 250 KernelBench L1/L2/L3 tasks** via parallel exploration with execution-based verification. "Rigid contracts" (typed interfaces per operation) enable autonomous operation.

- **KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta** (Meta, Dec 2025) — [arxiv 2512.23236](https://arxiv.org/abs/2512.23236)
  Production-scale agentic kernel coding for DLRM across NVIDIA, AMD, and Meta-internal accelerators. Operates across Triton, CuTe DSL, and lower-level languages. Reports **100% pass on all 250 KernelBench problems**, 100% correctness across 160 PyTorch ATen operators on 3 heterogeneous platforms, and reduction of dev time from weeks to hours with substantial speedups over PyTorch baselines. Confirms the pattern works on AMD, not just NVIDIA.

- **KernelBench: Can LLMs Write Efficient GPU Kernels?** (Stanford, Scaling Intelligence Lab, Feb 2025) — [arxiv 2502.10517](https://arxiv.org/abs/2502.10517) / [blog](https://scalingintelligence.stanford.edu/blogs/kernelbench/) / [github](https://github.com/ScalingIntelligence/KernelBench)
  The benchmark that defines the field. 250 PyTorch tasks, `fast_p` metric (fraction correct and >p× speedup vs PyTorch baseline). Paper baseline result: frontier models **match PyTorch in less than 20% of cases** in single-shot generation; most failures are compile/runtime errors, not perf regressions. This is the honest floor — KernelFalcon/KernelEvolve 100% results are agentic (multi-turn + verification), not single-shot.

- **CUDA-LLM: LLMs Can Write Efficient CUDA Kernels** (June 2025) — [arxiv 2506.09092](https://arxiv.org/abs/2506.09092) / [html](https://arxiv.org/html/2506.09092v1)
  Proposes Feature Search and Reinforcement (FSR): joint optimization of compilation, correctness (via extensive test cases), and runtime. Reports generated kernels outperform human-written code by up to **179x**. Caveat: the human baseline is not always expert-hand-tuned production code; interpret 179x as "best case against weak baselines".

- **Towards Automated Kernel Generation in the Era of LLMs** (Jan 2026, v2) — [arxiv 2601.15727](https://arxiv.org/abs/2601.15727) / [pdf](https://arxiv.org/pdf/2601.15727)
  Survey of LLM-based kernel generation. Structures the field into codegen methods, agentic workflows, datasets/benchmarks. Useful as a map of the state-of-the-art as of early 2026 and to corroborate that the "frozen harness + mutable kernel + iterative feedback" pattern has become standard.

- **awesome-LLM-driven-kernel-generation** (flagos-ai) — [github](https://github.com/flagos-ai/awesome-LLM-driven-kernel-generation)
  Living index of papers, repos, and benchmarks in this area — companion to arxiv 2601.15727.

- **Sakana AI — The AI CUDA Engineer** (Feb 2025) — [project page](https://sakana.ai/ai-cuda-engineer/) / [paper PDF](https://pub.sakana.ai/static/paper.pdf)
  Evolutionary meta-generation with LLM crossover + innovation archive. Claims **10–100x over PyTorch eager** on common ML ops, and **up to 5x vs existing production CUDA** on a subset. Translates 230/250 torch ops; releases 17k+ verified-kernel dataset. Important note: headline numbers were later disputed by independent reviewers (kernel reward hacking / caching artefacts); treat the extremes with caution, but the method and dataset remain relevant.

- **Astra: A Multi-Agent System for GPU Kernel Performance Optimization** (Stanford, Sep 2025) — [arxiv 2509.07506](https://arxiv.org/abs/2509.07506) / [pdf](https://cs.stanford.edu/~anjiang/papers/Astra.pdf)
  Starts from *existing SGLang CUDA kernels* (not naive PyTorch) and optimizes them with specialized LLM agents. Reports **average 1.32x speedup** over SGLang's hand-tuned kernels using o4-mini zero-shot. Most relevant evidence for C5: these are already-production-optimized kernels that LLMs still improve on — small in magnitude but real, and on hand-tuned baselines (the same regime as vLLM).

- **Measuring Automated Kernel Engineering** (METR, Feb 2025) — [METR blog](https://metr.org/blog/2025-02-14-measuring-automated-kernel-engineering/)
  Independent evaluation on a KernelBench-derived test set. With light scaffolding, the best model gives **average 1.8x** on KernelBench. METR explicitly flags limits: baselines are naive PyTorch (not hand-tuned CUDA), single-GPU only, inference only, fixed shapes. Reading: single-digit-x is the honest number *when the baseline is not already optimized*.

### Classical autotuning baselines

- **Triton `triton.autotune`** — [Triton docs](https://triton-lang.org/main/python-api/generated/triton.autotune.html) / [autotuner source](https://github.com/triton-lang/triton/blob/main/python/triton/runtime/autotuner.py)
  Configs over `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, `num_warps`, `num_stages`; benchmarks empirically per key. vLLM's V1 Triton kernels (fused MoE, LoRA, prefill attention) are tuned with this. Baseline to beat.

- **Achieving Platform Portability for vLLM by using Triton Autotuning** (IBM Research, Ray Summit 2024) — [IBM Research](https://research.ibm.com/publications/achieving-platform-portability-for-vllm-by-using-triton-autotuning-and-remembering-it)
  Documents that without autotune, Triton kernel performance varies by >1 order of magnitude across platforms. Quantifies exactly the kind of shape-specific delta LLM agents claim to capture.

- **NVIDIA CUTLASS Profiler + `nvMatmulHeuristics`** — [heuristics blog](https://developer.nvidia.com/blog/improving-gemm-kernel-auto-tuning-efficiency-on-nvidia-gpus-with-heuristics-and-cutlass-4-2/) / [profiler docs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html)
  Exhaustive build+profile search; with heuristic ranking, **96% of peak in ~150 minutes using only 16 candidate kernels** vs full exhaustive search. vLLM's Marlin/Machete and FP4/FP8 GEMMs sit inside this search space.

- **TVM Ansor (auto-scheduler)** — [TVM blog](https://tvm.apache.org/2021/03/03/intro-auto-scheduler) / Ansor paper [arxiv 2006.06762](https://arxiv.org/abs/2006.06762)
  Template-free auto-scheduling of tensor expressions; evolutionary + learned cost model. Established baseline for "can automated search beat hand-written?" on general DNN kernels.

- **TVM MetaSchedule** — [RFC](https://github.com/apache/tvm-rfcs/blob/main/rfcs/0005-meta-schedule-autotensorir.md) / [discuss](https://discuss.tvm.apache.org/t/rfc-meta-schedule-autotensorir/10120)
  Third-generation TVM autotuner; unifies AutoTVM + Ansor. Relevant as the classical counterpart to LLM-agent search: stochastic transformations + evolutionary search + cost model, no LLM required.

- **Halide Autoscheduler (SIGGRAPH 2019)** — [project](https://halide-lang.org/papers/autoscheduler2019.html) / [tutorial](https://halide-lang.org/tutorials/tutorial_lesson_21_auto_scheduler_generate.html)
  Beam search over a parameterized schedule space with a learned cost model. First autoscheduler to significantly outperform human experts on image-processing pipelines on average; a precedent for "classical search beats humans" that predates the LLM wave.

- **AITemplate** (Meta) — [github](https://github.com/facebookincubator/AITemplate)
  Python-to-CUDA/HIP C++ compiler with template-based codegen and profiler-driven tuning over CUTLASS configs; fuses GEMM + LayerNorm + epilogues. Meta reported **up to 12x on NVIDIA and 4x on AMD** vs PyTorch eager. Relevant as a production-deployed classical baseline for transformer inference.

- **vLLM Triton Attention Backend Deep Dive** (Mar 2026) — [vLLM blog](https://blog.vllm.ai/2026/03/04/vllm-triton-backend-deep-dive.html)
  Documents how vLLM's Triton attention path is autotuned across hardware. Primary source for understanding what classical autotuning already covers inside vLLM.

### vLLM kernel landscape and known gap regions

- **vLLM `csrc/` and CustomOp system** — [Paged Attention design](https://docs.vllm.ai/en/latest/design/paged_attention/) / [CustomOp design](https://docs.vllm.ai/en/latest/design/custom_op/) / [vllm-project/vllm](https://github.com/vllm-project/vllm)
  `csrc/attention/attention_kernels.cu` holds the custom PagedAttention kernel; CustomOp system dispatches per-platform (`forward_cuda`, `forward_rocm`, ...). vLLM ships hand-written CUDA/HIP for PagedAttention, quantized GEMMs (Marlin, Machete), RMSNorm, RoPE, activations, plus Triton for fused MoE / LoRA / prefill attention, plus FlashInfer and FlashAttention integrations as optional backends.

- **FlashInfer** — [FlashInfer launch blog (Feb 2024)](https://flashinfer.ai/2024/02/02/introduce-flashinfer.html) / [FlashInfer paper (arxiv 2501.01005)](https://arxiv.org/pdf/2501.01005) / [vLLM flashinfer backend](https://docs.vllm.ai/en/latest/api/vllm/v1/attention/backends/flashinfer/)
  Reports: single-request GQA decode **2–3x vs vLLM PagedAttention**; batch-64 GQA decode **3x vs vLLM PagedAttention**. This is direct evidence that vLLM's original PagedAttention kernel was under-optimized for GQA decode shapes — and was specifically replaced. This is also exactly the kind of gap an LLM kernel search could have found.

- **FLASHINFER slower than FLASH_ATTN on H100 (issue #9471)** — [github](https://github.com/vllm-project/vllm/issues/9471)
  Confirms the gap cuts both ways: FP8 throughput with FLASHINFER is significantly lower than FLASH_ATTN on certain H100 configs. Backend-selection heuristics in vLLM are themselves a shape-dependent optimization problem.

- **FlashMLA** (DeepSeek) — [github](https://github.com/deepseek-ai/FlashMLA) / [X announcement](https://x.com/deepseek_ai/status/1893836827574030466) / [vLLM integration blog](https://blog.vllm.ai/2025/09/29/deepseek-v3-2.html)
  MLA decode kernel for Hopper: **3000 GB/s memory-bound, 580 TFLOPS compute-bound on H800 SXM5**; improved to **up to 660 TFlops**. vLLM integrates FlashMLA plus DeepSeek's lightning-indexer and sparse-attention kernels for V3.2. Before this integration, long-context DeepSeek decode on vLLM used less-specialized paths — a concrete example of vLLM importing (rather than autoring) a specialist kernel.

- **Marlin / Machete quantized GEMM** — [MARLIN paper arxiv 2408.11743](https://arxiv.org/pdf/2408.11743) / [GPTQModel docs](https://docs.vllm.ai/en/latest/features/quantization/gptqmodel/)
  INT4×FP16 Marlin achieves ~**3.9x over FP16** at batch 1–32 on A10. Machete is the Hopper successor. Hand-tuned, highly shape-specific; main risk area is new hardware (Blackwell SM120) and new dtypes (MXFP4/NVFP4) where the hand-tuning has not caught up.

- **Blackwell FP4 / SM120 gap (TFLOPS Gap blog)** — [HuggingFace blog](https://huggingface.co/blog/apsys/blackwell-nvfp4-comparison) / [vLLM issue #31085 (SM120 NVFP4 MoE)](https://github.com/vllm-project/vllm/issues/31085) / [vLLM issue #30135 (MXFP4 fallback to Marlin)](https://github.com/vllm-project/vllm/issues/30135)
  **145 TFLOPS gap; SGLang 1.32x faster than vLLM on FP4 at batch=1.** Root causes named: missing Blackwell-specific CUTLASS schedules (FP4 warp specialization), insufficient kernel fusion (7 vs 5 memory passes), non-adaptive grid sizing for small batches. SM120 is not recognized by MXFP4 backend selection, falling back to Marlin and losing native-FP4 throughput. This is a textbook under-optimized shape region — exactly where an LLM agent loop could compete.

- **vLLM on GB10 MXFP4 slower than SGLang/llama.cpp** — [NVIDIA devforum](https://forums.developer.nvidia.com/t/vllm-on-gb10-gpt-oss-120b-mxfp4-slower-than-sglang-llama-cpp-what-s-missing/356651)
  Independent, user-reported confirmation of the MXFP4 gap on GB10/DGX Spark.

- **AMD MI300X MoE — default configs sub-optimal** — [vLLM issue #17619](https://github.com/vllm-project/vllm/issues/17619) / [ROCm vLLM 0.9.x blog](https://rocm.blogs.amd.com/software-tools-optimization/vllm-0.9.x-rocm/README.html) / [fused MoE Kimi-K2.5 optimization](https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html) / [PR #12408 MI300 tuned configs](https://github.com/vllm-project/vllm/pull/12408)
  vLLM emits "Using default MoE config. Performance might be sub-optimal!" for unknown (model, hardware) combos — i.e. the autotune result registry is incomplete. AITER provides ROCm-specific fused kernels; when not active, Triton fused MoE is the path and is under-tuned on MI300X. FP8 MoE on MI300X can be **slower than BF16 in steady-state decode** (issue #31475). Another clean under-optimized region.

- **Batch-1 decode memory-boundness** — [Mind the Memory Gap (arxiv 2503.08311)](https://arxiv.org/html/2503.08311v2) / [vLLM optimization docs](https://docs.vllm.ai/en/stable/configuration/optimization/)
  Independent analysis: vLLM attention at batch=1 shows inefficient DRAM access patterns, L1 hit rate <12%, L2 hit rate <2%. Chunked prefill and process-level PD-disaggregation are current mitigations. The underlying kernel is a candidate for LLM-driven kernel search on the small-batch regime.

- **vLLM Triton backend deep dive** (vLLM, Mar 2026) — [blog](https://blog.vllm.ai/2026/03/04/vllm-triton-backend-deep-dive.html) and [Enabling vLLM V1 on AMD GPUs With Triton](https://pytorch.org/blog/enabling-vllm-v1-on-amd-gpus-with-triton/)
  Official vLLM/PyTorch documentation of the Triton-based attention path and its cross-hardware autotune story. Use this as the authoritative "what vLLM ships today" source.

## Key measured numbers

- AutoKernel on H100, Triton kernels vs PyTorch eager: **5.29x RMSNorm, 2.82x softmax, 2.21x cross-entropy**; vs `torch.compile(max-autotune)`: **2.83x / 3.44x / 2.94x**. Source: [arxiv 2603.21331](https://arxiv.org/abs/2603.21331).
- KernelFalcon / KernelEvolve: **100% correctness on all 250 KernelBench L1/L2/L3** tasks; up to **17x vs PyTorch baselines**. Sources: [PyTorch blog](https://pytorch.org/blog/kernelfalcon-autonomous-gpu-kernel-generation-via-deep-agents/), [arxiv 2512.23236](https://arxiv.org/abs/2512.23236).
- KernelBench single-shot frontier-model performance: **<20% of cases match PyTorch**; most failures are compile/runtime errors. Source: [arxiv 2502.10517](https://arxiv.org/abs/2502.10517).
- METR independent KernelBench eval with light scaffolding: **average 1.8x speedup**. Source: [METR blog](https://metr.org/blog/2025-02-14-measuring-automated-kernel-engineering/).
- Astra vs hand-tuned SGLang CUDA kernels: **average 1.32x** with o4-mini zero-shot. Source: [arxiv 2509.07506](https://arxiv.org/abs/2509.07506).
- CUDA-LLM / FSR: **up to 179x** vs "general human-written code" (note: baseline quality weak). Source: [arxiv 2506.09092](https://arxiv.org/abs/2506.09092).
- Sakana AI CUDA Engineer: **10–100x** over PyTorch eager on common ops, **up to 5x** over "existing CUDA" on subset (disputed headline numbers; still useful as method reference). Source: [sakana.ai/ai-cuda-engineer](https://sakana.ai/ai-cuda-engineer/).
- FlashInfer vs vLLM PagedAttention: **2–3x single-request GQA decode, 3x at batch=64**. Source: [FlashInfer blog](https://flashinfer.ai/2024/02/02/introduce-flashinfer.html).
- FlashMLA on H800 SXM5: **3000 GB/s memory-bound, 580→660 TFLOPS compute-bound**. Source: [FlashMLA GitHub](https://github.com/deepseek-ai/FlashMLA).
- Marlin INT4×FP16 GPTQ: **≈3.9x over FP16** at batch 1–32 on A10. Source: [MARLIN paper](https://arxiv.org/pdf/2408.11743).
- Blackwell FP4, SGLang vs vLLM: **1.32x at batch=1, 145 TFLOPS gap**. Source: [HuggingFace blog](https://huggingface.co/blog/apsys/blackwell-nvfp4-comparison).
- AITemplate vs PyTorch eager: **up to 12x NVIDIA, 4x AMD**. Source: [AITemplate github](https://github.com/facebookincubator/AITemplate).
- CUTLASS + nvMatmulHeuristics: **96% of peak with 16 candidate kernels in ~150 min** vs exhaustive search. Source: [NVIDIA blog](https://developer.nvidia.com/blog/improving-gemm-kernel-auto-tuning-efficiency-on-nvidia-gpus-with-heuristics-and-cutlass-4-2/).
- Triton autotune across platforms: performance varies by **>1 order of magnitude** without autotune. Source: [IBM Research / Ray Summit 2024](https://research.ibm.com/publications/achieving-platform-portability-for-vllm-by-using-triton-autotuning-and-remembering-it).

## Honest assessment

The empirical picture, as of Apr 2026, is mixed but tilting in favour of C5 —
with caveats. Against **naive PyTorch** baselines, LLM-agent loops reliably
deliver low-single-digit to double-digit multiples, and frontier agentic
systems (KernelFalcon, KernelEvolve) now hit 100% correctness on all 250
KernelBench tasks, which is itself a step-change from the <20% single-shot
numbers Stanford reported a year earlier ([arxiv 2502.10517](https://arxiv.org/abs/2502.10517)).
But KernelBench's baseline is explicitly naive PyTorch, and METR notes this
limits external validity ([METR](https://metr.org/blog/2025-02-14-measuring-automated-kernel-engineering/)).
So "LLM agents can beat PyTorch" is well-supported; "LLM agents can beat
hand-tuned production kernels" needs separate evidence.

That separate evidence does exist. Astra starts from already-deployed SGLang
CUDA kernels and still gets **1.32x average** ([arxiv 2509.07506](https://arxiv.org/abs/2509.07506)) —
modest, but these are production-optimized. The vLLM kernel landscape supplies
several concrete gap regions where LLM search is credibly better-placed than
classical autotune: Blackwell SM120 NVFP4 MoE (SGLang 1.32x faster than vLLM,
145 TFLOPS gap — [HF blog](https://huggingface.co/blog/apsys/blackwell-nvfp4-comparison)),
MI300X fused MoE (default configs sub-optimal — [issue #17619](https://github.com/vllm-project/vllm/issues/17619);
FP8 slower than BF16 — [issue #31475](https://github.com/vllm-project/vllm/issues/31475)),
MXFP4 on GB10 ([devforum](https://forums.developer.nvidia.com/t/vllm-on-gb10-gpt-oss-120b-mxfp4-slower-than-sglang-llama-cpp-what-s-missing/356651)),
and batch-1 decode with <12% L1 / <2% L2 hit rates ([arxiv 2503.08311](https://arxiv.org/html/2503.08311v2)).
In all four regions the vLLM specialist authors have demonstrably under-tuned,
either because the hardware/dtype is new, the config registry is incomplete,
or the shape regime is unusual.

Classical autotune still covers a surprising amount. Triton autotune catches
platform-to-platform order-of-magnitude swings ([IBM Research](https://research.ibm.com/publications/achieving-platform-portability-for-vllm-by-using-triton-autotuning-and-remembering-it)),
and CUTLASS+nvMatmulHeuristics hits 96% of peak with 16 candidates ([NVIDIA](https://developer.nvidia.com/blog/improving-gemm-kernel-auto-tuning-efficiency-on-nvidia-gpus-with-heuristics-and-cutlass-4-2/)).
A fair reading is: for shapes covered by existing autotune configs and within
existing schedule templates, LLM agents typically match rather than beat
classical autotune; where LLM agents genuinely win is when the schedule
template itself is wrong (missing Blackwell warp-spec, wrong fusion depth,
wrong grid shape for small batches) or when the autotune config registry
simply doesn't have an entry. That matches the AutoKernel pattern, which
changes kernel.py — i.e. the schedule — not just the knobs. On the downside,
headline numbers are often measured against weak baselines or have been
disputed (Sakana AI's 10–100x numbers were flagged for reward-hacking during
validation); the more honest per-kernel improvement over already-good
baselines sits around **1.3–2x**.

## Gaps

No public study evaluates LLM-agent kernel search *directly* against vLLM's
custom-ops surface as the baseline (Astra uses SGLang, KernelBench uses
PyTorch, AutoKernel uses PyTorch/torch.compile, Sakana uses PyTorch). The
closest indirect signals are FlashInfer's 2–3x over vLLM PagedAttention and
SGLang's 1.32x over vLLM FP4, but both replacements were human-authored, not
LLM-generated. A direct apples-to-apples study — frozen vLLM benchmark
harness, mutable vLLM custom-op file, agent loop, reporting per-shape deltas
on Marlin / FlashMLA / fused MoE / RMSNorm / RoPE — does not yet exist in the
literature as of 2026-04-22. That is precisely the experiment autoinfer's L3
track needs to run to close the evidence gap on C5.
