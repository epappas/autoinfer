# Alexandria coverage map (seed, 2026-04-22)

Snapshot of what the `global` alexandria workspace already contains for
inference-engine topics. Purpose: avoid duplicate ingests and identify gaps.

## Strong coverage (vLLM-heavy)

- **vLLM core**: engine_args, V1 architecture, scheduler, fusion passes
  (`vllm/docs/fusions.md`), prefix caching (`vllm/docs/prefix_caching.md`),
  batch invariance (`vllm/docs/batch_invariance.md`), sleep mode,
  `supported_models.md`.
- **vLLM attention kernels**: FlashInfer TRT-LLM, Triton unified attention,
  FlashMLA sparse, KV-cache kernels — all under `wiki/vllm/tests/*`.
- **Speculative decoding**: MTP, PARD parallel draft models, MLP draft, EAGLE,
  Speculators repo — `wiki/vllm/docs/mlp.md`, `parallel_draft_model.md`,
  `speculators.md`.
- **Distributed serving (llm-d)**: PD-disaggregation on CPU/AMD/Intel Gaudi
  HPU/Intel XPU/TPU v6e/TPU v7/GKE-RDMA, inference-scheduling values files,
  Inference Resilience Operator (IRO), NixlConnector compatibility, wide-EP.
- **RLHF + inference-time scaling**: `rlhf.md`, "Inference-Time Scaling for
  Generalist Reward Modeling" (2504.02495).
- **Multimodal**: `multimodal_inputs.md`.

## Gap areas (need ingest)

- **SGLang**: only referenced indirectly (values_sglang.md, readme.sglang.md
  in llm-d). No SGLang paper, no RadixAttention deep dive, no SGLang
  frontend/runtime ingest. → ingest arxiv 2312.07104, SGLang repo docs.
- **AutoKernel / KernelAgent / KernelFalcon**: nothing. → ingest
  RightNow-AI/autokernel, meta-pytorch/KernelAgent, PyTorch blog on
  KernelFalcon, arxiv 2601.15727 ("Towards Automated Kernel Generation"),
  2603.21331 (AutoKernel paper).
- **TensorRT-LLM, LMDeploy, ExLlama, MLC LLM, llama.cpp**: partial TRT-LLM
  refs only through vLLM. → to ingest per hypothesis needs.
- **Karpathy-style minimal inference**: nothing. → ingest nano-vllm,
  nanochat inference path, llm.c `dev/cuda`.
- **Covenant AI / Templar distributed training**: nothing in alexandria.
  → ingest Covenant-72B paper (arxiv 2603.08163v2), Templar Research
  blog "Checkpoint One", SparseLoCo/CCLoco paper and repo.
- **Pulse**: user-referenced but unidentified — see `02-covenant-pulse.md`.

## Implication for autoinfer

vLLM is the most heavily documented reference engine we have. Good baseline
substrate for the first hypothesis: **vLLM-as-substrate + autoinfer-as-search-loop**.
SGLang is the biggest blind spot if we want an apples-to-apples comparison.
