# L1 engine-config evidence (2026-04-22)

Scope: primary-source evidence for the L1 track of autoinfer (engine-config search over vLLM). Four claims (C1, C2, C3, C8). Each claim has 4-8 sources with a 1-2 sentence summary and a URL. Conflicts with the autoinfer thesis are noted inline.

## C1 - vLLM tunable surface

The vLLM EngineArgs + runtime-env surface is broad and interacts non-trivially (scheduler knobs, KV cache sizing, parallelism, chunked prefill, disaggregation, kernel backends). Published tuning guides from vLLM, Red Hat, NVIDIA-adjacent, AMD, and Google all report double-digit throughput/latency swings from changing a handful of flags on a fixed model/GPU pair, and the vLLM repo itself ships an `auto_tune` benchmark that treats `max_num_seqs` x `max_num_batched_tokens` as a search problem with latency constraints. This is exactly the signature of a config-search problem, not a "pick the defaults" problem.

- **Optimization and Tuning - vLLM stable docs** (vLLM project, 2025-2026) - https://docs.vllm.ai/en/stable/configuration/optimization/
  Official guide lists `max_num_batched_tokens`, `max_num_seqs`, chunked prefill, CUDA graphs, prefix caching, parallelism, and speculative decoding as throughput/latency levers. Explicitly recommends `>8192` for throughput and `~2048` for ITL, confirming a real tradeoff surface rather than a single optimum.

- **Practical strategies for vLLM performance tuning** (Red Hat Developer, 2026-03-03) - https://developers.redhat.com/articles/2026/03/03/practical-strategies-vllm-performance-tuning
  Red Hat engineers walk through the default 0.9 `gpu-memory-utilization`, when to raise/lower it, and interactions with tensor parallelism and KV-cache sizing on production AI Inference Server deployments. Reports measurable throughput shifts from single-flag changes at realistic scales.

- **Serving LLMs on AMD MI300X: Best Practices** (vLLM blog / AMD, 2024-10-23) - https://blog.vllm.ai/2024/10/23/vllm-serving-amd.html
  Reports 1.5-1.8x throughput and 1.7-5.1x TTFT wins over TGI on Llama 3.1 70B/405B once tuned; flags `--num-scheduler-steps` 10-15, disabling chunked prefill on MI300X, and `NCCL_MIN_NCHANNELS=112` as non-obvious wins. Demonstrates platform-specific optima that defaults miss.

- **vLLM V1 performance optimization** (AMD ROCm docs, 2025) - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
  Documents throughput going from 2373 to 3774 tok/s as `max_num_batched_tokens` varies 256 -> 4096 on MI300X, plus AITER kernel toggles (`VLLM_ROCM_USE_AITER=1`). Explicit evidence that a single knob moves throughput ~1.6x at fixed hardware.

- **vLLM Performance Tuning: The Ultimate Guide to xPU Inference Configuration** (Google Cloud, 2025) - https://cloud.google.com/blog/topics/developers-practitioners/vllm-performance-tuning-the-ultimate-guide-to-xpu-inference-configuration
  GCP engineering blog walking through `gpu_memory_utilization`, max-tokens-per-iteration accounting, and TP/PP combinations across Google xPUs. Frames vLLM tuning explicitly as multi-knob exploration, not default-then-ship.

- **vllm auto_tune benchmark** (vLLM repo, main) - https://github.com/vllm-project/vllm/blob/main/benchmarks/auto_tune/README.md
  In-tree automation that sweeps `max-num-seqs` and `max-num-batched-tokens`, subject to E2E P99 latency and prefix-cache-hit constraints, to find max sustainable throughput. The project itself ships config search because hand tuning does not scale.

- **vLLM v0.6.0: 2.7x Throughput Improvement and 5x Latency Reduction** (vLLM blog, 2024-09-05) - https://blog.vllm.ai/2024/09/05/perf-update.html
  Internal engineering post documenting how removing CPU-side overheads and re-tuning the scheduler (not changing kernels) produced 2.7x throughput / 5x latency gains. Evidence that the config+runtime surface contains order-of-magnitude moves the user does not see by default.

- **Red Hat AI Inference Server 3.0: Complete list of vLLM server arguments** (Red Hat, 2025-2026) - https://docs.redhat.com/en/documentation/red_hat_ai_inference_server/3.0/html/vllm_server_arguments/all-server-arguments-server-arguments
  Enumerates the full exposed server-argument surface of a production vLLM distribution; the page has well over a hundred flags across scheduling, quantization, parallelism, KV, speculative decoding, and multimodal subsystems. Surface-area proof point.

- **Inside vLLM: Anatomy of a High-Throughput LLM Inference System** (Aleksa Gordic, 2025-08-29, base commit `42172ad`) - https://www.aleksagordic.com/blog/vllm
  Long-form walk-through of vLLM V1 from single-GPU offline inference to multi-node disaggregated serving; names every L1/L2 knob the autoinfer search targets (`block_size`, `long_prefill_token_threshold`, `gpu_memory_utilization`, prefix-cache toggles, spec-dec backend `ngram`/`eagle`/`medusa`, chunked prefill, TP/PP/DP, P/D-disagg connectors) in the live API, and frames `vllm bench {latency,throughput,serve}` + the in-tree auto-tune script as the natural public baseline for L1. Also gives the roofline/saturation-batch framing (piecewise-linear step time around `B_sat`) that the L1 surrogate should encode, and confirms V1 mixes prefill+decode in one batch, invalidating disjoint-regime models. See `docs/research/raw/07-vllm-v1-architecture.md`.

## C2 - Speculative decoding tradeoffs

Speculative decoding (EAGLE / Medusa / MTP / PARD / draft-model variants) reliably speeds up decode, but the wins are acceptance-rate x batch-load dependent. Papers report *average* speedup and *average* acceptance length; serving-systems papers (SmartSpec/TurboSpec, Nebius MoE note) show that under high load or low acceptance, speculation can *increase* latency, meaning p99 and quality-variance effects are not captured by throughput-only benchmarks. EAGLE-3 / PARD claim lossless sampling but still task-dependent acceptance, so tail-behaviour is a first-class SLO concern.

- **EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty** (Li et al., ICML 2024, arXiv:2401.15077) - https://arxiv.org/abs/2401.15077
  Original EAGLE: drafts at the second-to-top feature level and claims ~3x decoding speedup over vanilla autoregression with strict acceptance (lossless in distribution). Acceptance rate explicitly acknowledged as position- and context-dependent.

- **EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees** (Li et al., EMNLP 2024, arXiv:2406.16858) - https://arxiv.org/abs/2406.16858
  Directly states "acceptance rate ... is not only position-dependent but also highly context-dependent, with significant variance in acceptance rates at the same position depending on context". Uses the draft model's calibration to reshape the tree dynamically, acknowledging variance as the central problem.

- **EAGLE-3: Scaling up Inference Acceleration via Training-Time Test** (Li et al., NeurIPS 2025, arXiv:2503.01840) - https://arxiv.org/abs/2503.01840
  Claims 3.0-6.5x speedup and 20-40% improvement over EAGLE-2, but reports speedup as task-dependent average across MT-Bench-style tasks. Lossless by strict sampling, so all variance moves into latency, not quality - which is exactly the p99 story.

- **Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads** (Cai et al., ICML 2024, arXiv:2401.10774) - https://arxiv.org/abs/2401.10774
  Introduces tree-attention + multiple heads; reports 2.2x (Medusa-1) and 2.3-3.6x (Medusa-2) speedup. Medusa-2 fine-tunes the backbone, so "lossless" is weaker than EAGLE and opens a real quality-variance question that task-level evals downplay.

- **DeepSeek-V3 Technical Report (Multi-Token Prediction)** (DeepSeek-AI, 2024, arXiv:2412.19437) - https://arxiv.org/abs/2412.19437
  Trains with an MTP auxiliary loss to densify training signal; explicitly notes the MTP heads can be repurposed for speculative decoding at inference. Shifts draft-model design from "external" to "trained-in", changing the acceptance-rate distribution shape.

- **PARD: Accelerating LLM Inference with Low-Cost PARallel Draft Model Adaptation** (Liu et al., 2025, arXiv:2504.18583) - https://arxiv.org/abs/2504.18583
  Target-independent parallel draft reports up to 3.67x on LLaMA3.1-8B (264.88 tok/s, 1.15x over EAGLE-3) on vLLM. Speedup stated as max, not p99; same tail-behaviour caveat applies.

- **Spec-Bench: A Comprehensive Benchmark for Speculative Decoding** (Xia et al., ACL Findings 2024) - https://sites.google.com/view/spec-bench / https://github.com/hemingkx/Spec-Bench
  Six-subtask benchmark (MT-bench, summarization, RAG, translation, QA, math) exposing that speedup rankings flip across task types, confirming that single-number benchmarks under-report variance.

- **TurboSpec / "Optimizing Speculative Decoding for Serving LLMs Using Goodput"** (Liu et al., 2024, arXiv:2406.14066) - https://arxiv.org/abs/2406.14066
  Primary source for the anti-speculation finding: "under higher request rates or low speculation accuracy, it paradoxically increases latency". Introduces "goodput" metric and a closed-loop controller that dynamically varies draft length - direct evidence throughput-only benchmarks miss the failure mode.

- **Why large MoE models break latency budgets and what speculative decoding changes in production systems** (Nebius Engineering blog, 2025) - https://nebius.com/blog/posts/moe-spec-decoding
  Production write-up stating speculative decoding "reshapes the latency distribution" with disproportionate effect on P90/P99 in long-context, non-streaming serving. Complements the SmartSpec paper with deployment-level evidence.

- **Faster LLM Inference via Sequential Monte Carlo (SMC-SD)** (Abdelfattah lab, 2026, arXiv:2604.15672) - https://arxiv.org/abs/2604.15672 / repo https://github.com/abdelfattah-lab/smcsd / blog https://makora.com/blog/smc-sd
  Replaces rejection-sampled verification with importance-sampled resampling over N draft particles; no KV rollback path, bounded approximation error instead of bit-exact sampling. Reports 5.2x vs AR / 2.36x vs SOTA spec-dec within 3% accuracy loss on Llama 70B / 4xH100 (GSM8K, MATH, AlpacaEval, DS-1000). Exposes five inference-time knobs (`n_particles`, `gamma`, resample threshold/method, draft/target temperatures) on top of SGLang + Triton + FlashAttention-3 - first lossy-but-bounded spec-dec variant, forcing the quality gate to be a live reference replica rather than strict-accept-by-construction. See `docs/research/raw/05-smc-speculative-decoding.md`.

*Conflict note:* The EAGLE papers claim "lossless" sampling, and vLLM's strict-accept speculative decoding is by construction distribution-preserving. If one believes strict-accept is correctly implemented in every kernel, quality variance reduces to *latency* variance only. The falsifiable piece of the thesis is: in the presence of FP8/fp16 mixed kernels, batch-variant matmul, and tree verification, strict-accept may not hold bit-exact - so the claim "speculative decoding has zero quality risk" is empirical, not axiomatic. This is where C3 connects.

## C3 - Quantization quality and batch invariance

Weight quantization (AWQ, GPTQ, SmoothQuant) and low-precision serving (FP8 E4M3/E5M2, FP8/INT4 KV cache) are well-studied, and task-level benchmarks usually report "minimal degradation". But Thinking Machines Lab and the LMSYS/SGLang follow-up show inference is non-deterministic even at temperature 0 because common kernels (matmul, RMSNorm, attention) are not batch-invariant: the same request gets different logits depending on co-scheduled batch. Quantization widens the numerical noise floor, which interacts with batch-variance and can silently drift logits inside an SLO. Gating quantization rollouts on logit divergence + batch-invariance under production load (not just MMLU/GSM8K deltas) is the defensible bar.

- **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration** (Lin et al., MLSys 2024 Best Paper, arXiv:2306.00978) - https://arxiv.org/abs/2306.00978
  Protects ~1% salient weights via activation-observed per-channel scaling; 4-bit weight-only with small perplexity cost. Establishes the "small average degradation" baseline that task-evals use and that this claim says is insufficient.

- **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers** (Frantar et al., ICLR 2023, arXiv:2210.17323) - https://arxiv.org/abs/2210.17323
  One-shot 3-4 bit quantization of 175B-scale models using approximate second-order info. Same caveat: reports minimal perplexity delta at dataset level, silent on per-logit / per-batch drift.

- **SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs** (Xiao et al., ICML 2023, arXiv:2211.10438) - https://arxiv.org/abs/2211.10438
  W8A8 via activation-outlier migration; preserves accuracy across OPT/BLOOM/Llama families. Establishes that aggressive activation quantization is feasible, but again evaluated at task level, not logit-divergence level.

- **Defeating Nondeterminism in LLM Inference** (Thinking Machines Lab blog, He, 2025-09) - https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
  Primary source for the core claim: at temp=0, outputs still differ run-to-run because matmul/RMSNorm/attention kernels lack batch invariance, so request logits depend on co-batched load. Motivates the batch-invariance gate that this project's quantization policy should adopt.

- **Towards Deterministic Inference in SGLang and Reproducible RL Training** (LMSYS blog, 2025-09-22) - https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/
  Independent reproduction and extension: batch-invariant RMSNorm/matmul/attention kernels integrated into SGLang with chunked prefill + CUDA graphs + radix cache, achieving 100% reproducibility at 2.8x less overhead than the reference Thinking Machines Lab kernels. Proves batch-invariance is compatible with production throughput paths.

- **Quantized KV Cache** (vLLM docs, current) - https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/
  Documents FP8 E4M3 / E5M2 KV cache paths, three calibration strategies (default=1.0 scales, on-the-fly random-token, dataset-calibrated via llm-compressor), and recommends lm-eval-harness on GSM-8K. Confirms the production surface and the evaluation practice this claim says is necessary-but-insufficient.

- **vLLM reproducibility documentation** (vLLM docs) - https://docs.vllm.ai/en/latest/usage/reproducibility/ (also present in alexandria raw: `raw/git/vllm-e8ba7172314e/docs/usage/reproducibility.md`)
  vLLM explicitly disclaims default reproducibility "for the sake of performance" and points to `VLLM_ENABLE_V1_MULTIPROCESSING=0`, batch-invariant kernels, and seeded sampling as opt-ins. Engine-level admission that determinism is off by default and must be engineered on.

- **An Inquiry into Datacenter TCO for LLM Inference with FP8** (arXiv:2502.01070, 2025) - https://arxiv.org/abs/2502.01070
  Measures E4M3 vs E5M2 on Intel Gaudi-2 with instruction-tuned LLaMA on MMLU. E4M3 consistently outperforms E5M2 at the task level; the paper itself notes scaling-strategy dependence, which is a logit-drift proxy.

- **NestedFP: High-Performance, Memory-Efficient Dual-Precision** (arXiv:2506.02024, 2025) - https://arxiv.org/abs/2506.02024
  Reports "consistent accuracy degradation" in FP8 and explicit throughput-vs-quality tradeoff, counter to the common "FP8 is effectively free" narrative. Supports the claim that FP8 needs divergence gating, not just task evals.

- **vLLM brings FP8 inference to the open source community** (Red Hat Developer, 2024-07-15) - https://developers.redhat.com/articles/2024/07/15/vllm-brings-fp8-inference-open-source-community
  Production-facing announcement of FP8 in vLLM with calibration guidance; positions FP8 as the default forward path on H100+. Useful as the "industry consensus" counter-source this claim pushes back against.

*Conflict note:* Most quantization papers frame their output as "minimal degradation" or "lossless" at task level, which on its face conflicts with the claim that quantization "silently drifts logits". Both are true at different measurement resolutions: dataset-mean accuracy is nearly preserved, per-request logit vectors can drift measurably, and the drift composes with batch-variance. The thesis is falsifiable: if a batch-invariant + logit-KL gate never fires during a quantization rollout that also passes task evals, then task evals alone were sufficient.

## C8 - Minimal-rig transfer limits

Minimal reimplementations (nano-vLLM and kin) are excellent for pedagogy and for iterating on a single kernel or scheduler decision in isolation, but they do not reproduce the features that dominate production engine-config search: chunked prefill, speculative decoding with tree verification, quantization fast paths (FP8 weight + KV), multimodal runners, and prefill/decode disaggregation. Config-search results on nano-vLLM will not transfer to vLLM V1, because the search dimensions are different. This matches the autoinfer framing (nano-vLLM is a sandbox, not a substrate).

- **GeeeekExplorer/nano-vllm** (GitHub repo) - https://github.com/GeeeekExplorer/nano-vllm
  ~1200 LoC Python. Implements prefix caching, TP, torch.compile, CUDA graphs. Reported to beat vLLM by ~5% on a single synthetic Qwen3-0.6B / RTX 4070 benchmark; absent: chunked prefill, speculative decoding, FP8 KV, multimodal, PD-disagg, LoRA, structured output.

- **vLLM V1 Alpha: A major upgrade to vLLM's core architecture** (Red Hat Developer / vLLM, 2025-01-28) - https://developers.redhat.com/articles/2025/01/28/vllm-v1-a-major-upgrade-vllms-core-architecture
  V1 unifies prefill and decode in one scheduler loop, integrates chunked prefill as default, and is the baseline the autoinfer search has to target. None of the V1-unique scheduler behaviour is present in nano-vLLM.

- **Inside vLLM: Anatomy of a High-Throughput LLM Inference System** (vLLM blog, 2025-09-05) - https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html
  Engine-internals walkthrough: block manager, paged attention, scheduler policy, CUDA graph capture, speculative decoding integration, KV connector layer. Makes explicit the feature surface a minimal engine omits.

- **Disaggregated Prefilling (experimental)** (vLLM docs) - https://docs.vllm.ai/en/latest/features/disagg_prefill/
  Documents N-prefill + M-decode deployment, KV-connector-mediated transfer, and per-phase autoscaling. This is an engine-config axis (split, ratio, connector) entirely absent from nano-vLLM.

- **Bringing State-Of-The-Art PD Speed to vLLM v1 with LMCache** (LMCache blog, 2025-04-29) - https://blog.lmcache.ai/2025-04-29-pdbench/
  Concrete PD-disagg benchmarks on vLLM V1 + LMCache; shows TTFT/ITL sensitivity to connector and cache-placement choices. Another axis not searchable on nano-vLLM.

- **vLLM vs Nano vLLM: Choosing the Right LLM Inference Engine** (F22 Labs blog, 2025) - https://www.f22labs.com/blogs/vllm-vs-nano-vllm-choosing-the-right-llm-inference-engine/
  Third-party comparison stating explicitly nano-vLLM is for learning/prototyping/small-scale, vLLM for production scale. Corroborates the transfer-limit claim.

- **Nano-vLLM meets Inference Endpoints** (HuggingFace blog, angt) - https://huggingface.co/blog/angt/nano-vllm-meets-inference-endpoints
  Documents the footprint and use case as "a minimal engine you can drop into endpoints for small models". Consistent with nano-vLLM as pedagogical/light, not a search substrate.

- **Mini-SGLang: Efficient Inference Engine in a Nutshell** (LMSYS blog, 2025-12-17) - https://www.lmsys.org/blog/2025-12-17-minisgl/
  LMSYS's own minimal engine - explicitly framed as didactic stripping of SGLang. Same message from a second ecosystem: minimal engines teach, production engines serve, and the feature delta (chunked prefill, structured decoding, radix cache, deterministic kernels) is the search surface.

*Conflict note:* nano-vLLM's reported 5% throughput edge on a narrow Qwen3-0.6B benchmark is real (prior autoinfer note `04-karpathy-style-minimal-inference.md` cites 1434 vs 1362 tok/s). That number is compatible with C8 because it is measured on a workload that exercises none of vLLM V1's distinguishing features - small model, no chunked prefill, no speculation, no quantization, no multimodal, no disagg. Extrapolating from it to "nano-vLLM is a valid search substrate" is where the claim cuts.

## Scope note — routing policy is not L1

Request-level load-balancing and PD-disagg dispatch (policies
`cache_aware` / `power_of_two` / `consistent_hash` / `round_robin` /
`random`, plus `--vllm-pd-disaggregation` with prefill/decode pools) live
*above* the engine — in `vllm-project/router` (see
`08-vllm-router-dataplane.md`) or llm-d, not in `EngineArgs`. They
belong to L2 topology, not L1. This matters for the L1 track because
any multi-replica "L1 beats defaults" claim is under-specified without
stating the router policy: switching policy with the same engine args
moves the effective workload seen by each worker and can swamp
single-flag L1 wins. L1 experiments at iteration zero stay single-
replica (§8 of `00-hypothesis-seed.md`), which side-steps this — but the
moment L2 brings a second replica online, router policy must be part of
the trial record or the L1 numbers become non-comparable.

## Gaps

What is still not well-documented in open primary sources, and would be worth ingesting into alexandria before moving L1 into implementation: (1) quantitative studies of **batch-variance x quantization composition** at logit resolution - Thinking Machines and SGLang both fix batch-variance for deterministic kernels but report perplexity/accuracy, not per-logit KL under production batch mixes, so the exact bound on "silent drift" is still empirical; (2) acceptance-rate distributions (not means) for EAGLE-3 / PARD / MTP under continuous batching at varying QPS, which is where the TurboSpec paper points but does not exhaustively sweep; (3) third-party, reproducible vLLM V1 config-search studies (not blog posts) that publish per-flag sensitivity across models and GPUs - the vLLM repo has `benchmarks/auto_tune` but its results are not a public dataset; (4) disagg-prefill tuning guides with published numbers for connector choice, prefill/decode ratios, and KV-cache placement policies - LMCache has one benchmark but the config space is barely mapped. Candidates to ingest: the `vllm/benchmarks/auto_tune/README.md` itself, the TurboSpec paper, Spec-Bench harness output format, the Thinking Machines + LMSYS deterministic-inference blogs, and the NestedFP paper. Alexandria already has the vLLM repo (including `docs/usage/reproducibility.md`) and the llm-d planner proposal; adding the four blogs and two papers above would cover the remaining cross-references this track needs.
