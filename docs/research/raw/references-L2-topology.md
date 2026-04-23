# L2 topology evidence (2026-04-22)

Evidence collected for the L2 (hardware pairing and topology search) track of autoinfer. Two claims are backed below with primary sources: arXiv preprints, USENIX / ISCA / MLSys / SC proceedings, vendor engineering blogs, and upstream project design docs. Where sources conflict, the disagreement is flagged inline.

## C4 — PD-disaggregation and heterogeneous serving

**Statement.** Splitting prefill (compute-bound, FLOPs-heavy) from decode (memory-bandwidth-bound) onto separate workers, and — going further — onto separate GPU SKUs, Pareto-dominates homogeneous collocated serving on tokens-per-dollar (or tokens-per-Joule / TCO) at production scale. The dominance is not universal (it needs RDMA-class interconnect, long-enough prefills to amortize KV transfer, and workload-dependent P:D ratios), but independent measurements from Microsoft (Splitwise), UCSD / PKU (DistServe), Moonshot (Mooncake), Princeton / UW (SPAD), HKUST (HexGen-2), ETH (Hetis), plus upstream engineering docs from vLLM, SGLang, NVIDIA Dynamo, and llm-d, all point the same direction at production scale.

- **Splitwise: Efficient Generative LLM Inference Using Phase Splitting** (Microsoft / UW, ISCA 2024, best paper) — https://arxiv.org/abs/2311.18677
  Characterizes prompt vs. token phases on A100 and H100 using production Azure traces and shows phase-splitting across machine pools can deliver 1.4x more throughput at 20% lower cost, or 2.35x throughput under iso-cost/iso-power. Explicitly recommends heterogeneous pools (H100 for prefill, A100 for decode) for lowest TCO/Watt. PDF: https://www.cs.cmu.edu/~18742/papers/Patel2024.pdf.

- **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving** (PKU / UCSD, OSDI 2024) — https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin (arXiv: https://arxiv.org/abs/2401.09670)
  Introduces goodput (tokens/sec meeting per-phase TTFT and TPOT SLOs) as the optimization target and co-optimizes parallelism per phase. Reports up to 7.4x more requests served at SLO or 12.6x tighter SLO vs. colocated state-of-the-art, and bandwidth-aware placement to minimize KV transfer.

- **Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving** (Moonshot AI + Tsinghua, FAST 2025, best paper) — https://www.usenix.org/conference/fast25/presentation/qin (arXiv: https://arxiv.org/abs/2407.00079)
  Describes the production serving platform behind Kimi: disaggregates prefill and decode and adds a global KV cache tier across CPU/DRAM/SSD/NIC. On real traces Mooncake increases effective request capacity 59%–498% vs. baseline collocated serving while honoring SLOs, at the scale of O(100B) tokens/day.

- **Sarathi-Serve: Taming Throughput-Latency Tradeoff in LLM Inference** (GA Tech / MSR India, OSDI 2024) — https://www.usenix.org/conference/osdi24/presentation/agrawal (arXiv: https://arxiv.org/abs/2403.02310)
  Orthogonal-but-related: keeps prefill and decode collocated but uses chunked prefills + stall-free schedules to reduce P/D interference. Reports 2.6x–5.6x serving capacity gains vs. vLLM baseline. Useful as the foil: disaggregation beats this design when interconnect is good, at larger scale, or with heterogeneous SKUs.

- **SPAD: Specialized Prefill and Decode Hardware for Disaggregated LLM Inference** (Princeton / UW, 2025) — https://arxiv.org/abs/2510.08544
  Extends the disaggregation argument to silicon: simulated prefill chips (larger systolic arrays + GDDR) achieve 8% higher prefill perf at 52% lower hardware cost; decode chips hit 97% of decode perf with 28% lower TDP. End-to-end on production traces: 19%–41% cheaper and 2%–17% less TDP than modelled H100-only clusters at iso-performance.

- **HexGen-2: Disaggregated Generative Inference of LLMs in Heterogeneous Environment** (HKUST, ICLR 2025) — https://openreview.net/forum?id=Cs6MrbFuMq
  Directly targets heterogeneous GPU pools under the disaggregated paradigm, formulating device-to-phase assignment as a constrained optimization over cost/latency. Companion paper "Demystifying Cost-Efficiency in LLM Serving over Heterogeneous GPUs" (https://arxiv.org/abs/2502.00722) argues the optimal cluster is heterogeneous for almost all realistic traces.

- **Hetis: Serving LLMs in Heterogeneous GPU Clusters with Fine-grained and Dynamic Parallelism** (SC 2025) — https://arxiv.org/abs/2509.08309
  Builds on HexGen and quantifies where static heterogeneous partitioning underperforms; argues for per-module device assignment. Supports the direction but flags that naive heterogeneity without dynamic scheduling can leave 20–40% of goodput on the table.

- **Dynamo Disaggregation: Separating Prefill and Decode for Enhanced Performance** (NVIDIA, 2025) — https://docs.nvidia.com/dynamo/latest/architecture/architecture.html and https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/
  NVIDIA's own guidance: Dynamo (shipping inside NIM) disaggregates P/D, routes requests KV-aware, and transfers over NIXL/RDMA. Explicitly warns that without RDMA, TTFT blows up by ~40x (from ~355ms to 10+s), which is a load-bearing caveat for heterogeneous fleets.

- **Disaggregated Serving in llm-d: Prefill/Decode Separation with NIXL KV Transfer** (Red Hat / Google / IBM, 2025) — https://llm-d.ai/docs/architecture and https://llm-d.ai/docs/guide/Installation/pd-disaggregation
  Upstream reference implementation over vLLM + SGLang backends. Documents EPP with pd-profile-handler, selective P/D (skip prefill on cache hit), vLLM nixlv2 two-phase protocol vs. SGLang concurrent bootstrap protocol, and pluggable KVConnector for LMCache/Mooncake/KVBM tiers. Primary source for the topology-search scheduling surface autoinfer targets.

- **vLLM Disaggregated Prefilling & Batch Invariance design docs** — https://docs.vllm.ai/en/latest/features/disagg_prefill/ and https://docs.vllm.ai/en/latest/examples/online_serving/disaggregated_serving/
  vLLM upstream describes the Connector / LookupBuffer / Pipe primitives that make disaggregation composable with vLLM's scheduler and PagedAttention. Motivates the design as "tune TTFT without affecting ITL, and vice versa, control tail ITL by removing prefill-preemption of decode."

- **SGLang PD Disaggregation documentation** — https://docs.sglang.ai/advanced_features/pd_disaggregation.html and LMSYS blog "Deploying DeepSeek with PD Disaggregation and Large-Scale Expert Parallelism on 96 H100 GPUs" (2025-05-05) — https://www.lmsys.org/blog/2025-05-05-large-scale-ep/
  Production report: 52.3k input tok/s and 22.3k output tok/s per node on DeepSeek at 2k-token inputs by running P/D disaggregation + expert parallelism on 96 H100s — one of the few public numbers at that scale.

- **Unleashing AMD Instinct MI300X GPUs for LLM Serving: Disaggregating Prefill & Decode with SGLang** (AMD ROCm blog, 2025-08-28) — https://rocm.blogs.amd.com/software-tools-optimization/disaggregation/README.html
  Cross-vendor evidence: disaggregation is a win on AMD silicon too, and MI300X's 192GB HBM3 makes it an unusually attractive decode-tier SKU. Important for autoinfer's heterogeneous (NVIDIA+AMD) deployment axis.

- **Splitting LLM inference across different hardware platforms** (Gimlet Labs, 2025) — https://gimletlabs.ai/blog/multivendor-prefill-decode-disaggregation
  Engineering report on multi-vendor P/D: H100 prefill + MI300X decode; concretely shows heterogeneous-cross-vendor serves within TTFT/TPOT SLOs. Useful as a proof-of-concept for the Basilica-style mixed-fleet case.

- **BurstGPT: A Real-World Workload Dataset to Optimize LLM Serving Systems** (arXiv 2401.17644, SC 2025) — https://arxiv.org/abs/2401.17644 and **Azure LLM Inference Traces** — https://github.com/Azure/AzurePublicDataset
  Primary open traces used by Splitwise, DistServe, and Mooncake for their cost/goodput comparisons. BurstGPT is 5.29M Azure OpenAI traces over 121 days, documenting the bursty, heavy-tailed prompt/response length distributions that make disaggregation pay off.

## C9 — Quality checks under heterogeneous FP behavior

**Statement.** "Pick any available GPU and diff the output against a stored reference" is not a sound quality gate in a heterogeneous pool. Floating-point non-associativity, batch-invariance violations in matmul / RMSNorm / attention kernels, and vendor-/arch-specific FP8 encodings and rounding modes cause legitimately-correct inferences on different hardware to diverge bit-for-bit — and sometimes to diverge in *task-relevant* ways (e.g., 9% accuracy swing on AIME'24 under bf16). A live reference replica running the exact same software/hardware profile as the replica under test is therefore required; static reference values (logits, token strings, hashes) are insufficient.

- **Defeating Nondeterminism in LLM Inference** (Thinking Machines Lab, He Horace et al., 2025-09) — https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
  The foundational write-up. Identifies batch invariance — not reduction-order randomness — as the dominant cause of nondeterministic greedy decoding. Demonstrates 80 unique completions out of 1000 temperature-0 samples of Qwen3-235B, and ships batch-invariant kernels: https://github.com/thinking-machines-lab/batch_invariant_ops.

- **Batch Invariance — vLLM docs** — https://docs.vllm.ai/en/latest/features/batch_invariance/ and RFC issue https://github.com/vllm-project/vllm/issues/27433
  vLLM's production design doc: makes RMSNorm, matmul, and attention batch-invariant through FlexAttention + Triton kernels. Explicitly states that "processing the 1000th query token must have identical reduction order regardless of whether 0 tokens are in the KV cache (prefill) or 999 (decode)" — which is a constraint no non-batch-invariant inference server satisfies by default.

- **Towards Deterministic Inference in SGLang and Reproducible RL Training** (LMSYS / SGLang, 2025-09-22) — https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/
  Ports Thinking Machines' batch-invariant ops into SGLang. Reports 2.8x speedup vs. TML baseline with CUDA graphs, overhead reduced from 61.5% → 34.35%. Confirms that determinism is achievable but *expensive* — which is why autoinfer can't just "turn it on everywhere" and must use a reference replica instead.

- **Give Me FP32 or Give Me Death? Challenges and Solutions for Reproducible Reasoning** (Rice / Adobe, arXiv 2506.09501, 2025-06) — https://arxiv.org/abs/2506.09501
  The sharpest primary source on C9. Shows under bf16 greedy decoding, DeepSeek-R1-Distill-Qwen-7B's AIME'24 accuracy varies by up to 9% and response length by ~9k tokens purely from changing GPU count, GPU type, or eval batch size. FP32 recovers near-perfect reproducibility; bf16 is the worst offender. Proposes LayerCast (bf16 weights, FP32 compute) as a mitigation.

- **Accuracy is Not All You Need** (Dutta et al., Microsoft Research, NeurIPS 2024) — https://arxiv.org/abs/2407.09141 and https://openreview.net/forum?id=QVG7j29Sta
  Introduces the "flips" metric: even when aggregate accuracy is preserved under compression, individual-question answers flip correct↔incorrect at high rates. KL-divergence on the output distribution correlates strongly with flip rate (Spearman 0.981 on MMLU). The load-bearing implication for C9: matching benchmark score does not prove behavioral equivalence across hardware.

- **Deterministic Inference across Tensor Parallel Sizes That Scales** (arXiv 2511.17826, 2025-11) — https://arxiv.org/abs/2511.17826
  Extends batch invariance to tensor parallelism via Tree-Based Invariant Kernels (TBIK) — a hierarchical binary reduction tree guaranteeing bitwise-identical results regardless of TP size. Confirms that *even after* batch-invariant kernels, changing TP degree breaks equivalence unless the reduction tree is also fixed. Another reason reference replicas must pin their topology.

- **Does Quantization Affect Models' Performance on Long-Context Tasks?** (EMNLP 2025) — https://aclanthology.org/2025.emnlp-main.479.pdf
  Evaluates FP8 vs. FP4 on long-context tasks. On average 8-bit preserves accuracy (~0.8% drop) but 4-bit drops up to 59% on long contexts — directly contradicts any assumption that "FP8 ≈ FP16 across the board." Hardware-enforced FP8 encoding differences therefore matter for quality SLOs.

- **ROCm Precision Support / FP8 Numbers** (AMD, 2025) — https://rocm.docs.amd.com/en/latest/reference/precision-support.html and https://rocm.docs.amd.com/projects/HIP/en/docs-6.3.0/reference/fp8_numbers.html
  Primary documentation of the vendor-specific FP8 formats. CDNA3 (MI300-series) natively supports **FNUZ** variants of E4M3 and E5M2 — a different encoding than NVIDIA Hopper's IEEE-ish E4M3/E5M2. ROCm 6.1 also replaced deterministic FP8 rounding with stochastic rounding. Net: the same model weights on H100 vs. MI300X, both labelled "FP8", are *not* numerically equivalent at the kernel level.

- **Using FP8 and FP4 with Transformer Engine — FP8 Primer** (NVIDIA, 2025) — https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html and **Accurate Models of NVIDIA Tensor Cores** (arXiv 2512.07004) — https://arxiv.org/abs/2512.07004
  Primary documentation and measured reverse-engineering of rounding behavior across V100 / A100 / H100 / L40S Ada / B200. The arXiv paper documents concrete differences: Ada tensor cores accumulate FP8 with 13 fractional bits (vs. 25 on H100/H200/B200), and B200 uses round-to-nearest instead of bit truncation in some paths. These are silent, architecture-level divergence sources.

- **Faster Inference of LLMs using FP8 on the Intel Gaudi** (arXiv 2503.09975) — https://arxiv.org/abs/2503.09975
  Third-vendor data point: Gaudi's FP8 group quantization is implementation-defined. The paper reports <1% accuracy degradation but explicitly notes FP8 scaling granularity is vendor-specific, which again breaks cross-vendor bitwise equivalence. Relevant if autoinfer ever onboards Gaudi nodes.

- **Reproducibility — vLLM docs** — https://docs.vllm.ai/en/latest/usage/reproducibility/
  vLLM's own caveat: enumerates the many sources of non-determinism even within a single vLLM deployment (seeding, kernel selection, CUDA graphs, TP size). Implicitly makes the C9 point: since vLLM itself cannot guarantee reproducibility across configurations without batch_invariance mode, a remote "reference hash" strategy is worse still.

- **The Temperature=0 Myth: Why Your LLM Still Isn't Deterministic** (reliableai, 2025) — https://reliableai.substack.com/p/the-temperature0-myth-why-your-llm
  Secondary but practitioner-focused: connects the theory (Thinking Machines, FP32 paper) to observed behavior in production OpenAI / Anthropic endpoints. Useful for motivating the reference-replica design choice to non-systems audiences.

## Key numbers worth anchoring

- **Splitwise:** 1.4x throughput at 20% lower cost, OR 2.35x throughput at iso-cost/iso-power (ISCA 2024, Microsoft + UW Azure traces). Heterogeneous Splitwise-HA (H100 prompt + A100 token) is best on cost/power. https://arxiv.org/abs/2311.18677
- **DistServe:** 7.4x more requests at SLO, or 12.6x tighter SLO vs. colocated SOTA (OSDI 2024). https://arxiv.org/abs/2401.09670
- **Mooncake:** 59%–498% more effective request capacity vs. baseline while honoring SLOs; serves 100B+ tokens/day in Kimi production (FAST 2025). https://arxiv.org/abs/2407.00079
- **Sarathi-Serve (collocated chunked prefill):** 2.6x–5.6x capacity vs. vLLM without disaggregation (OSDI 2024). https://arxiv.org/abs/2403.02310 — sets the floor for how good collocated-but-well-scheduled serving can be.
- **SPAD:** 19%–41% cluster cost reduction and 2%–17% TDP reduction at iso-performance vs. modeled H100-only baseline (arXiv 2510.08544). https://arxiv.org/abs/2510.08544
- **SGLang on 96 H100s with PD + EP (DeepSeek):** 52.3k input tok/s, 22.3k output tok/s per node at 2k-token inputs (LMSYS, 2025-05). https://www.lmsys.org/blog/2025-05-05-large-scale-ep/
- **Dynamo / NIXL:** Without RDMA, TTFT goes from ~355ms → 10+s, i.e. ~40x regression (NVIDIA, 2025). https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/
- **Thinking Machines:** 80 unique greedy-decode outputs out of 1000 temperature-0 samples of Qwen3-235B-A22B-Instruct on the prompt "Tell me about Richard Feynman" (2025-09). https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
- **FP32-or-Death paper:** up to 9% accuracy delta and ~9k-token response-length delta on AIME'24 from changing GPU count / GPU type / eval batch size under bf16 greedy decoding (arXiv 2506.09501). https://arxiv.org/abs/2506.09501
- **Accuracy-is-not-all-you-need:** KL-divergence ↔ flip-rate Spearman ρ = 0.981 on MMLU across quantization schemes (NeurIPS 2024). https://arxiv.org/abs/2407.09141
- **SGLang deterministic mode:** 2.8x speedup with CUDA graphs vs. Thinking Machines reference implementation; overhead 61.5% → 34.35% (LMSYS, 2025-09). https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/
- **FP8 cross-vendor:** CDNA3 (MI300) uses FP8 FNUZ encoding, different from Hopper's; ROCm 6.1 switched FP8 rounding from deterministic → stochastic. https://rocm.docs.amd.com/projects/HIP/en/docs-6.3.0/reference/fp8_numbers.html
- **Ada tensor-core FP8 accumulation:** 13 fractional bits on Ada vs. 25 on H100/H200/B200, and B200 replaces truncation with round-to-nearest in some paths (arXiv 2512.07004). https://arxiv.org/abs/2512.07004

### Disagreements / caveats

- **Splitwise vs. DistServe headline numbers differ by ~5x** (1.4x vs. 7.4x). They measure different metrics (iso-cost throughput vs. iso-SLO request rate), use different traces (Azure production vs. ShareGPT + synthetic), and DistServe's larger gains come from co-optimizing parallelism per phase — something Splitwise intentionally does not do. Not a contradiction, but they cannot be compared directly; the direction of improvement is consistent, the magnitude is workload- and metric-dependent.
- **HexGen (v1, ICML 2024) collocates P and D**, and argues that on sufficiently heterogeneous fleets the communication cost of disaggregation can outweigh the specialization gains. HexGen-2 (ICLR 2025) reverses this and embraces disaggregation. Interpret HexGen v1 as setting the boundary condition: if inter-node interconnect is weak (e.g., Ethernet-only), collocated may still win.
- **Sarathi-Serve implicitly contests disaggregation** at small-to-medium scale: chunked prefill with a single GPU type can recover most of the P/D-interference loss without paying the KV transfer cost. In single-tenant, single-SKU settings below the KV-transfer-amortization threshold, Sarathi-style scheduling may Pareto-dominate disaggregation on tokens/\$. Autoinfer's search should therefore consider both points.
- **Batch invariance cost:** Thinking Machines' original implementation was ~61% slower; SGLang got it to ~34%; vLLM's tracker issue (#27433) suggests further gains are in flight. Any design that *requires* batch invariance globally is paying a material throughput tax — reinforcing the reference-replica choice for C9 rather than fleet-wide determinism.

## Gaps

The cleanest open gap is quantitative: none of the primary sources above measures tokens/\$ for a truly heterogeneous *multi-vendor* (NVIDIA + AMD + consumer-4090/5090) disaggregated fleet against a homogeneous H100 baseline at production scale over a publicly-released trace. Splitwise covers H100↔A100 only; Mooncake is H800/A800-internal to Moonshot; HexGen-2 uses simulated cost curves; Gimlet Labs' multi-vendor report is a blog with limited load; the AMD ROCm SGLang post is single-vendor on MI300X. For C4 this means the "Pareto-dominates" claim is directionally well-supported but lacks a single end-to-end published number for the autoinfer / Basilica target topology — which is itself a research contribution autoinfer could make. For C9, the Ada-vs-Hopper-vs-Blackwell FP8 rounding differences are now documented at the tensor-core level (arXiv 2512.07004), but there is no public study that converts those ULP-level differences into *task-eval deltas* on standard benchmarks; that gap is why a live reference replica, rather than static logit hashes, is the defensible engineering choice today.
