# Search-policy evidence (2026-04-22)

Scope: back two claims relevant to autoinfer's search-policy track.
Primary question for engineering context: at a ~200-evaluation budget over a
vLLM config space, is LLM-guided search a defensible choice over random or
Bayesian optimization (BO)?

## C6 — LLM-guided search sample efficiency

The literature shows a consistent but narrow pattern. LLM proposers beat random
and matched or exceeded BO in the early regime (tens of evaluations) on small,
well-known, low-dimensional HPO benchmarks. At higher budgets the picture
inverts: classical methods (TPE, CMA-ES, BOHB) catch up or win, and the most
competitive results come from hybrids that use the LLM for warm-start /
candidate proposal and a GP or TPE for credit assignment. The one study that
targeted an ML-systems-like search space (tuning a small LM under a fixed
compute budget) reported that CMA-ES and TPE beat pure LLM agents, with a
CMA-ES+LLM hybrid (Centaur) winning overall.

- **Using Large Language Models for Hyperparameter Optimization** (Zhang,
  Desai, Bae, Lorraine, Ba — U. of Toronto, NeurIPS 2023 workshop, arXiv
  2312.04528) — https://arxiv.org/abs/2312.04528
  On HPOBench (8 datasets x 4 model families) plus ViT/ResNet on CIFAR-10,
  LLM-proposed hyperparameters match or beat BO at "constrained search
  budgets" (their reported budget is ~10–30 trials). The edge shrinks as the
  budget grows. Also introduces the code-as-hyperparameter variant relevant
  to autoinfer's "LLM proposes code" loop.

- **Large Language Models as Optimizers (OPRO)** (Yang, Wang, Lu, Liu, Le,
  Zhou, Chen — Google DeepMind, ICLR 2024, arXiv 2309.03409) —
  https://arxiv.org/abs/2309.03409
  Demonstrates LLM-as-optimizer on linear regression, TSP, and prompt
  optimization. Prompt-optimization results (+8% GSM8K, up to +50% on BBH
  over human prompts) are the strongest, but note the objective there is
  cheap to evaluate. OPRO is the canonical method reference for "LLM
  proposes, evaluator scores, history goes back into the prompt" — the same
  loop autoinfer is reusing from autoresearch-rl.

- **LLAMBO: Large Language Models to Enhance Bayesian Optimization** (Liu,
  van Breugel, Qian, van der Schaar — U. of Cambridge, ICLR 2024, arXiv
  2402.03921) — https://arxiv.org/abs/2402.03921
  Direct head-to-head with BO on hyperparameter-tuning benchmarks. LLAMBO's
  gain is largest in the zero-shot warm-start and early-iteration regime
  (sparse observations). The paper explicitly frames the LLM's role as
  augmenting the surrogate and acquisition, not replacing the BO machinery.
  Code: https://github.com/tennisonliu/LLAMBO

- **SLLMBO: Sequential Large Language Model-Based Hyper-parameter
  Optimization** (Mahammadli, Ertekin — METU, arXiv 2410.20302, Jan 2025) —
  https://arxiv.org/abs/2410.20302
  Benchmarks GPT-3.5-Turbo, GPT-4o, Claude-Sonnet-3.5, and Gemini-1.5-Flash
  against TPE/BO over multiple HPO tasks. Proposes an LLM–TPE hybrid sampler
  (LLM-TPE). Headline: pure LLM samplers under-exploit once the budget grows;
  LLM-TPE recovers and in several tasks matches or beats Optuna-TPE at equal
  budget. Important for autoinfer because it names the failure mode (LLM
  doesn't re-exploit near good regions) and the fix.

- **Can LLMs Beat Classical Hyperparameter Optimization Algorithms? A Study
  on autoresearch** (ELLIS Tübingen / U. of Freiburg / KIT, arXiv
  2603.24647, v5 Apr 2026) — https://arxiv.org/abs/2603.24647
  Most directly load-bearing source for autoinfer. Uses the autoresearch
  repo as a testbed for tuning a small LM under fixed compute. With a fixed
  search space, CMA-ES and TPE consistently outperform LLM-only agents
  (including frontier models). Allowing the LLM to edit source code narrows
  but does not close the gap. Their Centaur hybrid (CMA-ES + LLM) wins.
  The stated reason LLMs lose is weak state tracking across trials, which is
  exactly the regime (100s of evaluations, OOM-heavy space) autoinfer will
  be in.

- **EvoPrompt: Connecting LLMs with Evolutionary Algorithms Yields Powerful
  Prompt Optimizers** (Guo et al., Microsoft Research, ICLR 2024, arXiv
  2309.08532) — https://arxiv.org/abs/2309.08532
  An LLM-as-mutation-operator inside GA/DE. Up to +25% over human-engineered
  prompts on BBH. Relevant as a reference for the "LLM proposes mutations,
  search operator keeps the survivors" pattern, which is how autoinfer's
  code-proposal variant effectively behaves.

- **Evolutionary Optimization of Model Merging Recipes** (Akiba, Shing, Tang,
  Sun, Ha — Sakana AI, Nature Machine Intelligence 2025, arXiv 2403.13187) —
  https://arxiv.org/abs/2403.13187
  Not HPO per se, but a concrete case where an evolutionary outer loop using
  CMA-ES over parameter-space and data-flow-space recipes beat
  hand-engineered merges on Japanese-math benchmarks. Included because the
  follow-up from Sakana (AB-MCTS) and community frameworks like OpenEvolve
  generalize the same outer loop. Relevant to "LLM proposes, evaluator
  scores" scaled up. https://sakana.ai/evolutionary-model-merge/

- **AlphaEvolve: A Coding Agent for Scientific and Algorithmic Discovery**
  (Novikov et al., Google DeepMind, arXiv 2506.13131, May 2025) —
  https://arxiv.org/abs/2506.13131 ; blog:
  https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/
  Strongest production-grade evidence for the LLM-propose / evaluator-score
  loop on real systems workloads. Among the reported wins: 23% speedup on a
  matmul kernel used in Gemini training, ~1% end-to-end Gemini training time
  saved, datacenter scheduling improvements. This is the closest analogue to
  what autoinfer wants to do on vLLM kernels / configs. Note: compute budget
  per target is huge (evolutionary loops over many days), not 200 trials.

- **Google Vizier: A Service for Black-Box Optimization** (Golovin, Solnik,
  Moitra, Kochanski, Karro, Sculley — Google, KDD 2017) —
  https://dl.acm.org/doi/10.1145/3097983.3098043 (PDF mirror:
  https://www.cl.cam.ac.uk/~ey204/teaching/ACS/R244_2020_2021/papers/golovin_KDD_2017.pdf)
  The BO baseline reference. Default algorithm is batched GP bandits; used
  in production across Google. This is what "BO baseline" concretely means
  in the C6 comparison.

Direct answer to the engineering question (200 evals over a vLLM config
space, e.g. `max-num-seqs` x `max-num-batched-tokens` x
`gpu-memory-utilization` x `tensor-parallel` x quantization mode, with
multi-modal OOM failure regions): the papers above support a **hybrid**
(LLM warm-start + LLM proposal on top of a TPE or CMA-ES surrogate) rather
than pure LLM-guided search. Pure LLM will probably tie random by trial ~30
and fall behind a competent BO / CMA-ES by trial ~150, with the additional
risk called out in 2603.24647 that LLMs drift into OOM regions because they
don't track failure boundaries across trials. Pure random is only defensible
for the first ~20 trials as a Sobol-style seed.

## C7 — Joint / layered search benefits

The claim "joint / layered search beats single-layer search when cross-layer
coupling is strong" has solid support in three places: (1) tensor-program
compilation (TVM/Ansor's joint schedule search with a task scheduler beat
single-subgraph tuning), (2) joint NAS+HPO (training hyperparameters and
architecture interact and cannot be optimized independently without loss),
(3) disaggregated LLM serving (DistServe shows the SLO-feasible frontier
requires co-optimizing resource allocation *and* parallelism per phase). For
multi-fidelity specifically, BOHB and its successors dominate single-fidelity
BO under anything resembling a realistic evaluation budget. The weakest link
is "joint" work that cuts across kernel + engine + topology at once — that
is largely aspirational in the literature.

- **Ansor: Generating High-Performance Tensor Programs for Deep Learning**
  (Zheng et al., UC Berkeley / Amazon / Tsinghua, OSDI 2020, arXiv
  2006.06762) — https://arxiv.org/abs/2006.06762 ; paper PDF
  https://www.usenix.org/system/files/osdi20-zheng.pdf
  Three-component search (hierarchical sampler + evolutionary performance
  tuner + cross-subgraph task scheduler). The task-scheduler component is
  the "joint" bit: it allocates tuning budget across subgraphs using a
  gradient-descent-style reallocation, and Ansor reports up to 3.8x / 2.6x /
  1.7x over the prior state of the art on Intel CPU / ARM / NVIDIA GPU.
  Direct evidence that *allocating search budget jointly* across layers
  beats independent per-layer tuning.

- **Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization**
  (Li, Jamieson, DeSalvo, Rostamizadeh, Talwalkar — CMU/UCLA/Google, JMLR
  2018, arXiv 1603.06560) — https://arxiv.org/abs/1603.06560
  Foundational multi-fidelity result. Reports >10x speedup over random and
  standard BO on deep-learning and kernel HPO. Relevant because the vLLM
  config space has natural fidelity knobs: short-horizon benchmark,
  full-horizon benchmark, short vs. long sequences, one GPU vs. full cluster.

- **BOHB: Robust and Efficient Hyperparameter Optimization at Scale**
  (Falkner, Klein, Hutter — U. of Freiburg, ICML 2018) —
  https://proceedings.mlr.press/v80/falkner18a/falkner18a.pdf
  Combines TPE with Hyperband. The canonical reference for multi-fidelity BO
  outperforming both pure BO and pure Hyperband. For autoinfer this is the
  strongest baseline to beat: "random + Hyperband" has better sample
  efficiency than "BO alone" on most published HPO benchmarks.

- **NAS-HPO-Bench-II: A Benchmark Dataset on Joint Optimization of CNN
  Architecture and Training Hyperparameters** (Hirose, Yoshinari, Shirakawa
  — YNU, ACML 2021, arXiv 2110.10165) — https://arxiv.org/abs/2110.10165
  Builds 192K-configuration benchmark of (architecture, learning rate, batch
  size). Directly shows that the best architecture under default
  hyperparameters is not the best architecture under tuned hyperparameters —
  i.e., cross-layer coupling is strong and independent search is lossy.
  Companion: **Bag of Baselines for Multi-objective Joint NAS+HPO** (Guerrero
  Viu et al., 2021, arXiv 2105.01015) — https://arxiv.org/abs/2105.01015.

- **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized
  LLM Serving** (Zhong, Liu, Chen, Lin, Cao, Zhang, Jin, Zhang —
  PKU/UCSD/Ant, OSDI 2024, arXiv 2401.09670) —
  https://arxiv.org/abs/2401.09670 ; PDF
  https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf
  Reports up to 7.4x more requests or 12.6x tighter SLO compared to prior
  state-of-the-art, achieved specifically by co-optimizing *resource
  allocation and parallelism* per phase under TTFT/TPOT constraints. This is
  the cleanest LLM-serving-specific evidence that joint (engine-topology)
  search beats single-knob tuning. 18-month retrospective from the authors
  notes the disaggregation paradigm has been adopted by NVIDIA Dynamo,
  SGLang, vLLM, llm-d, LMCache, and MoonCake:
  https://haoailab.com/blogs/distserve-retro/

- **Llumnix: Dynamic Scheduling for Large Language Model Serving** (Sun et
  al., Alibaba, OSDI 2024) —
  https://www.usenix.org/conference/osdi24/presentation/sun-biao ; PDF
  https://www.usenix.org/system/files/osdi24-sun-biao.pdf
  Cross-instance scheduling layer on top of vLLM. Reports order-of-magnitude
  tail-latency improvement and up to 36% cost reduction. Supports the weaker
  version of C7: even without a formal joint search, coordinating scheduling
  decisions across instances (topology layer) yields gains not reachable by
  per-instance engine tuning.

- **Serving DNNs like Clockwork: Performance Predictability from the Bottom
  Up** (Gujarati et al., MPI-SWS / Emory, OSDI 2020) —
  https://www.usenix.org/conference/osdi20/presentation/gujarati ; PDF
  https://www.usenix.org/system/files/osdi20-gujarati.pdf
  Earlier-generation evidence (pre-LLM) that cross-layer consolidation of
  scheduling decisions yields very tight tail latencies (100ms p99.9999 at
  thousands of models). Cited as precedent for "joint control across layers
  beats independent per-layer heuristics" in inference serving.

- **REASONING COMPILER: LLM-Guided Optimizations for Efficient Model
  Serving** (Chen et al., arXiv 2506.01374, 2025) —
  https://arxiv.org/abs/2506.01374
  Pairs an LLM proposer with MCTS search over a compiler optimization space.
  Reports substantial speedups "with markedly fewer samples than leading
  neural compilers." Most directly relevant precedent for C7's LLM-guided
  *joint* (kernel + passes) variant. Sample numbers are not comparable to
  BOHB head-to-head, so treat as qualitative evidence.

- **Chameleon: Adaptive Code Optimization for Expedited DNN Compilation**
  (Ahn, Pilligundla, Yazdanbakhsh, Esmaeilzadeh — UCSD/Georgia Tech, ICLR
  2020, arXiv 2001.08743) — https://arxiv.org/abs/2001.08743
  Baseline reference that RL + adaptive sampling reduces hardware
  measurements in AutoTVM by ~2–4x. Establishes that adaptive search over a
  joint schedule space beats grid/random by a clear margin, prior to any LLM
  involvement.

## Honest assessment

How strong is "LLM beats BO" really? On published HPO benchmarks, pure LLM
proposers tie or beat BO in the *first tens of trials* and lose by the time
the budget is in the hundreds. The one study directly on a systems-flavored
space (2603.24647) reports the opposite of the popular headline: CMA-ES and
TPE *beat* LLM agents on fixed-compute LM tuning, with failure-region
tracking cited as the reason. The defensible position for autoinfer at a
~200-eval vLLM budget is: use the LLM as (a) warm-start / initial design,
(b) proposal generator on top of a TPE or CMA-ES surrogate, (c)
code-proposal operator for kernel-level edits where the surrogate has
nothing to model over. Do not expect a pure LLM-guided loop to beat BOHB on
a config sweep with hard OOM cliffs. The strongest "LLM does something the
surrogate can't" evidence is code/algorithm discovery (AlphaEvolve, Sakana
evolutionary merging, REASONING COMPILER) — not numeric knob tuning. That
distinction maps onto autoinfer's two modes: the "params" loop should
probably anchor on a classical sampler with LLM assistance; the "code" loop
is where the LLM earns its budget.

Is joint search a real finding or a handwave? It is real in well-scoped
pairs: (schedule x layout) in Ansor, (architecture x training hyperparams)
in NAS-HPO-Bench-II, (resource x parallelism per phase) in DistServe, and
(model-state predictability x admission control) in Clockwork. It is a
handwave when authors claim "joint optimization across the whole stack" —
the literature does not contain a rigorous single-study result that jointly
searches kernel + engine + topology with measured wins over layered
(decoupled) search. Multi-fidelity (Hyperband/BOHB) is the best-supported
single technique for stretching a small budget across expensive evaluations.

## Gaps

The most obvious missing piece is a study that fixes the search space to a
realistic vLLM configuration (max-num-seqs, max-num-batched-tokens,
gpu-memory-utilization, tensor-parallel, chunked-prefill toggle,
quantization mode, KV cache dtype) and compares pure random, BO (Vizier /
Optuna-TPE), BOHB, CMA-ES, LLM-only (OPRO-style), and LLM+BO hybrid (LLAMBO
/ SLLMBO / Centaur) at matched 100–500 evaluation budgets, with
OOM-as-failure handling, over real workloads (ShareGPT-style + long-context
+ bursty). No published study does this cleanly. The autoinfer project is
well positioned to be that study; the closest prior art is 2603.24647 on
autoresearch (small-LM tuning) and vLLM's own `auto_tune.sh` grid-search
harness, neither of which benchmarks against BO or LLM-guided search at a
matched budget.
