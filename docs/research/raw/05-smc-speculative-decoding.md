# SMC Speculative Decoding (raw, 2026-04-23)

Scholastic extraction from primary sources. This file records WHAT the
method is, WHY it exists, HOW it works, and what remains un-verified from
the publicly visible material at the time of writing. Anything not
directly attested by a source is called out as such — no synthesis
beyond the source material.

## Source materials

- **Paper.** Emara, Y.; Barba da Costa, M.; Chang, C.-C.; Freer, C.;
  Vieira, T.; Cotterell, R.; Abdelfattah, M. S. *Faster LLM Inference
  via Sequential Monte Carlo.* arXiv:2604.15672 [cs.LG], submitted
  2026-04-17. DOI 10.48550/arXiv.2604.15672. PDF ~533 KB.
  https://arxiv.org/abs/2604.15672 · https://arxiv.org/pdf/2604.15672
- **Blog.** Emara, Y. *SMC-SD* (Makora engineering blog), 2026-04-20.
  https://makora.com/blog/smc-sd
- **Reference implementation.** `abdelfattah-lab/smcsd`
  (Apache-2.0). Fork of SGLang with SMC decoding backend and a custom
  Triton KV-cache assignment kernel.
  https://github.com/abdelfattah-lab/smcsd

Author affiliations are not attested verbatim in the fetched material,
but authorship signatures (Vieira, Cotterell for formal-language/LM
inference; Abdelfattah for the GitHub org `abdelfattah-lab`) are
consistent with ETH Zürich NLP and Cornell ECE. Do not cite affiliations
as confirmed until the PDF is read.

---

## WHAT — precise algorithm

**Name:** Sequential Monte Carlo Speculative Decoding (SMC-SD).

**One-sentence definition (from the paper abstract):** SMC-SD
"replaces token-level rejection with importance-weighted resampling
over a population of draft particles."

**Distinguishing property (from the blog):** "Every round, each
particle is extended by exactly K+1 tokens — no rollback, no
truncation." Verification is "a vectorized, fixed-size operation with
no rollback" (paper).

**Positioning relative to prior work.**

- *Strict-accept speculative decoding* (EAGLE / EAGLE-2 / EAGLE-3,
  Medusa, vLLM's in-tree spec-dec, DeepSeek MTP). Drafts one candidate
  trajectory; target model accepts a prefix and rejects the first
  diverging token; the rejected suffix's KV cache is rolled back.
  Distribution-preserving but throughput-fragile: acceptance-rate
  dependent, with documented p99/latency regressions under load
  (TurboSpec/SmartSpec; Nebius MoE blog, cited in
  `references-L1-engine-config.md` under C2).
- *SMC-SD.* Drafts **N** candidate trajectories ("particles") per
  request; each round extends every particle by a fixed **K+1** tokens;
  the target model scores the extensions; particles are reweighted and
  resampled (systematic or multinomial) when the effective sample size
  crosses a threshold. **All drafted tokens are accepted** — there is
  no KV-cache rollback path.

The abstract frames the method as "a principled approximate inference
scheme that trades exactness for additional speed" and claims
"theoretical bounds on its per-step approximation error". The exact
statement of those bounds is not reproduced in the abstract or blog;
it is in the paper body which we have not yet read.

---

## WHY — motivation

### 1. Rejection truncates the draft block

The abstract states: "rejection truncates the draft block at the first
error, [and] throughput degrades when draft and target diverge." This
is the pathology. When the draft model's distribution diverges from
the target's — which is common on long-tail tokens, reasoning, or
code — a single rejected token discards all KV state computed for the
suffix. Throughput therefore scales poorly with draft quality in the
regimes where speculative decoding is most needed.

### 2. Compute is outpacing memory bandwidth

The blog explicitly argues: "compute is growing faster than memory
bandwidth," citing Blackwell-generation GPUs which keep HBM bandwidth
roughly constant while "scaling FLOPs hard." Autoregressive LLM decode
is memory-bandwidth-bound (the weight matrices are streamed per token).
A **population** method that runs N parallel particles through one
target forward pass converts idle FLOPs into reduced wall-clock per
token — the arithmetic intensity per memory transfer rises with N.

This is the same "free compute, scarce bandwidth" framing that
justifies batching, CUDA graphs, and speculative decoding in general.
SMC-SD monetises it via a population rather than a single draft.

### 3. Lossy-but-bounded is a valid operating point

The authors explicitly accept approximation error as a knob. This
differs from EAGLE/Medusa's "lossless by strict sampling" framing and
also from Medusa-2's ambiguous position (it fine-tunes the backbone,
which changes the target distribution before spec-dec even runs). By
making the error budget a first-class parameter, the user chooses a
point on the approximation ↔ throughput Pareto. The blog: "trade a
small, bounded amount of approximation error for substantial
throughput gains."

---

## HOW — algorithm (as attested by blog + README)

The abstract and blog do not give full pseudocode. What follows is
assembled only from what is directly attested; the paper body (not
yet read) contains the formal statement.

**Per-request state.** A population of `N = --smc-n-particles`
trajectories, each carrying its own KV cache extension and an
importance weight.

**Per-round loop.**

1. **Draft extension.** Each particle is extended by `γ+1` tokens
   using the draft model, where `γ = --smc-gamma`. The blog shows
   `K = 4` as an example; the flag is named `gamma` in the CLI. (The
   mapping between `K` in the blog and `γ` in the CLI is not
   explicitly documented — it is likely `γ = K` with the "+1" being
   the verified target token, but this should be confirmed in code.)
2. **Target scoring.** The target model scores all N particles'
   extensions in a single vectorised forward pass — "a vectorized,
   fixed-size operation with no rollback."
3. **Importance reweighting.** Each particle receives a weight derived
   from the target/draft likelihood ratio over the extension. The blog
   illustrates weights like `.03` (low) and `.79` (high); the exact
   form is the standard SMC importance weight
   `w ∝ p_target(x) / q_draft(x)`, but this is inference by convention
   and not verbatim in the fetched material.
4. **Resample decision.** Compute effective sample size
   `ESS = 1 / Σ w_i²` (normalised weights). If
   `ESS < N × --smc-resample-threshold`, resample the population
   using `--smc-resample-method ∈ {systematic, multinomial}`. A
   threshold of `0` disables resampling.
5. **Emit.** Accepted tokens are the ones the population agrees on,
   or a consensus pick — the exact emission rule is in the paper body
   and is *not* directly quoted in the material we fetched.

**KV cache assignment.** The repo ships a custom Triton kernel under
`sgl-kernel/` to handle particle-to-cache-slot assignment when the
population is resampled (duplicating a high-weight particle's KV and
evicting a pruned one without a full copy). The existence of this
kernel is attested by the README directory layout; its exact
implementation has not been read.

**What is missing from our fetched extraction.** The formal per-step
error bound, the emission rule, the ESS accounting at batch level
(is ESS per-request or global?), and the precise weight formula are
all in the paper body, not in the abstract or blog. Mark as
follow-ups to confirm from the PDF.

---

## Tunable knobs (exhaustive, from README)

These flags are exposed on SGLang's CLI in the fork.

| Flag | Type | Role | Notes |
|---|---|---|---|
| `--speculative-algorithm` | enum | Enables SMC when set to `SMC`. | Coexists with the existing spec-dec algorithms in SGLang. |
| `--speculative-draft-model-path` | path | Draft model checkpoint. | Example pair: `meta-llama/Llama-3.2-1B-Instruct` drafting for `meta-llama/Llama-3.1-8B-Instruct`. |
| `--smc-n-particles` | int | Population size `N` per request. | Example value: 8 (throughput example), 12 (accuracy example). Default not stated in README. |
| `--smc-gamma` | int | Draft tokens per speculative step `γ`. | Example value: 8. Blog shows `K=4`; the naming gap (`K` vs `γ`) is not reconciled in the README. |
| `--smc-draft-temperature` | float | Sampling temperature for the draft model. | Example: 0.7. |
| `--smc-target-temperature` | float | Scoring temperature for the target model. | Example: 0.7. Separate from draft temperature. |
| `--smc-resample-threshold` | float | Resample when `ESS < N × threshold`. | `0` disables resampling. |
| `--smc-resample-method` | enum | `systematic` or `multinomial`. | Standard SMC choices; systematic has lower variance, multinomial is simpler. |

Inherited SGLang knobs that the README uses in benchmarks:

| Flag | Benchmarked value | Role |
|---|---|---|
| `--attention-backend` | `fa3` | FlashAttention-3 kernel backend. |
| `--mem-fraction-static` | `0.60` | Static memory fraction reserved for KV. |
| `--max-running-requests` | `128` | Batch-concurrency cap. |
| `--cuda-graph-max-bs` | `128` | Max batch size under CUDA graphs. |
| `--dataset-name` | `sharegpt` | Throughput benchmark workload. |
| `--num-prompts` | `200` | Throughput prompt count. |

**Knob-space dimensionality estimate for L1 policy search.** Five
continuous/ordinal SMC knobs × four ordinal SGLang knobs × two
model-pair choices = roughly 10 axes with strong interactions. This
is squarely in the regime where TPE or CMA-ES with Hyperband
performs well and where the P7 warm-start-from-LLM operator adds
signal (the warm-start LLM can read the blog and seed with
`N=8, γ=8, threshold=0.5, method=systematic, temps=0.7`).

---

## Reference implementation — repo anatomy

From the README listing, attested directory structure:

- `python/sglang/srt/smc/` — Python backend:
  - `smc_workers.py` — draft/target forward orchestration
  - `smc_resampler.py` — systematic/multinomial resampling
  - `smc_manager.py` — per-request population lifecycle
  - `smc_info.py` — metadata/telemetry for particles
  - `smc_utils.py` — helpers
- `sgl-kernel/` — custom Triton kernels (KV-cache assignment)
- `scripts/smc/` — benchmark scripts (incl. `accuracy_test_gsm8k.py`)
- `docs/smc/` — architecture documentation (not read)
- `test/`
- `LICENSE` (Apache-2.0), `README.md`, `.gitignore`

Install path (README verbatim):

```
# create python-3.12 venv with uv
uv pip install -e "python"
```

No pinned dependency versions are given in the README. The fork is a
point-in-time clone of SGLang — the exact SGLang commit it forks from
is **not documented in the README** and must be read from the git
history before any adapter work.

**Stability disclaimer (verbatim):** "This repository is under active
development. APIs, configuration flags, and internal interfaces may
go through breaking changes."

---

## Reproduction — exact commands (verbatim from README)

### Throughput (ShareGPT)

```
python -O -m sglang.bench_offline_throughput \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --speculative-algorithm SMC \
  --speculative-draft-model-path meta-llama/Llama-3.2-1B-Instruct \
  --smc-n-particles 8 --smc-gamma 8 \
  --smc-draft-temperature 0.7 --smc-target-temperature 0.7 \
  --attention-backend fa3 \
  --mem-fraction-static 0.60 \
  --max-running-requests 128 \
  --cuda-graph-max-bs 128 \
  --dataset-name sharegpt \
  --num-prompts 200
```

### Accuracy (GSM8K, 400 questions)

```
python scripts/smc/accuracy_test_gsm8k.py \
  --mode smc \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --draft-model meta-llama/Llama-3.2-1B-Instruct \
  --particles 12 --gamma 8 \
  --temperature 0.7 \
  --attention-backend fa3 \
  --max-running-requests 128 \
  --cuda-graph-max-bs 128 \
  --num-questions 400
```

Note the discrepancy: the accuracy script uses `--particles` and
`--gamma` (no `smc-` prefix) and a single `--temperature`, while the
`bench_offline_throughput` CLI uses the `--smc-*` prefix and separate
draft/target temperatures. This is likely a wrapper-script artefact,
not a semantic difference; confirm from code before building the
adapter.

---

## Reported results (all numbers from primary sources)

### Headline (paper abstract + blog)

- **5.2×** throughput over autoregressive decoding.
- **2.36×** throughput over state-of-the-art speculative decoding
  (the blog identifies the SOTA baseline as SGLang's own spec-dec).
- **Within 3%** accuracy loss across "reasoning,
  instruction-following, and coding benchmarks."

### Test matrix (as attested)

- **Model (headline):** Llama 70B.
- **Hardware (headline):** 4 × NVIDIA H100.
- **Draft/target pair (repo examples):** Llama-3.2-1B-Instruct
  drafting Llama-3.1-8B-Instruct — a 1B/8B pair, not the 70B the
  headline uses. The 70B headline pair is not specified in the
  abstract or README.
- **Benchmark surfaces named:**
  - **Throughput:** ShareGPT.
  - **Accuracy:** GSM8K (400 Q). Paper mentions MATH, AlpacaEval,
    and (per an earlier fetch) DS-1000, but these names appear in
    summarised form in our fetches, not as verbatim quotes.

### What the material does **not** report (gaps)

- Latency distributions (p50/p90/p99). All headline numbers are
  throughput averages. This matters for SLO gating.
- Variance across seeds or request orderings.
- Memory footprint as a function of `N`. Population methods scale
  KV usage roughly linearly in `N`; the paper must discuss this but
  the number isn't in the abstract.
- Breakdown of the 3% accuracy loss per task. "Within 3%" is an
  aggregate; per-task deltas are what a quality gate should enforce.
- Acceptance-rate analogue. Strict-accept spec-dec reports
  "acceptance length"; the SMC analogue is ESS trajectory and
  resample frequency — not surfaced in the abstract.

---

## Theoretical position (as attested)

The abstract asserts "theoretical bounds on its per-step approximation
error." The bound statement itself is in the paper body. Until that
is read, we treat the bound as a claim rather than a verified
theorem.

Conceptually: standard SMC gives an unbiased Monte Carlo estimate of
the target expectation as `N → ∞`, with variance scaling as `1/N`
under non-degenerate weights. Resampling when ESS drops controls
particle degeneracy. The paper's "bounded per-step error" is
presumably a finite-`N` result over the K+1 extension — we should
confirm whether the bound is in KL, TV, or log-likelihood, and what
it depends on (draft-target divergence, `N`, `γ`).

---

## Limitations and roadmap (from README)

**Stability.** Pre-1.0; breaking changes expected.

**Roadmap items listed in README.**

1. **EAGLE support.** SMC-SD composed with EAGLE draft trees. If
   realised, this narrows the gap with the EAGLE family: SMC
   supplies the population/no-rollback machinery, EAGLE supplies
   tree-attention and calibrated draft proposals.
2. **Async/delayed resampling.** Resample off the critical path.
   Reduces the synchronisation cost that systematic resampling
   currently incurs.
3. **Disaggregation (draft/target split).** Run draft and target on
   different devices or groups — the P/D-disagg pattern applied to
   speculative decoding. Strong coupling to autoinfer's L2 topology
   axis.

---

## Relevance to autoinfer — WHY we should care

Each bullet is tied to a principle (P#) or claim (C#) from
`docs/research/references/00-hypothesis-seed.md`.

### As a target workload

1. **L1 target surface (P1, P3, C1).** The five SMC knobs plus the
   inherited SGLang scheduler/memory flags form a clean ~10-axis
   search space with strong interactions and a Pareto-tunable
   error budget. This is the best-case L1 target: low-dim, fast to
   evaluate per trial once deployed, with a published Pareto that
   lets us validate the policy's found optima.

2. **First forcing function for multi-engine L1.** Current L1
   adapts vLLM `EngineArgs`. SMC-SD is implemented in an SGLang
   fork. Absorbing it requires either (a) a sibling
   `l1_engine/sglang.py` adapter, or (b) waiting for upstream vLLM
   to adopt SMC (no such PR is attested today). This decision has
   been deferred across other candidate workloads; SMC-SD is the
   one that makes the deferral expensive enough to resolve.

3. **P8/C9 load-bearing.** "Within 3% accuracy loss" is empirical
   and per-workload. Cached-logit divergence gates cannot validate
   a sampling-equivalent-up-to-ε algorithm. The FP16 reference
   replica plus task-level accuracy gate (currently stubs at
   `harness/gate.py`, `harness/replica.py`) is the *only*
   principled verifier for this method. SMC-SD promotes gate + replica
   from "nice to have" to "blocking for this workload."

4. **P9 maps cleanly.** Typed `FailureRecord` categories for
   SMC-SD:
   - `ESS_COLLAPSE` — population degeneracy; detect when
     resample frequency → 1 per round for sustained windows.
   - `OOM_PARTICLES` — `N × batch × KV_per_particle` exceeds
     `mem-fraction-static`. Reproducible pre-run from knobs.
   - `ACCURACY_REGRESSION` — reference-replica gate fails
     per-task threshold even if aggregate is within 3%.
   - `THROUGHPUT_INVERSION` — slower than strict-accept spec-dec
     at current `(batch, draft/target pair, GPU)`. The regime
     where the paper's averages hide a regression corner.

5. **L3 overlap.** The repo ships a custom Triton KV-cache
   assignment kernel and depends on FlashAttention-3. Both sit in
   L3 kernel territory. Cross-layer stale-signal invalidation (P4)
   applies: when L2 changes GPU class (H100 → A100) or TP degree,
   the KV assignment kernel's optimal layout changes, which shifts
   the `(N, γ)` Pareto in L1.

### As evidence for the hypothesis seed

- **C1 (real knobs worth searching): +1** — five new documented
  knobs with explicit Pareto behaviour.
- **C2 (spec-dec tradeoffs): extends the claim.** SMC-SD is the
  first lossy-but-bounded spec-dec variant in the corpus. The
  TurboSpec/SmartSpec finding that throughput-only benchmarks hide
  latency regressions applies here in a new form — ESS collapse
  and resample-frequency spikes replace the strict-accept rejection
  latency story.
- **C6 (pure-LLM search insufficient): +1** — no LLM proposes
  "sequential Monte Carlo" from scratch, but once the adapter
  exists, tuning `(N, γ, threshold, method, temps)` is exactly the
  surrogate's job.
- **C9 (reference replica required): +1** — "bounded approximation
  error" is empirically verifiable only with a live reference
  replica; a cached-logit gate would silently pass.

---

## Open questions

These are what we should answer before shipping an adapter, grouped
by who can answer them.

### Read the paper body to resolve

- Exact form of the per-step approximation-error bound (KL? TV?
  log-likelihood? what does it depend on?).
- Emission rule — how a single output token is chosen from N
  particles after each round.
- Whether ESS is per-request or global across a served batch.
- Scaling: is the 5.2×/2.36× stable across `N ∈ {2, 4, 8, 16, 32}`
  or does it collapse at one extreme?
- The 70B/4×H100 configuration details (TP degree, batch size,
  sequence length, draft model for the 70B headline).

### Read the code to resolve

- The CLI naming gap: is `smc-gamma` the blog's `K` or `K+1`?
- SGLang base commit the fork is built on (needed for the adapter).
- Whether the custom Triton KV kernel is backend-specific (fa3 only,
  or also paged-attention V2).
- How the accuracy script's `--temperature` relates to the serving
  CLI's separate draft/target temperatures.

### Autoinfer-specific follow-ups

- Does SMC-SD compose with RadixAttention (SGLang's prefix-sharing)?
  Particle divergence could defeat prefix sharing unless all
  particles share the same prompt prefix, which they do by
  construction — so likely yes, but under what hit-rate regime does
  the gain survive?
- Under what `(N, γ, batch, GPU)` regimes does SMC-SD regress below
  strict-accept spec-dec? Paper reports averages; the policy's job
  is to surface the regression corners on *our* workload mix.
- Does the 3% aggregate hide a task where the method is catastrophic
  (e.g. code correctness at zero-temperature)? A per-task gate is
  the only answer.
- Could the SMC population be shared across co-located requests
  (batch-level importance sampling)? Out of scope for absorbing the
  method but an interesting extension once SMC is in L1.

---

## What is *not* in this document (honesty register)

- We did not read the 533 KB PDF in full; the abstract was fetched
  but the body was not. The `arxiv.org/pdf/...` ingest into Alexandria
  failed with an embedded-null-byte worker error; the
  `arxiv.org/abs/...` ingest was queued. All paper-body claims in
  this document are derived from the abstract plus the blog.
- We have not read the SGLang fork's source, only the README. All
  claims about `smc_resampler.py`, the Triton kernel, etc., rest on
  README directory listings.
- We have not reproduced any of the reported numbers. All speedup
  and accuracy figures are as-reported-by-authors.

---

## References (full)

### Primary

- Emara et al., arXiv:2604.15672 (2026-04-17) —
  https://arxiv.org/abs/2604.15672
- Emara, Y., Makora blog, 2026-04-20 —
  https://makora.com/blog/smc-sd
- Repo `abdelfattah-lab/smcsd` (Apache-2.0) —
  https://github.com/abdelfattah-lab/smcsd

### Adjacent autoinfer corpus

- SGLang (RadixAttention, frontend+runtime) — see
  `docs/research/raw/01-sglang.md`. SMC-SD is implemented on a fork
  of this system.
- vLLM L1 surface and V1 architecture — see
  `docs/research/raw/references-L1-engine-config.md` C1 (vLLM tunable
  surface) and Gordic's V1 walkthrough. The SMC adapter will parallel
  the vLLM adapter in shape.
- Speculative-decoding priors (EAGLE/EAGLE-2/EAGLE-3, Medusa,
  DeepSeek MTP, SmartSpec/TurboSpec, Nebius MoE) — see
  `references-L1-engine-config.md` C2. SMC-SD extends C2 with the
  first lossy-but-bounded entry.
- Quality-gate / batch-invariance background (Thinking Machines Lab,
  SGLang invariance follow-up) — see `references-L1-engine-config.md`
  C3. SMC-SD's per-step error bound is meaningless without an
  invariance-respecting gate.
