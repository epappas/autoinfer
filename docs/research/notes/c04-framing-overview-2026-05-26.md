# C04 framing — plain-language overview (2026-05-26)

> Companion to the campaign pre-reg at
> `docs/research/campaigns/04-l1-autotune-comparable-2026-05-26.md` and
> the T-37 baseline at `docs/research/raw/auto_tune-baseline-2026-05-26.md`.
> This note exists so a future contributor — agent or human — can pick
> up the project mid-stream and understand what C04 is actually for,
> without having to reverse-engineer the intent from individual PRs.

---

## What this project is trying to prove

**autoinfer's load-bearing claim:** a search system that simultaneously
tunes (a) the inference engine's knobs, (b) the hardware topology it
runs on, and (c) the GPU kernels inside it — with a shared evaluation
harness and feasibility classifier — finds *better deployment recipes*
than any single-layer specialist tool, on the same trial budget.

That is a strong claim and has not been proven yet. The way you prove
it is: pick the most directly-published per-layer specialist (vLLM has
one for the engine-config layer), set up an honest head-to-head on a
fixed workload, and either beat them or learn why you can't. Then
repeat across layers and workloads until the claim either holds or
breaks.

---

## What T-37 was

**T-37 is the answer to the question "what number are we trying to
beat?"**

vLLM ships a script called `benchmarks/auto_tune/auto_tune.sh`. Their
team uses it to recommend engine-config defaults. It is a small bash
grid-search over two knobs — how many sequences can run concurrently
(`max_num_seqs`) and how big the per-step token budget is
(`max_num_batched_tokens`) — looking for the highest throughput that
stays under a configured latency ceiling. **It is the published,
citable, reproducible baseline at the engine-config layer.**

T-37's only job: run that script on Basilica (so we control the
conditions), capture the throughput it picks, and write it down. That
number becomes the line in the sand for the next experiment. T-37
does not test autoinfer. It just measures the opponent.

### What T-37 actually produced

Not one baseline, three. The first attempt revealed that the original
recon's chosen workload (Llama-3.1-8B / 1× A100 / INPUT=1800,
OUTPUT=20 / 500 ms SLO) was **physically infeasible** — the
1800-token prefill alone on A100 takes hundreds of ms, before
generation starts. So the experiment was reframed as three
independent reference points (per the C04 recon addendum's
"Both, in sequence" + H100 anchor decision):

| Baseline | The question it answers | Best result |
|---|---|---|
| **A** (A100, INPUT=1800/OUTPUT=20, no SLO) | "How many req/sec can vLLM grind through at peak, ignoring tail latency?" | **8.53 req/s** |
| **B** (A100, INPUT=256/OUTPUT=20, 500 ms SLO) | "How many req/sec can vLLM serve at chat-realistic latency on commodity hardware?" | **21.39 req/s** (goodput) |
| **C** (H100, INPUT=1800/OUTPUT=20, 500 ms SLO) | "What's vLLM's published target throughput on the hardware their team optimised for?" | **2.97 req/s** (goodput) |

Each one is a different framing of "what does the specialist tool
achieve here." Total cost ~$3.17 across all attempts. Every per-cell
measurement, every server log, every benchmark detail is archived
locally; the orchestrator code that produced them is reproducible
from `main`.

### What the 6 PRs that landed during T-37 were about

T-37 only "ran a script", but it took 6 PRs to land cleanly because
the vLLM container image is real software and `auto_tune.sh` is a
real bash script. Running them inside a Basilica deployment exposed
three concrete environmental quirks:

1. The image was missing `bc` (basic calculator) — the script needs
   it for the GMU-decrement floating-point math (PR #43).
2. The script does `cd vllm/` before invoking `vllm bench serve`,
   and Python's import-path resolution then picked the cloned source
   tree (no compiled `_C` extension) instead of the installed wheel
   (PR #44).
3. The script auto-detected the host's hostname (`$(hostname)`) to
   pass to `vllm bench serve --host`, which inside a Basilica pod
   resolves to a cluster-internal address the bench client can't
   reach from within the same pod (PR #45).

Each was a real bug in the deployment setup, each got a real fix
with a regression test. The scaffolding (PRs #41, #42) and the final
artifact doc (PR #46) bracket the three diagnostic fixes. None of
this is theatre — it is the cost of being allowed to make an honest
comparison.

---

## What C04 is

**C04 is the actual head-to-head experiment.** Now that the T-37
baselines exist, you run autoinfer's L1 engine-config search on the
exact same workloads and compare. That's the entire campaign.

There is a structural choice in *how* you compare, because vLLM's
`auto_tune.sh` only searches 2 knobs and autoinfer searches 12. The
C04 recon's "Both, in sequence" decision means C04 covers two
sequential sub-campaigns:

### C04a — the conservative comparison

Restrict autoinfer-L1 to the same 2 knobs `auto_tune.sh` searches
(`max_num_seqs` × `max_num_batched_tokens`). Everything else fixed
at sensible defaults matching auto_tune's behaviour. Same workload
as the chosen T-37 baseline. The question is:

> **Does a Bayesian-optimisation surrogate beat a bash grid on the
> same 2-knob surface in the same trial budget?**

This is the smaller claim. Winning it shows the *search method* is
better than the *search method* used by vLLM — a real but narrow
result. Losing it (a bash grid beating the surrogate on its own
home turf) is a problem with the core search engine that needs
fixing before scaling to multiple layers.

### C04b — the actual thesis test

Let autoinfer use its full 12-knob catalog (kv-cache dtype,
attention backend, quantization, prefix caching, the V1 knobs
landed in T-33, etc.). Same workload as C04a. The question is:

> **Does the wider search find recipes the 2-knob grid literally
> cannot reach, and are those recipes meaningfully better?**

This is the load-bearing claim. The whole thesis is that joint
search across more axes wins because the axes interact —
`kv_cache_dtype=fp8` × `attention_backend=FLASHINFER` × certain
batch sizes opens up combinations a 2-knob grid can't see. C04b
tests whether that's actually true for L1 alone, before we even
start thinking about L2 and L3.

### Four possible outcomes — what each one means

| C04a | C04b | Interpretation |
|---|---|---|
| Win | Win | Both the search method *and* the wider surface help. Clean thesis support at L1; proceed to L2/L3 with confidence. |
| Win | Loss | Surrogate is better than grid on the same surface, but extra knobs don't pay off at this workload. *Publishable nuance* — narrows the claim from "joint search wins" to "joint search wins on workloads where the knobs interact." |
| Loss | Win | Surrogate is worse than grid on the 2-knob surface, but the wider surface compensates. Suggests the surrogate has a bug (it should never lose to a grid in its own search space) — fix that first. |
| Loss | Loss | The core thesis is in trouble at L1. Wider search doesn't help and the search engine is weaker than grid. Major rethink required. |

This is what an honest experiment looks like: pre-register what each
outcome means *before* you see data, and let the data sort itself
into one of them.

---

## What's actually on `main` right now

**14 PRs from this session, organised by what they enable:**

- **Search engine improvements.** The feasibility classifier now
  distinguishes between *kinds* of failure (OOM looks different from
  a quality regression looks different from a startup error), so the
  surrogate can learn each failure region separately. This was the
  load-bearing item; without it, the surrogate spends most of its
  trial budget on infeasible configurations and a bash grid wins by
  sheer hit-rate. (PR #34, T-26c)

- **Catalog completeness.** The engine-config search space now
  covers the V1 knobs vLLM ships with — chunked-prefill threshold,
  eager-execution toggle — so the head-to-head isn't a smaller
  surface than `auto_tune.sh`. (PR #35, T-33)

- **Honest measurement.** The workload driver passes a real SLO to
  vLLM's bench, the optimisation axis switches from raw throughput
  to *requests-meeting-SLO* when an SLO is set, and the reference
  replica's determinism levers (seed, batch-invariant kernels,
  multiprocessing mode) are a typed contract instead of ad-hoc env
  vars. (PRs #36, #38, T-34, T-36)

- **Reproducibility.** The workload-corpus sha256 and the per-trial
  seed are recorded in every run's metadata; the vLLM version is
  locked to a specific Docker tag matching the Basilica image; every
  campaign launch can be reproduced from a single commit. (PRs #37,
  #40, T-35 + the lock upgrade)

- **The baseline numbers.** Those three goodput figures (8.53 /
  21.39 / 2.97 req/s) with full per-cell grids, server logs,
  benchmark logs, and deployment provenance. (PRs #39-#46,
  the C04 recon retraction + T-37)

**What is not on `main`:** any data point that says autoinfer is
better or worse than vLLM at anything. That data does not exist yet.
C04 is what generates it.

---

## The honest framing

If a reviewer asked today "what have you proven?", the answer is:

- *Engineering*: a defensible head-to-head can be run honestly. The
  harness, search policy, reproducibility primitives, and reference
  measurements are in place.
- *Research*: nothing yet about whether autoinfer beats vLLM. T-37
  is the opponent's number; C04 is the match.

That is an honest position. The thing that was hard about T-37 was
not the conclusion — it was making sure the comparison would be a
comparison and not theatre. Three discovered fixes, three reframings
of the workload, one published reference set we can actually point to.

The next move is straightforward: write the C04 pre-reg (a no-GPU
doc PR) that names the three baselines, pre-registers what autoinfer
will be measured against in each one, and writes down what each
outcome means *before* the GPU run that produces the answer.

---

## Pointers for the next agent

- The campaign pre-reg lives at
  `docs/research/campaigns/04-l1-autotune-comparable-2026-05-26.md`
  (next-session deliverable; not yet on `main` when this note lands).
- The T-37 baseline artifact (with all per-cell numbers) is at
  `docs/research/raw/auto_tune-baseline-2026-05-26.md`.
- The original C04 recon (with the 2026-05-26 corrective addendum
  documenting the three errors found and the "Both, in sequence"
  reframing) is at
  `docs/research/notes/c04-l1-autotune-comparable-recon.md`.
- The strategic anchor for *why* C04 sits in a six-phase program is
  `docs/research/roadmap/best-tuning-proof-program.md`.
- The thesis + measurable claims (C1-C9) + design principles
  (P1-P12) are at `docs/research/references/00-hypothesis-seed.md`.

If you are picking this up mid-stream, read this note first, then
the T-37 baseline, then the C04 pre-reg (when it lands), and only
then dive into individual PRs.
