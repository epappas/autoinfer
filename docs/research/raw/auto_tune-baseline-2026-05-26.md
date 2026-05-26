# T-37 baseline — vLLM `auto_tune.sh` on Basilica (2026-05-26)

**Status:** COMPLETE — 3 baseline tuples captured.

This document records vLLM's published `benchmarks/auto_tune/auto_tune.sh`
baseline measurements on Basilica, against which Campaign 04 (autoinfer-L1
head-to-head) will be measured. Per the C04 recon's 2026-05-26 corrective
addendum, three baselines were captured ("Both, in sequence" + a third
H100 long-context anchor):

| Baseline | Hardware | Workload (input/output/max_model_len) | SLO | Purpose |
|---|---|---|---|---|
| **A** | 1× A100 spot | 1800 / 20 / 2048 | `MAX_LATENCY=infinite` | Max throughput on long-context, unconstrained |
| **B** | 1× A100 spot | 256 / 20 / 512 | `MAX_LATENCY=500 ms` | Throughput under realistic chat-shape SLO |
| **C** | 1× H100 spot | 1800 / 20 / 2048 | `MAX_LATENCY=500 ms` | auto_tune README's published target (long-context with SLO) |

All three ran on **vllm 0.21.0** (image `vllm/vllm-openai:v0.21.0`), with the
vllm git checkout pinned to commit `ad7125a431e176d4161099480a66f0169609a690`
(captured by the script's own `git rev-parse HEAD` write into each `result.txt`).

---

## Reproduction recipe

All runs used the autoinfer T-37 orchestrator (PR #41 + fixes #42-#45):

```
uv run python -u scripts/run_auto_tune_baseline.py \
    --gpu-models <A100|H100> --spot true --ttl-hours 3 \
    --input-len <1800|256> --output-len 20 --max-model-len <2048|512> \
    --max-latency-allowed-ms <10000000|500> \
    --artifacts-dir <local-out> --log-file <local-log> \
    --yes
```

The orchestrator builds a `vllm/vllm-openai:v0.21.0` Basilica deployment,
clones `vllm-project/vllm@v0.21.0`, renames the cloned `vllm/` Python
package dir out of the way (avoids shadowing the installed wheel), sed-
patches `auto_tune.sh` to bind on `localhost` (avoids Basilica's pod-
hostname routing), and runs `auto_tune.sh` with the recon-recommended
env block. The bootstrap source lives in `src/autoinfer/target/auto_tune.py`
(`AutoTuneBaselineSpec`).

Defaults preserved from `auto_tune.sh`:
- `NUM_SEQS_LIST="128 256"`
- `NUM_BATCHED_TOKENS_LIST="512 1024 2048 4096"`
- `TP=1`, `MIN_CACHE_HIT_PCT=0`

---

## Headline numbers

### Baseline A — A100, long-context, unconstrained throughput

**Best config (peak throughput at `request_rate=inf`):**

| Metric | Value |
|---|---|
| `max_num_seqs` | **256** |
| `max_num_batched_tokens` | **4096** |
| Request throughput | **8.53 req/s** |
| Goodput (= throughput, no SLO) | **8.53 req/s** |
| Output token throughput | 170.54 tok/s |
| Total token throughput | 15,519.44 tok/s |
| Mean TPOT | 256.55 ms |
| Mean E2EL | 63,747 ms (saturation queue) |
| P99 E2EL | 117,103 ms |

Long-context at saturation: prefill of 1800 tokens × concurrency 256 keeps
the GPU busy; the script's `--request-rate inf` measurement intentionally
runs at queue saturation so per-request latency is inflated. Throughput is
what matters in this regime.

### Baseline B — A100, short-chat, 500 ms SLO

**Best config:**

| Metric | Value |
|---|---|
| `max_num_seqs` | **256** |
| `max_num_batched_tokens` | **512** |
| Best request rate | **23 req/s** |
| Request throughput | **21.60 req/s** |
| **Goodput (= req meeting SLO/s)** | **21.39 req/s** |
| Output token throughput | 432.04 tok/s |
| Total token throughput | 5,962.12 tok/s |
| Mean TTFT | 69.29 ms |
| P99 TTFT | 126.41 ms |
| Mean TPOT | 16.37 ms |
| P99 TPOT | 22.32 ms |
| Mean E2EL | 380.35 ms |
| **P99 E2EL** | **494.60 ms** (under 500 ms SLO ✓) |

This is the most directly C04-comparable baseline: realistic chat-shape
workload, SLO-bounded, on A100 (autoinfer's tested hardware).

### Baseline C — H100, long-context, 500 ms SLO

**Best config:**

| Metric | Value |
|---|---|
| `max_num_seqs` | **256** |
| `max_num_batched_tokens` | **512** |
| Best request rate | **3 req/s** |
| Request throughput | **2.97 req/s** |
| **Goodput (= req meeting SLO/s)** | **2.97 req/s** |
| Output token throughput | 59.36 tok/s |
| Total token throughput | 5,401.63 tok/s |
| Mean TTFT | 104.00 ms |
| P99 TTFT | 180.36 ms |
| Mean TPOT | 11.25 ms |
| P99 TPOT | 18.29 ms |
| Mean E2EL | 317.76 ms |
| **P99 E2EL** | **457.27 ms** (under 500 ms SLO ✓) |

This is the auto_tune README's published target tuple. H100 makes the
500 ms SLO achievable on long-context (1800-token prefill dominates
per-request time) but throughput is capped at ~3 req/s.

---

## Full per-cell grids

### Baseline A grid

Workload: `INPUT=1800, OUTPUT=20, MAX_MODEL_LEN=2048, MAX_LATENCY=10^10 ms`.
GMU auto-found at 0.98 on first try.

| `max_num_seqs` | `max_num_batched_tokens` | rate | P99 E2EL (ms) | throughput (req/s) | goodput (req/s) |
|---|---|---|---|---|---|
| 128 | 512 | inf | 0.00 | 0.00 | 0.00 |  (cold-start; vllm killed mid-`torch.compile`) |
| 128 | 1024 | inf | 127,200 | 7.84 | 7.84 |
| 128 | 2048 | inf | 119,439 | 8.35 | 8.35 |
| 128 | 4096 | inf | 117,162 | 8.52 | 8.52 |
| 256 | 512 | inf | 135,588 | 7.31 | 7.31 |
| 256 | 1024 | inf | 126,673 | 7.86 | 7.86 |
| 256 | 2048 | inf | 119,361 | 8.36 | 8.36 |
| **256** | **4096** | **inf** | **117,104** | **8.53** | **8.53** | **← BEST** |

### Baseline B grid

Workload: `INPUT=256, OUTPUT=20, MAX_MODEL_LEN=512, MAX_LATENCY=500 ms`.

| `max_num_seqs` | `max_num_batched_tokens` | rate | P99 E2EL (ms) | throughput | goodput |
|---|---|---|---|---|---|
| 128 | 512 | inf | 0.00 | 0.00 | 0.00 |  (cold-start artifact) |
| 128 | 1024 | 18 | 497.93 | 17.14 | 16.96 |
| 128 | 2048 | 18 | 497.06 | 17.13 | 16.96 |
| 128 | 4096 | 18 | 499.65 | 17.13 | 16.96 |
| **256** | **512** | **23** | **494.60** | **21.60** | **21.39** | **← BEST** |
| 256 | 1024 | 19 | 498.08 | 18.04 | 17.86 |
| 256 | 2048 | 18 | 489.75 | 17.16 | 16.99 |
| 256 | 4096 | 18 | 498.31 | 17.15 | 16.97 |

Observation: `max_num_batched_tokens` plateaus at 1024 for 128 seqs; at
`max_num_seqs=256` the throughput jumps significantly only at the smallest
batched-tokens setting (512). Larger batched-tokens values force lower
sustainable request rates due to per-batch latency overhead.

### Baseline C grid

Workload: `INPUT=1800, OUTPUT=20, MAX_MODEL_LEN=2048, MAX_LATENCY=500 ms`.

| `max_num_seqs` | `max_num_batched_tokens` | rate | P99 E2EL (ms) | throughput | goodput |
|---|---|---|---|---|---|
| 128 | 512 | inf | 0.00 | 0.00 | 0.00 |  (cold-start artifact) |
| 128 | 1024 | 2 | 402.68 | 1.99 | 1.99 |
| 128 | 2048 | 2 | 430.10 | 1.99 | 1.99 |
| 128 | 4096 | 2 | 433.93 | 1.99 | 1.99 |
| **256** | **512** | **3** | **457.27** | **2.97** | **2.97** | **← BEST** |
| 256 | 1024 | 2 | 403.64 | 1.99 | 1.99 |
| 256 | 2048 | 2 | 430.49 | 1.99 | 1.99 |
| 256 | 4096 | 2 | 433.99 | 1.99 | 1.99 |

Observation: H100 long-context with 500 ms SLO bottlenecks on the
1800-token prefill. `max_num_seqs=256` + minimal `max_num_batched_tokens`
(512) wins because higher batched-tokens settings each force lower rates
under the SLO ceiling.

---

## Workload-feasibility notes (from earlier aborted attempts)

The recon's original tuple (`A100 / INPUT=1800 / OUTPUT=20 / 500 ms SLO`)
was tested in attempt 3 before being aborted: every cell measured P99
E2EL ≫ 500 ms even at `request_rate=1` (cell 128/1024 at rate=1 still
exceeded 1000 ms E2EL). The 1800-token prefill alone on A100 takes
several hundred ms; the SLO ceiling is unreachable on this hardware. The
"Both, in sequence" reframing (A: drop SLO, B: shorter input, C: faster
hardware) was chosen at that point so the comparison surface had
measurable goodput.

Earlier-attempt failure modes (all fixed before Baseline A landed):
| # | Issue | Fix | PR |
|---|---|---|---|
| 1 | `bc: command not found` in vllm/vllm-openai image | `apt-get install bc` in bootstrap | #43 |
| 2 | Cloned `vllm/` source shadowed the installed wheel → `ModuleNotFoundError: No module named 'vllm._C'` | `mv vllm/vllm vllm/.vllm_src_shadow_moved` after clone | #44 |
| 3 | `auto_tune.sh` line 21 `HOSTNAME=$(hostname)` picks Basilica pod name; `vllm bench serve --host POD_NAME` fails to connect from inside pod | `sed s|HOSTNAME=$(hostname)|HOSTNAME=localhost|` patch + verify-after-sed check | #45 |

---

## Cost summary

| Attempt | Hardware | Wall | Cost |
|---|---|---|---|
| 1 (bc missing) | A100 spot | ~3 min | ~$0.15 |
| 2 (source shadow) | A100 spot | ~3 min | ~$0.15 |
| 3 (hostname, aborted) | A100 spot | ~12 min | ~$0.50 |
| Baseline B 256/256 (deleted; infeasible SLO) | A100 spot | ~17 min | ~$0.15 |
| **Baseline A** (final) | A100 spot | ~26 min | **~$0.21** |
| **Baseline B revised** (final) | A100 spot | ~75 min | **~$0.61** |
| **Baseline C** (final) | H100 spot | ~67 min | **~$1.40** |
| **Total T-37** | | | **~$3.17** |

Within the recon's revised $3-5 budget. The C04 head-to-head campaigns
(C04a + C04b) will use Baselines B and C as the SLO-bounded references
and Baseline A as the unconstrained-throughput reference.

---

## What this enables for C04

Three honest comparable axes for autoinfer-L1:

1. **Direct comparable, SLO-bounded, A100 (use Baseline B).** autoinfer-L1
   on Llama-3.1-8B-Instruct / 1× A100 / INPUT=256, OUTPUT=20 / 500 ms SLO
   must beat **goodput 21.39 req/s** (or match goodput at fewer trials).
   This is the most direct C04 comparison — same hardware, same workload,
   same SLO, surrogate-vs-grid on the same surface.

2. **Direct comparable, SLO-bounded, H100 (use Baseline C).** autoinfer-L1
   on Llama-3.1-8B-Instruct / 1× H100 / INPUT=1800, OUTPUT=20 / 500 ms SLO
   must beat **goodput 2.97 req/s**. This matches the auto_tune README's
   published target — the highest-precedent comparison.

3. **Unconstrained throughput on long-context, A100 (use Baseline A).**
   autoinfer-L1 with no SLO must beat **throughput 8.53 req/s**. This
   isolates "can autoinfer find a better throughput-only config than
   auto_tune's grid" without latency confounding.

The C04 pre-reg will pre-register predictions against each of these
three reference values plus an explicit knob-surface comparison (C04a:
restrict autoinfer-L1 to the 2 knobs auto_tune searches; C04b: full
12-knob catalog).

---

## Artifacts

Local artifact dirs (each contains: `result.txt`, per-cell `vllm_log_*.txt`,
per-rate `bm_log_*.txt`, profile/, and the GMU-find log):

- `basilica-artifacts/t-37-baseline-A-a100-noslo-2026-05-26/2026_05_26_14_00/`
- `basilica-artifacts/t-37-baseline-B-a100-chatshort-slo500-2026-05-26/2026_05_26_14_46/`
- `basilica-artifacts/t-37-baseline-C-h100-longctx-slo500-2026-05-26/2026_05_26_16_29/`

Orchestrator stdout/log files:

- `basilica-artifacts/t-37-baseline-A-a100-noslo-2026-05-26.log`
- `basilica-artifacts/t-37-baseline-B-a100-chatshort-slo500-2026-05-26.log`
- `basilica-artifacts/t-37-baseline-C-h100-longctx-slo500-2026-05-26.log`

Deployment IDs (for audit; deployments deleted by orchestrator after
artifact fetch):

- Baseline A: `86a0df76-f6ac-47d8-b846-b8e2ffc13507`
- Baseline B revised: `151463c4-caab-49ef-81c6-af87d0818cea`
- Baseline C: `9a39c4d6-d362-4478-b51c-92c7699aa18d`

vllm commit captured in each `result.txt`: `ad7125a431e176d4161099480a66f0169609a690`
(vllm-project/vllm tag `v0.21.0`).

---

## Closing

T-37 is closed by the commit landing this document on `main`. The C04
pre-reg (separate PR) will cite this artifact as its baseline reference
and pre-register predictions for autoinfer-L1's goodput at each of the
three reference tuples.
