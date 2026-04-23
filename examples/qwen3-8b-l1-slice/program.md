# qwen3-8b-l1-slice — iteration zero example

First runnable autoinfer slice: search vLLM's L1 engine-config surface
for Qwen3-8B on a single H100 (or equivalent 80 GB GPU).

## Prerequisites

- Python 3.10+ with `uv`.
- A GPU that fits Qwen3-8B in FP16 (~16 GB weights + KV). H100-80GB
  is the reference; A100-80GB or 4090 with reduced `max_num_seqs`
  works.
- vLLM installed: `uv sync --extra dev --extra vllm`.
- `nvidia-smi` on `$PATH` (the adapter uses it to sample peak HBM).
- Two free TCP ports: 8000 (candidate) and 8001 (reference).

## One-time setup

1. **Start the reference replica** on a separate GPU at FP16. It must
   stay running for the entire search. Port 8001 matches
   `harness.gate.replica_uri` below.

   ```bash
   CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-8B \
     --port 8001 --dtype auto --gpu-memory-utilization 0.85
   ```

2. **Fetch a workload trace.** ShareGPT-replay is the default in the
   serving-benchmark literature. Place the result at
   `./trace/sharegpt_sample.jsonl`. One option:

   ```bash
   mkdir -p trace && \
   curl -L https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json \
     -o trace/sharegpt_full.json && \
   python -c "
   import json, random
   random.seed(0)
   convs = json.load(open('trace/sharegpt_full.json'))
   sample = random.sample(convs, 500)
   with open('trace/sharegpt_sample.jsonl', 'w') as f:
       for c in sample:
           turns = c.get('conversations', [])
           if turns and turns[0].get('value'):
               f.write(json.dumps({'prompt': turns[0]['value']}) + '\n')
   "
   ```

3. **Build a quality-gate prompt set** at `./quality/500_prompts.jsonl`.
   500 held-out prompts the gate feeds to both candidate and reference
   to compare per-token logits. You can reuse a slice of the ShareGPT
   sample (different seed) or any diverse 500 prompts you trust.

   ```bash
   mkdir -p quality && \
   head -500 trace/sharegpt_sample.jsonl > quality/500_prompts.jsonl
   ```

## Run

From this directory:

```bash
uv run autoinfer validate ./config.yaml
uv run autoinfer run ./config.yaml
```

The runner:

1. Loads `knobs.yaml` from `src/autoinfer/layers/l1_engine/`.
2. Warm-starts 8 configs from the deterministic seed (vLLM defaults)
   then hands off to Optuna TPE for the remaining `max_trials - 8`
   trials.
3. For each trial: spawns `vllm serve` on port 8000 with the proposed
   config, runs `vllm bench serve` against the trace, then runs the
   quality gate (logit KL + batch invariance) against the reference on
   port 8001. Rejects configs whose `mean_kl > max_kl` or whose
   outputs differ across batch sizes.
4. Writes per-trial JSON to `./runs/qwen3-8b-l1-slice/`.
5. On completion, prints the Pareto frontier over `(tokens_per_sec,
   tpot_p99_ms, peak_hbm_gb)`.

## What success looks like

This is iteration zero; success is substrate validation, not a hero
number. Accept the run if:

- Every harness component produced a trustworthy signal (trial JSONs
  have non-zero values, quality gate caught at least one injected
  canary if you add one, ledger frontier is non-degenerate).
- At least one config Pareto-dominates the vLLM defaults on at least
  one axis with no quality regression.

If neither holds, the blocker is the harness or the policy, not the
search — those are the signals we want early.

## Notes

- The policy here is `deterministic` (cycles vLLM defaults). Switch to
  `anthropic` or `openai_compatible` once you have API keys by
  editing `policy.warmstart.provider` and setting `api_key_env` to the
  env var holding your key. The deterministic mode is a valid real
  baseline — don't treat it as a placeholder.
- The config uses relative paths; run from the example directory.
- For a faster smoke run: `autoinfer run ./config.yaml --max-trials 3`.
