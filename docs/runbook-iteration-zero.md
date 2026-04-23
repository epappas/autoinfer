# Iteration-zero runbook — Qwen3-8B L1 slice on Basilica

First hardware run for autoinfer. Seven steps, cheapest first, halt on the
first surprise. The goal is **substrate validation**, not a hero number —
success is every component producing a trustworthy signal (see thesis §8).

Budget guide: ~$5–10 for the smoke (steps 0–6), ~$40–80 for the full
40-trial run (step 7). Numbers depend on Basilica's current GPU pricing.

## Prereqs

- Basilica account with API token exported: `export BASILICA_API_TOKEN=...`
- `basilica` CLI installed and authenticated.
- SSH access configured for Basilica deployments.
- Two GPUs on a single deployment (one candidate, one reference).
  Reference: H100-80GB recommended (Qwen3-8B FP16 ≈ 16 GB weights +
  KV headroom). Candidate: same or comparable.

## 0. Sanity: config validates locally

```bash
git clone git@github.com:epappas/autoinfer.git && cd autoinfer
uv sync --extra dev
uv run autoinfer validate examples/qwen3-8b-l1-slice/config.yaml
uv run pytest -q  # 126+ tests, no GPU required
```

`validate` should echo `ok: qwen3-8b-l1-slice-iteration-zero`. If it
fails, halt and fix locally before paying for GPU time.

## 1. Create the Basilica deployment

Allocate two GPUs on one node with enough RAM and storage for Qwen3-8B +
ShareGPT + run artifacts. Example via `basilica` CLI:

```bash
basilica deploy create \
  --name autoinfer-iter0 \
  --image pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel \
  --gpus 2 \
  --memory 128Gi \
  --cpu 16 \
  --storage 200Gi \
  --ttl 43200
```

Capture the deployment ID; you'll use it in subsequent `basilica deploy
exec` / `basilica deploy ssh` commands. Keep the deployment alive until
step 7 completes. Destroy it with `basilica deploy destroy <id>` when
done.

## 2. Bootstrap inside the deployment

SSH into the node (or use `basilica deploy exec`) and set up:

```bash
# inside the deployment
cd /workspace
apt-get update && apt-get install -y curl git
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

git clone https://github.com/epappas/autoinfer.git
cd autoinfer
uv sync --extra dev --extra vllm

python scripts/fetch_sharegpt.py --out-dir . --seed 0
```

`fetch_sharegpt.py` downloads ShareGPT (~1 GB), samples 500 trace
prompts and 500 held-out quality prompts, writes both as JSONL. The
script prints sha256 digests — record them if you want reproducibility
guarantees later.

## 3. Start the reference replica (GPU 1)

```bash
bash scripts/start_reference.sh Qwen/Qwen3-8B 8001 1
```

The script spawns `vllm serve` on GPU 1, polls `GET /v1/models` until
200, and exits 0 when ready. Logs land in `./runs/reference.log`; PID
in `./runs/reference.pid`. Leave the replica running for the rest of
the run.

Sanity check from the same node:

```bash
curl http://127.0.0.1:8001/v1/models | head
curl http://127.0.0.1:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-8B", "prompt": "Hello,", "max_tokens": 8, "logprobs": 5}'
```

If the logprobs call succeeds, the quality gate has a live reference.

## 4. One manual candidate trial (GPU 0)

Before letting autoinfer drive, spawn one candidate by hand on GPU 0
with a known-good config and hit it with `vllm bench serve`:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-8B \
  --port 8000 --dtype auto --gpu-memory-utilization 0.90 \
  --max-num-seqs 128 &

# wait for ready
until curl -s http://127.0.0.1:8000/v1/models | grep -q '"object"'; do sleep 5; done

uv run vllm bench serve \
  --backend openai-chat \
  --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen3-8B \
  --dataset-name custom \
  --dataset-path ./trace/sharegpt_sample.jsonl \
  --num-prompts 50 \
  --save-result \
  --result-filename manual_probe.json \
  --result-dir ./runs/probe

pkill -f "port 8000"
```

Check `./runs/probe/manual_probe.json` — it should have
`output_throughput`, `median_ttft_ms`, `p99_tpot_ms`, etc. If these are
missing or zero, autoinfer's `parse_bench_output` needs a schema
update before step 5.

## 5. Canary — verify the fail-fast path

The compat rule `kv_fp8_requires_compatible_backend` rejects
`kv_cache_dtype=fp8` with `attention_backend=XFORMERS`. The seed config
in `scripts/canary_config.yaml` hits exactly that combination:

```bash
# adjust the relative knobs path if needed (see the yaml comment)
uv run autoinfer run scripts/canary_config.yaml
```

Expected: one trial recorded with `failure.kind == "startup"`, no
`measurement`, empty Pareto frontier. The ledger writes to
`./runs/canary/`. If autoinfer *doesn't* reject the config, the
compat-rule pipeline is broken — halt and debug (no GPU time was
spent on this anyway, the rejection happens pre-subprocess).

## 6. 3-trial smoke run

```bash
bash scripts/smoke_run.sh
```

Drives the main example config with `--max-trials 3` on GPU 0 against
the reference on GPU 1. Walltime ~30–60 min depending on trace length.

Exit criteria:

- Ledger JSONs in `./runs/qwen3-8b-l1-slice/*.json` — at least one
  measurement, at least one Pareto entry.
- No uncaught exceptions in `./runs/smoke_run.log`.
- Reference replica still up (`curl /v1/models` on 8001).

If this works, the substrate is validated end-to-end. Halt here if
any of the exit criteria fail — debug with a smaller `--max-trials 1`
and longer logs before step 7.

## 7. Full 40-trial iteration zero

```bash
CUDA_VISIBLE_DEVICES=0 uv run autoinfer run \
  examples/qwen3-8b-l1-slice/config.yaml 2>&1 | tee ./runs/full_run.log
```

Expected walltime on a single H100 candidate: ~8–12 hours for 40
trials (1 config spawns vLLM ≈ 30 s, trace replay ≈ 10–15 min, gate
≈ 1–2 min, teardown ≈ 5 s).

## 8. Retrieve artifacts and destroy the deployment

```bash
# from your dev machine
basilica deploy exec autoinfer-iter0 -- tar czf /tmp/runs.tar.gz -C /workspace/autoinfer ./runs
basilica deploy download autoinfer-iter0 /tmp/runs.tar.gz ./runs-iter0.tar.gz
basilica deploy destroy autoinfer-iter0
```

Paste the full-run Pareto-frontier output and the contents of
`./runs/qwen3-8b-l1-slice/*.json` summary into our next pair session.
With that evidence, we can say whether the thesis survives or has to
be updated.

## Halt criteria

Stop and debug at the first of these:

- Step 0: tests fail locally. Never burn GPU time on a red tree.
- Step 2: `uv sync --extra vllm` fails. vLLM has CUDA/driver
  requirements; fix before proceeding.
- Step 3: reference replica doesn't serve after 10 min. Check
  `./runs/reference.log` for CUDA/model-download errors.
- Step 4: manual candidate trial produces zero throughput. Something
  is wrong with the Docker image's GPU access or vLLM install.
- Step 5: canary isn't rejected. Compat-rule pipeline broken.
- Step 6: smoke fails. Fix before spending 10× on step 7.

## What "success" means at iteration zero

Primary (substrate validation, per thesis §8):

- Every ledger entry carries either a Measurement or a typed
  FailureRecord.
- The quality gate rejected at least one config (step 5 canary OR a
  natural fail during steps 6–7).
- The Pareto frontier is non-empty and non-degenerate.

Secondary (bonus):

- At least one config Pareto-dominates the vLLM defaults on at least
  one axis without quality regression.

A secondary win without a primary win — e.g., we got a number but
couldn't trust the harness — is a failure, not a success.
