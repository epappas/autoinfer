# Iteration-zero runbook — Qwen3-8B L1 slice on Basilica (SDK)

First hardware run for autoinfer. Cheapest first, halt on the first
surprise. Goal is **substrate validation**, not a hero number — success
is every component producing a trustworthy signal (thesis §8).

Budget guide: ~$5–10 for the smoke (3 trials), ~$40–80 for the full
40-trial run. Depends on Basilica pricing at time of run.

## Prereqs

- `BASILICA_API_TOKEN` exported (required).
- `HF_TOKEN` exported if you want gated HF models to just work (optional
  but recommended — the orchestrator forwards it to the deployment).
- `uv` on dev machine (installed once).
- Nothing else on dev: the orchestrator installs the Basilica SDK
  locally via `autoinfer[basilica]` and makes only HTTP calls. All
  heavy work (vLLM install, model download, benchmark) happens on
  Basilica.

## 0. Sanity — dev-machine checks (fast, no cost)

```bash
git clone git@github.com:epappas/autoinfer.git && cd autoinfer
uv sync --extra dev --extra basilica
uv run pytest -q                                               # 138 tests, ~1.5s
uv run autoinfer validate examples/qwen3-8b-l1-slice/config.yaml
```

## 1. Dry-run the orchestrator (no API calls, no cost)

```bash
uv run python scripts/orchestrate_iteration_zero.py --dry-run
uv run python scripts/orchestrate_iteration_zero.py --dry-run --max-trials 3
```

Prints the `BasilicaClient.deploy` kwargs it would use and the first
60 lines of the container source. Confirm the deployment shape (image,
gpu_count, memory, ttl) and the campaign config path look right before
spending money.

## 2. Smoke run — 3 trials on Basilica

```bash
export BASILICA_API_TOKEN="..."
export HF_TOKEN="..."   # optional

uv run python scripts/orchestrate_iteration_zero.py \
    --max-trials 3 \
    --artifacts-dir ./basilica-artifacts/smoke \
    --log-file ./basilica-artifacts/smoke/deployment.log
```

What happens:

1. Orchestrator (on dev) calls `BasilicaClient.deploy(...)` with a
   2-GPU, 128 GiB deployment. Returns a `Deployment` with a public URL.
2. Inside the deployment, the generated `campaign` source runs:
   `install_uv → clone autoinfer → uv sync --extra vllm →
   fetch_sharegpt → start_reference (GPU 1) → autoinfer run
   --max-trials 3 (GPU 0) → summarize → serve_artifacts_forever`.
3. Orchestrator (dev) tails `deployment.logs()` until it sees
   `campaign finished rc=...` (the completion marker printed by the
   campaign).
4. Orchestrator HTTP-fetches every `*.json` under the deployment's
   artifacts server (port 9000, exposed at `deployment.url`) into
   `./basilica-artifacts/smoke/`.
5. Orchestrator deletes the deployment.

Expected walltime ~1–2 hours (most of it is the one-time vLLM install
and Qwen3-8B model download inside the container; actual trial work
is ~15 min × 3).

Success criteria:

- `deployment.log` ends with `campaign finished rc=0`.
- `./basilica-artifacts/smoke/` contains per-trial JSONs.
- At least one trial has a `measurement`; if any have `failure`, the
  `failure.kind` is a real typed failure (OOM, HANG, QUALITY_KL, etc.),
  not UNKNOWN.

Halt if:

- `wait_until_ready` times out (deployment never becomes ready).
- Logs show `install_deps` or `prepare_data` failing repeatedly.
- `campaign finished rc=` never appears in logs within the TTL.

## 3. Canary — single trial, constraint-violation check

Same orchestrator with the canary config forces the L1 adapter's
fail-fast path (compat-rule rejection, no vLLM subprocess spawned):

```bash
uv run python scripts/orchestrate_iteration_zero.py \
    --config scripts/canary_config.yaml \
    --max-trials 1 \
    --artifacts-dir ./basilica-artifacts/canary
```

Expected artifact: one JSON with `failure.kind == "startup"` and
`measurement: null`.

Walltime ~45 min (dominated by vLLM install + reference replica
warmup — the trial itself is instant because it's rejected pre-spawn).

## 4. Full iteration zero — 40 trials

```bash
uv run python scripts/orchestrate_iteration_zero.py \
    --artifacts-dir ./basilica-artifacts/full \
    --log-file ./basilica-artifacts/full/deployment.log \
    --ttl-hours 16
```

Expected walltime ~10–14 hours.

## 5. Interpretation

Once artifacts are on dev:

```bash
find ./basilica-artifacts -name "*.json" | wc -l
jq -s 'map(select(.measurement)) | length' ./basilica-artifacts/full/*.json
jq -s 'map(select(.failure)) | group_by(.failure.kind)' ./basilica-artifacts/full/*.json
```

Post-run, paste into the pair session:

- `deployment.log` tail (last 200 lines).
- The artifact summary above.
- Any JSON where `measurement` is non-null and ranks highly on
  `tokens_per_sec` or `tpot_p99_ms`.

With that evidence, we evaluate whether the thesis survives.

## Success definition (thesis §8)

Primary (substrate validation):

- Every ledger entry has either a Measurement or a typed FailureRecord.
- The quality gate rejected at least one config (step 3 canary OR a
  natural fail during steps 2/4).
- Pareto frontier is non-empty and non-degenerate.

Secondary (bonus):

- At least one config Pareto-dominates the vLLM defaults on at least
  one axis without quality regression.

A secondary win without a primary win is a failure, not a success:
we got a number we cannot trust.

## Cleanup

If the orchestrator crashes mid-run or you Ctrl-C, verify the
deployment is gone:

```bash
uv run python -c "
import basilica
c = basilica.BasilicaClient()
for d in c.list_deployments().deployments:
    if d.name.startswith('autoinfer-iter0'):
        print(d.name, d.state)
"
```

Manually delete any stragglers (they cost until TTL or delete):

```bash
uv run python -c "
import basilica
c = basilica.BasilicaClient()
c.delete_deployment('autoinfer-iter0-<timestamp>')
"
```
