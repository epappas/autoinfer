#!/usr/bin/env python3
"""Drive the autoinfer iteration-zero campaign on Basilica.

Creates ONE Basilica deployment with 2 GPUs that runs the entire
campaign: the FP16 reference replica on GPU 1 and the autoinfer
controller (which spawns candidate vLLM processes per trial) on GPU 0.
The deployment serves ``runs/`` over HTTP on the deployment's public URL
after the campaign completes so dev can fetch per-trial artifacts
without SSH.

Dev-side responsibilities:
- Create the deployment (this script).
- Stream deployment logs to stdout until the campaign finishes.
- On completion, download ``runs/<trial>.json`` files via HTTP.
- Delete the deployment.

Dry-run mode prints the deployment spec and the generated container
source, makes NO API calls, and exits 0. Use it to eyeball what the
campaign will do before spending GPU money.

Usage:
    export BASILICA_API_TOKEN="..."
    # optional but recommended so gated models can be fetched:
    export HF_TOKEN="..."

    # dry-run first (no API calls, no cost)
    uv run python scripts/orchestrate_iteration_zero.py --dry-run

    # smoke (3 trials)
    uv run python scripts/orchestrate_iteration_zero.py --max-trials 3

    # full iteration zero (40 trials, 8-12 hours)
    uv run python scripts/orchestrate_iteration_zero.py
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

from autoinfer.target.basilica import CampaignSpec

if TYPE_CHECKING:  # pragma: no cover
    from basilica import Deployment


LOG_POLL_S = 10.0
ARTIFACT_POLL_S = 30.0
CAMPAIGN_DONE_MARKER = "campaign finished rc="


def _require_token() -> None:
    if not os.environ.get("BASILICA_API_TOKEN"):
        print("ERROR: BASILICA_API_TOKEN not set", file=sys.stderr)
        sys.exit(2)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--name", default=f"autoinfer-iter0-{int(time.time())}")
    p.add_argument("--dry-run", action="store_true", help="Print plan, make no API calls.")
    p.add_argument("--max-trials", type=int, default=None)
    p.add_argument(
        "--layer-trials",
        action="append",
        default=[],
        help="Per-layer cap LAYER=N, repeatable; forwarded through to `autoinfer run`.",
    )
    p.add_argument("--config", default="examples/qwen3-8b-l1-slice/config.yaml")
    p.add_argument("--branch", default="main")
    p.add_argument("--repo", default="https://github.com/epappas/autoinfer.git")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--image", default="vllm/vllm-openai:latest")
    p.add_argument("--gpus", type=int, default=2)
    p.add_argument("--cpu", default="4", help="CPU allocation, e.g. '4' or '500m'.")
    p.add_argument("--memory", default="64Gi")
    p.add_argument("--min-gpu-memory-gb", type=int, default=40)
    p.add_argument(
        "--gpu-models",
        default=None,
        help=(
            "Comma-separated list of GPU model strings the campaign "
            "container must run on (e.g. 'H100' or 'A100,A100-80GB'). "
            "When set, Basilica's scheduler is constrained to these "
            "exact models — useful for hardware-class campaigns where "
            "the L3 kernels must run on a specific GPU."
        ),
    )
    p.add_argument(
        "--spot",
        choices=["auto", "true", "false"],
        default="auto",
        help=(
            "Whether to request spot (cheaper, lottery) or on-demand "
            "(pricier, reliable) instances. Default 'auto' lets "
            "Basilica's scheduler pick. Use 'false' to force on-demand "
            "when spot scheduling is unreliable (e.g. tight H100 spot "
            "pool); use 'true' to explicitly request spot."
        ),
    )
    p.add_argument("--ttl-hours", type=float, default=12.0)
    p.add_argument("--artifacts-dir", type=Path, default=Path("./basilica-artifacts"))
    p.add_argument("--log-file", type=Path, default=None)
    p.add_argument("--keep-after-done", action="store_true", help="Don't auto-delete on success.")
    p.add_argument("--retries", type=int, default=3, help="Retry count for transient deploy failures.")
    return p.parse_args()


def _build_spec(args: argparse.Namespace) -> CampaignSpec:
    spec = CampaignSpec(
        repo_url=args.repo,
        branch=args.branch,
        autoinfer_config=args.config,
        model=args.model,
        max_trials=args.max_trials,
        layer_trials=list(args.layer_trials),
        hf_token_env="HF_TOKEN" if os.environ.get("HF_TOKEN") else None,
    )
    # Pass through any LLM-provider API keys so the campaign's warmstart
    # and operator policies can authenticate from within the container.
    # BASILICA_API_TOKEN also passes through: the L2 adapter spawns per-trial
    # deployments via the SDK from inside the campaign container.
    for var in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "KIMI_API_KEY",
        "BASILICA_API_TOKEN",
    ):
        val = os.environ.get(var)
        if val:
            spec.env[var] = val
    return spec


def _print_plan(spec: CampaignSpec, kwargs: dict[str, Any]) -> None:
    print("=== deployment kwargs ===")
    printable = {k: v for k, v in kwargs.items() if k != "source"}
    print(json.dumps(printable, indent=2, default=str))
    print()
    print("=== container source (first 60 lines) ===")
    for i, line in enumerate(kwargs["source"].splitlines()[:60]):
        print(f"{i+1:3d}| {line}")
    print(f"... ({len(kwargs['source'].splitlines())} total lines)")


def _dedup_key(line: str) -> str:
    """Stable dedup key for a log line.

    Basilica's ``deployment.logs()`` returns each line wrapped as
    ``data: {"message": "...", "stream": "...", "timestamp": "..."}``
    where the timestamp is the orchestrator-side INGESTION time, not the
    container's print time. That timestamp ticks forward on every fetch,
    so a naive ``line.strip()`` key treats every re-served line as new.
    Strip the wrapper before deduping so we key on actual content.
    """
    s = line.strip()
    if s.startswith("data: ") and '"message":' in s:
        try:
            payload = json.loads(s[len("data: "):])
            msg = payload.get("message", "")
            stream = payload.get("stream", "")
            return f"{stream}|{msg}"
        except (json.JSONDecodeError, AttributeError):
            pass
    return s


def _stream_logs(
    deployment: Deployment,
    log_file: Path | None,
) -> bool:
    """Tail deployment logs until the campaign-done marker appears.

    Returns True if the done marker was seen, False on interruption /
    timeout of the tail loop itself (the deployment keeps running).
    """
    seen: set[str] = set()
    fh = log_file.open("a") if log_file else None
    try:
        while True:
            try:
                chunk = deployment.logs()
            except Exception as e:  # noqa: BLE001
                print(f"[orchestrator] logs() error: {e}", file=sys.stderr)
                time.sleep(LOG_POLL_S)
                continue
            for line in chunk.splitlines():
                if not line.strip():
                    continue
                key = _dedup_key(line)
                if key in seen:
                    continue
                seen.add(key)
                print(line)
                if fh:
                    fh.write(line + "\n")
                    fh.flush()
                if CAMPAIGN_DONE_MARKER in line:
                    print("[orchestrator] campaign completion marker seen")
                    return True
            time.sleep(LOG_POLL_S)
    finally:
        if fh:
            fh.close()


def _fetch_artifacts(deployment: Deployment, out_dir: Path) -> None:
    """Fetch per-trial JSON artifacts served by the deployment on its HTTP port."""
    out_dir.mkdir(parents=True, exist_ok=True)
    base_url = deployment.url.rstrip("/")
    index_url = f"{base_url}/"
    print(f"[orchestrator] listing {index_url}")
    with urllib.request.urlopen(index_url, timeout=30) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    # tolerant parser: pick hrefs ending in .json / .jsonl / .tsv / .log
    import re

    pattern = re.compile(r'href="([^"]+\.(?:jsonl|json|tsv|log|out|err))"')
    candidates = set(pattern.findall(html))
    print(f"[orchestrator] {len(candidates)} artifacts")
    for rel in sorted(candidates):
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        url = f"{base_url}/{rel.lstrip('/')}"
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                dst.write_bytes(r.read())
            print(f"  fetched {rel}")
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED {rel}: {e}")


def _summarize_artifacts(artifacts_dir: Path) -> None:
    # only count trial files (skip events.jsonl, results.tsv, run_summary.json,
    # hw_context.json, *_bench.json which are auxiliary).
    files = [
        p for p in sorted(artifacts_dir.rglob("*.json"))
        if not p.name.endswith("_bench.json")
        and p.name not in ("run_summary.json", "hw_context.json")
    ]
    kept = 0
    failed = 0
    for f in files:
        try:
            payload = json.loads(f.read_text())
        except Exception:
            continue
        if payload.get("measurement"):
            kept += 1
        elif payload.get("failure"):
            failed += 1
    print(f"[orchestrator] {len(files)} trials: {kept} with measurements, {failed} failures")


def _extract_instance_name(exc: Exception) -> str | None:
    """Pull the Basilica instance name out of a DeploymentFailed message."""
    msg = str(exc)
    import re
    m = re.search(r"'([0-9a-f-]{36})'", msg)
    return m.group(1) if m else None


def _create_with_retry(client: Any, base_kwargs: dict[str, Any], retries: int) -> Deployment:
    """Wrap client.deploy in a retry loop — Basilica spot scheduling is flaky.

    On DeploymentFailed, extract the leaked instance_name and delete it,
    then issue a fresh ``name`` for the next attempt so Basilica treats
    it as a new deployment rather than reusing the failed instance.
    """
    import basilica.exceptions as bexc

    last_exc: Exception | None = None
    for attempt in range(1, retries + 2):
        kwargs = dict(base_kwargs)
        kwargs["name"] = f"{base_kwargs['name']}-t{attempt}"
        print(f"[orchestrator] deploy attempt {attempt}/{retries + 1} name={kwargs['name']}")
        try:
            return client.deploy(**kwargs)
        except bexc.DeploymentFailed as e:
            last_exc = e
            iname = _extract_instance_name(e)
            print(f"[orchestrator] attempt {attempt} failed: {iname or '(unknown)'}")
            if iname:
                try:
                    client.delete_deployment(iname)
                    print(f"[orchestrator] cleaned up {iname}")
                except Exception as de:  # noqa: BLE001
                    print(f"[orchestrator] cleanup {iname} failed: {de}")
            if attempt <= retries:
                time.sleep(20.0)
    raise last_exc if last_exc else RuntimeError("retry loop exited without exception")


def main() -> int:
    args = _parse_args()
    spec = _build_spec(args)
    ttl_seconds = int(args.ttl_hours * 3600)
    gpu_models = (
        [m.strip() for m in args.gpu_models.split(",") if m.strip()]
        if args.gpu_models
        else None
    )
    spot: bool | None
    if args.spot == "auto":
        spot = None
    elif args.spot == "true":
        spot = True
    else:
        spot = False
    kwargs = spec.build_deploy_kwargs(
        name=args.name,
        image=args.image,
        gpu_count=args.gpus,
        cpu=args.cpu,
        memory=args.memory,
        storage=True,
        ttl_seconds=ttl_seconds,
        timeout=1800,
        min_gpu_memory_gb=args.min_gpu_memory_gb,
        gpu_models=gpu_models,
        spot=spot,
    )

    if args.dry_run:
        _print_plan(spec, kwargs)
        print("\n[orchestrator] dry-run: no API calls made.")
        return 0

    _require_token()
    import basilica

    client = basilica.BasilicaClient()

    print(f"[orchestrator] creating deployment: {args.name}")
    print(f"[orchestrator] image={args.image} gpus={args.gpus} memory={args.memory} ttl={ttl_seconds}s")

    try:
        deployment = _create_with_retry(client, kwargs, retries=args.retries)
    except Exception as e:  # noqa: BLE001
        print(f"[orchestrator] all deploy attempts failed: {e}", file=sys.stderr)
        return 3
    print(f"[orchestrator] deployment.url: {deployment.url}")

    signal.signal(signal.SIGINT, lambda *_: _cleanup_and_exit(deployment, args))

    print("[orchestrator] deployment ready, tailing logs...")
    done = _stream_logs(deployment, args.log_file)
    if not done:
        print("[orchestrator] did not observe completion marker; skipping artifact fetch.")
    else:
        # give the in-container HTTP server a moment to be ready
        time.sleep(10.0)
        try:
            _fetch_artifacts(deployment, args.artifacts_dir)
            _summarize_artifacts(args.artifacts_dir)
        except Exception as e:  # noqa: BLE001
            print(f"[orchestrator] artifact fetch failed: {e}", file=sys.stderr)

    if args.keep_after_done:
        print(f"[orchestrator] keeping deployment alive; delete manually: "
              f"basilica deploy delete {args.name}")
    else:
        print(f"[orchestrator] deleting deployment: {args.name}")
        deployment.delete()
    return 0


def _cleanup_and_exit(deployment: Deployment, args: argparse.Namespace) -> None:
    print("\n[orchestrator] SIGINT — cleaning up")
    try:
        if not args.keep_after_done:
            deployment.delete()
    finally:
        sys.exit(130)


if __name__ == "__main__":
    sys.exit(main())
