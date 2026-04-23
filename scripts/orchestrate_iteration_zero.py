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
    p.add_argument("--config", default="examples/qwen3-8b-l1-slice/config.yaml")
    p.add_argument("--branch", default="main")
    p.add_argument("--repo", default="https://github.com/epappas/autoinfer.git")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--image", default="pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel")
    p.add_argument("--gpus", type=int, default=2)
    p.add_argument("--memory", default="128Gi")
    p.add_argument("--min-gpu-memory-gb", type=int, default=40)
    p.add_argument("--ttl-hours", type=float, default=12.0)
    p.add_argument("--artifacts-dir", type=Path, default=Path("./basilica-artifacts"))
    p.add_argument("--log-file", type=Path, default=None)
    p.add_argument("--keep-after-done", action="store_true", help="Don't auto-delete on success.")
    return p.parse_args()


def _build_spec(args: argparse.Namespace) -> CampaignSpec:
    return CampaignSpec(
        repo_url=args.repo,
        branch=args.branch,
        autoinfer_config=args.config,
        model=args.model,
        max_trials=args.max_trials,
        hf_token_env="HF_TOKEN" if os.environ.get("HF_TOKEN") else None,
    )


def _print_plan(spec: CampaignSpec, kwargs: dict[str, Any]) -> None:
    print("=== deployment kwargs ===")
    printable = {k: v for k, v in kwargs.items() if k != "source"}
    print(json.dumps(printable, indent=2, default=str))
    print()
    print("=== container source (first 60 lines) ===")
    for i, line in enumerate(kwargs["source"].splitlines()[:60]):
        print(f"{i+1:3d}| {line}")
    print(f"... ({len(kwargs['source'].splitlines())} total lines)")


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
                digest = line.strip()
                if not digest or digest in seen:
                    continue
                seen.add(digest)
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
    # very small, tolerant parser: pick hrefs ending in .json
    import re

    candidates = set(re.findall(r'href="([^"]+\.json)"', html))
    print(f"[orchestrator] {len(candidates)} json artifacts")
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
    files = sorted(artifacts_dir.rglob("*.json"))
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
    print(f"[orchestrator] {len(files)} artifacts: {kept} with measurements, {failed} failures")


def main() -> int:
    args = _parse_args()
    spec = _build_spec(args)
    ttl_seconds = int(args.ttl_hours * 3600)
    kwargs = spec.build_deploy_kwargs(
        name=args.name,
        image=args.image,
        gpu_count=args.gpus,
        memory=args.memory,
        storage=True,
        ttl_seconds=ttl_seconds,
        timeout=1800,
        min_gpu_memory_gb=args.min_gpu_memory_gb,
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
    deployment = client.deploy(**kwargs)
    print(f"[orchestrator] deployment.url: {deployment.url}")

    signal.signal(signal.SIGINT, lambda *_: _cleanup_and_exit(deployment, args))

    try:
        deployment.wait_until_ready(timeout=kwargs["timeout"])
    except Exception as e:  # noqa: BLE001
        print(f"[orchestrator] wait_until_ready failed: {e}", file=sys.stderr)
        if not args.keep_after_done:
            deployment.delete()
        return 3

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
