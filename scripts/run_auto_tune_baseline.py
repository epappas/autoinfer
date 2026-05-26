#!/usr/bin/env python3
"""Drive the vLLM ``auto_tune.sh`` baseline (T-37) on Basilica.

Creates a single-GPU Basilica deployment via the SDK, runs
``benchmarks/auto_tune/auto_tune.sh`` from the matching vllm tag with
the C04-recon-recommended env-var block (Llama-3.1-8B / synthetic
random / E2E SLO < 500ms), tails container logs until the
campaign-done marker, downloads ``result.txt`` + per-cell logs via the
built-in HTTP server, deletes the deployment, and writes a baseline-
artifact directory the C04 pre-reg can cite.

Dev-side responsibilities mirror ``orchestrate_iteration_zero.py``:
- Create the deployment (this script, via the Basilica SDK).
- Stream deployment logs to stdout until the campaign finishes.
- On completion, download artifacts via HTTP.
- Delete the deployment.

Dry-run mode prints the deployment spec and the generated container
source, makes NO API calls, and exits 0. Use it to eyeball what the
deployment will do before spending GPU money.

Usage:
    export BASILICA_API_TOKEN="..."
    # HF_TOKEN is REQUIRED — Llama-3.1-8B-Instruct is gated by Meta.
    export HF_TOKEN="..."

    # dry-run first (no API calls, no cost)
    uv run python scripts/run_auto_tune_baseline.py --dry-run

    # actual launch on 1x A100 spot (~$0.75-1.50 expected)
    uv run python scripts/run_auto_tune_baseline.py \\
        --gpu-models A100 --spot true --yes
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

from autoinfer.target.auto_tune import AutoTuneBaselineSpec

if TYPE_CHECKING:  # pragma: no cover
    from basilica import Deployment


LOG_POLL_S = 10.0
CAMPAIGN_DONE_MARKER = "campaign finished rc="


def _warn_if_no_basilica_token() -> None:
    """``BasilicaClient()`` auto-loads from ``~/.basilica/.env`` (the
    config file ``basilica login`` writes) when the env-var is absent,
    so this is a soft warning rather than a hard exit — the SDK will
    surface the real auth error itself if neither path works."""
    if not os.environ.get("BASILICA_API_TOKEN"):
        print(
            "[orchestrator] note: BASILICA_API_TOKEN not in env; "
            "SDK will fall back to ~/.basilica/.env (if present).",
            file=sys.stderr,
        )


def _require_hf_token() -> None:
    """Hard exit: the gated Llama-3.1-8B-Instruct download requires HF
    auth. The orchestrator must read ``HF_TOKEN`` from its own env so
    it can be passed *through* to the deployment container (which has
    no HuggingFace config file of its own)."""
    if not os.environ.get("HF_TOKEN"):
        print(
            "ERROR: HF_TOKEN not set — Llama-3.1-8B-Instruct is gated by "
            "Meta and the deployment will fail at model-download without it.\n"
            "Export the token before re-running, e.g.:\n"
            "    export HF_TOKEN=hf_xxx",
            file=sys.stderr,
        )
        sys.exit(2)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--name", default=f"autoinfer-auto-tune-{int(time.time())}")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan + container source; no API calls.",
    )
    p.add_argument("--yes", action="store_true", help="Confirm paid deployment.")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--vllm-version", default="0.21.0")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--input-len", type=int, default=1800)
    p.add_argument("--output-len", type=int, default=20)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--max-latency-allowed-ms", type=int, default=500)
    p.add_argument("--min-cache-hit-pct", type=int, default=0)
    p.add_argument("--num-seqs-list", default="128 256")
    p.add_argument("--num-batched-tokens-list", default="512 1024 2048 4096")
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--min-gpu-memory-gb", type=int, default=40)
    p.add_argument(
        "--gpu-models",
        default=None,
        help="Comma-separated GPU model strings (e.g. 'A100' or 'H100').",
    )
    p.add_argument(
        "--spot",
        choices=["auto", "true", "false"],
        default="auto",
    )
    p.add_argument("--cpu", default="4")
    p.add_argument("--memory", default="64Gi")
    p.add_argument("--ttl-hours", type=float, default=4.0)
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("./basilica-artifacts"),
    )
    p.add_argument("--log-file", type=Path, default=None)
    p.add_argument("--keep-after-done", action="store_true")
    p.add_argument("--retries", type=int, default=3)
    return p.parse_args()


def _build_spec(args: argparse.Namespace) -> AutoTuneBaselineSpec:
    return AutoTuneBaselineSpec(
        model=args.model,
        vllm_version=args.vllm_version,
        tp=args.tp,
        input_len=args.input_len,
        output_len=args.output_len,
        max_model_len=args.max_model_len,
        max_latency_allowed_ms=args.max_latency_allowed_ms,
        min_cache_hit_pct=args.min_cache_hit_pct,
        num_seqs_list=args.num_seqs_list,
        num_batched_tokens_list=args.num_batched_tokens_list,
        hf_token_env="HF_TOKEN" if os.environ.get("HF_TOKEN") else None,
    )


def _print_plan(spec: AutoTuneBaselineSpec, kwargs: dict[str, Any]) -> None:
    print("=== auto_tune.sh baseline plan ===")
    print(f"  model:           {spec.model}")
    print(f"  vllm version:    {spec.vllm_version}")
    print(f"  TP:              {spec.tp}")
    print(f"  INPUT_LEN:       {spec.input_len}")
    print(f"  OUTPUT_LEN:      {spec.output_len}")
    print(f"  MAX_MODEL_LEN:   {spec.max_model_len}")
    print(f"  MAX_LATENCY_MS:  {spec.max_latency_allowed_ms}")
    print(f"  MIN_CACHE_HIT:   {spec.min_cache_hit_pct}")
    print(f"  NUM_SEQS_LIST:   {spec.num_seqs_list}")
    print(f"  NUM_BATCHED:     {spec.num_batched_tokens_list}")
    print()
    print("=== deployment kwargs ===")
    printable = {k: v for k, v in kwargs.items() if k != "source"}
    print(json.dumps(printable, indent=2, default=str))
    print()
    print("=== container source (first 60 lines) ===")
    for i, line in enumerate(kwargs["source"].splitlines()[:60]):
        print(f"{i+1:3d}| {line}")
    print(f"... ({len(kwargs['source'].splitlines())} total lines)")


def _dedup_key(line: str) -> str:
    """Stable dedup key — same logic as orchestrate_iteration_zero."""
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


def _stream_logs(deployment: Deployment, log_file: Path | None) -> bool:
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
                    print("[orchestrator] auto_tune completion marker seen")
                    return True
            time.sleep(LOG_POLL_S)
    finally:
        if fh:
            fh.close()


def _fetch_artifacts(deployment: Deployment, out_dir: Path) -> None:
    """Fetch the auto-benchmark/<TAG>/ tree from the in-container server."""
    out_dir.mkdir(parents=True, exist_ok=True)
    base_url = deployment.url.rstrip("/")
    index_url = f"{base_url}/"
    print(f"[orchestrator] listing {index_url}")
    with urllib.request.urlopen(index_url, timeout=30) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    pattern = re.compile(r'href="([^"]+)"')
    candidates = sorted(set(pattern.findall(html)))
    print(f"[orchestrator] {len(candidates)} artifacts")
    for rel in candidates:
        if rel in ("/", ""):
            continue
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        url = f"{base_url}/{rel.lstrip('/')}"
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                dst.write_bytes(r.read())
            print(f"  fetched {rel}")
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED {rel}: {e}")


def _summarise_artifacts(artifacts_dir: Path) -> None:
    result_txt_candidates = list(artifacts_dir.rglob("result.txt"))
    print(f"[orchestrator] {len(result_txt_candidates)} result.txt file(s) found")
    for rt in result_txt_candidates:
        print(f"--- {rt} ---")
        print(rt.read_text())


def _extract_instance_name(exc: Exception) -> str | None:
    msg = str(exc)
    m = re.search(r"'([0-9a-f-]{36})'", msg)
    return m.group(1) if m else None


def _create_with_retry(
    client: Any, base_kwargs: dict[str, Any], retries: int
) -> Deployment:
    import basilica.exceptions as bexc

    last_exc: Exception | None = None
    for attempt in range(1, retries + 2):
        kwargs = dict(base_kwargs)
        kwargs["name"] = f"{base_kwargs['name']}-t{attempt}"
        print(
            f"[orchestrator] deploy attempt {attempt}/{retries + 1} "
            f"name={kwargs['name']}"
        )
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


def _cleanup_and_exit(deployment: Deployment, args: argparse.Namespace) -> None:
    print("\n[orchestrator] SIGINT — cleaning up")
    try:
        if not args.keep_after_done:
            deployment.delete()
    finally:
        sys.exit(130)


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

    if not args.yes:
        _print_plan(spec, kwargs)
        print(
            "\n[orchestrator] re-run with --yes to actually launch. "
            "This creates a paid Basilica deployment."
        )
        return 0

    _warn_if_no_basilica_token()
    _require_hf_token()
    import basilica

    client = basilica.BasilicaClient()
    print(f"[orchestrator] creating deployment: {args.name}")

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
        print(
            "[orchestrator] did not observe completion marker; "
            "skipping artifact fetch."
        )
    else:
        time.sleep(10.0)
        try:
            _fetch_artifacts(deployment, args.artifacts_dir)
            _summarise_artifacts(args.artifacts_dir)
        except Exception as e:  # noqa: BLE001
            print(f"[orchestrator] artifact fetch failed: {e}", file=sys.stderr)

    if args.keep_after_done:
        print(
            "[orchestrator] keeping deployment alive; "
            "delete manually with the SDK or via the Basilica console."
        )
    else:
        print(f"[orchestrator] deleting deployment: {args.name}")
        deployment.delete()
    return 0


if __name__ == "__main__":
    sys.exit(main())
