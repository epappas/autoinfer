#!/usr/bin/env python3
"""Autoinfer campaign runner.

Runs inside the Basilica deployment. Responsible for syncing extras,
fetching the trace, starting the reference replica on GPU 1, running
autoinfer on GPU 0, and summarizing the per-trial JSONs.

Separated from the bootstrap source because Basilica's deploy-time
validator rejects larger Python payloads, so the bootstrap stays tiny
and this full logic lives in-repo.

Usage (expected to run under uv in the cloned repo):

    uv run python scripts/campaign_runner.py \\
        --config examples/qwen3-8b-l1-slice/config.yaml \\
        --ref-port 8001 --max-trials 3
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


def log(msg: str) -> None:
    print("[campaign] " + msg, flush=True)


def wait_for_port(host: str, port: int, timeout_s: int) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            try:
                s.connect((host, port))
                return True
            except (OSError, TimeoutError):
                time.sleep(5.0)
    return False


def install_extras(workdir: Path) -> None:
    log("syncing extras: dev, vllm, llm, basilica")
    subprocess.run(
        [
            "uv", "sync",
            "--extra", "dev",
            "--extra", "vllm",
            "--extra", "llm",
            "--extra", "basilica",
        ],
        cwd=str(workdir), check=True,
    )


def prepare_data(workdir: Path) -> None:
    log("fetching sharegpt sample")
    subprocess.run(
        ["uv", "run", "python", "scripts/fetch_sharegpt.py", "--out-dir", "."],
        cwd=str(workdir), check=True,
    )


def start_reference(workdir: Path, model: str, port: int) -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"
    log_path = workdir / "runs" / "reference.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    out = log_path.open("wb")
    proc = subprocess.Popen(
        [
            "uv", "run", "vllm", "serve", model,
            "--port", str(port),
            "--dtype", "auto",
            "--gpu-memory-utilization", "0.85",
        ],
        env=env, cwd=str(workdir), stdout=out, stderr=subprocess.STDOUT,
    )
    log(f"reference replica pid={proc.pid} model={model} port={port}")
    if not wait_for_port("127.0.0.1", port, timeout_s=1200):
        proc.terminate()
        tail = log_path.read_text(errors="replace")[-2000:]
        raise RuntimeError(f"reference not ready in 20min\n{tail}")
    log("reference replica READY")
    return proc


def run_autoinfer(
    workdir: Path,
    config: str,
    max_trials: int | None,
    layer_trials: list[str],
) -> int:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    cmd = ["uv", "run", "autoinfer", "run", config]
    if max_trials is not None:
        cmd.extend(["--max-trials", str(max_trials)])
    for lt in layer_trials:
        cmd.extend(["--layer-trials", lt])
    log(f"running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, cwd=str(workdir))
    log(f"autoinfer exit code: {result.returncode}")
    return result.returncode


def summarize(workdir: Path) -> None:
    runs = workdir / "runs"
    if not runs.exists():
        log("no runs/ directory to summarize")
        return
    trial_files = sorted(runs.rglob("*.json"))
    log("--- SUMMARY ---")
    log(f"trial json files: {len(trial_files)}")
    for f in trial_files[:60]:
        try:
            payload = json.loads(f.read_text())
        except Exception:
            continue
        meas = "yes" if payload.get("measurement") else "no"
        fail = payload.get("failure")
        fail_kind = fail.get("kind") if isinstance(fail, dict) else "no"
        log(f"  {f.relative_to(runs)}: layer={payload.get('layer')} measurement={meas} failure={fail_kind}")
    log("--- END SUMMARY ---")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--ref-port", type=int, default=8001)
    p.add_argument("--max-trials", type=int, default=None)
    p.add_argument(
        "--layer-trials",
        action="append",
        default=[],
        help="Per-layer cap LAYER=N, repeatable; forwarded to `autoinfer run`.",
    )
    p.add_argument("--workdir", type=Path, default=Path.cwd())
    p.add_argument("--skip-install", action="store_true")
    p.add_argument("--skip-fetch", action="store_true")
    args = p.parse_args()

    workdir = args.workdir.resolve()
    log(f"workdir={workdir} python={sys.version.split()[0]}")

    if args.skip_install:
        log("skipping uv sync")
    else:
        install_extras(workdir)

    if args.skip_fetch:
        log("skipping trace fetch")
    else:
        prepare_data(workdir)

    ref_proc = start_reference(workdir, args.model, args.ref_port)
    try:
        code = run_autoinfer(workdir, args.config, args.max_trials, args.layer_trials)
        summarize(workdir)
    finally:
        log("terminating reference replica")
        try:
            ref_proc.terminate()
            ref_proc.wait(timeout=30)
        except Exception as e:
            log(f"reference teardown error: {e}")
    return code


if __name__ == "__main__":
    rc = main()
    print(f"campaign finished rc={rc}", flush=True)
    sys.exit(rc)
