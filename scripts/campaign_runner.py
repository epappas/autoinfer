#!/usr/bin/env python3
"""Autoinfer campaign runner.

Runs inside the Basilica deployment. Responsible for syncing extras,
fetching the trace, starting the reference replica, running
autoinfer, and summarizing the per-trial JSONs.

GPU placement is auto-detected:
- 2+ GPUs: reference replica on GPU 1, autoinfer candidates on GPU 0
  (independent — no contention; matches C01/C02 setup)
- 1 GPU: reference replica and candidates time-share GPU 0. Reference
  is constrained to ``--gpu-memory-utilization 0.40`` so the L1
  candidate vLLMs (which spawn separately and use 0.85-0.92 of
  remaining HBM by default) still fit. The reference's bench load is
  ~16 GB for Qwen3-8B at bf16 (40% of an 80 GB A100 = 32 GB headroom),
  candidates fit in the remainder.

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


def detect_gpu_count() -> int:
    """Return the number of GPUs visible to the container.

    Tries ``nvidia-smi -L`` first (most reliable; doesn't require any
    Python CUDA stack). Falls back to torch on failure. Returns 0 if
    neither path can detect any GPUs.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            # One non-empty line per GPU.
            lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
            return len(lines)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:  # noqa: BLE001
        return 0


def start_reference(
    workdir: Path, model: str, port: int, *, gpu_count: int
) -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    if gpu_count >= 2:
        # 2+ GPUs: reference on GPU 1, candidates on GPU 0 (independent).
        env["CUDA_VISIBLE_DEVICES"] = "1"
        gpu_mem_util = "0.85"
    else:
        # Single GPU: reference and candidates time-share index 0.
        # Reference takes 40% so each L1 candidate (default
        # gpu_memory_utilization 0.85-0.92) still fits in the remainder.
        env["CUDA_VISIBLE_DEVICES"] = "0"
        gpu_mem_util = "0.40"
    log(
        f"reference replica placement: gpu_count={gpu_count} "
        f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} "
        f"gpu_memory_utilization={gpu_mem_util}"
    )
    log_path = workdir / "runs" / "reference.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    out = log_path.open("wb")
    proc = subprocess.Popen(
        [
            "uv", "run", "vllm", "serve", model,
            "--port", str(port),
            "--dtype", "auto",
            "--gpu-memory-utilization", gpu_mem_util,
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
    *,
    gpu_count: int,
) -> int:
    env = os.environ.copy()
    # Candidates always run on GPU 0. With 2+ GPUs, the reference is on
    # GPU 1 (no contention). With 1 GPU, reference shares GPU 0 with
    # candidates — start_reference reserved 40% of HBM for the reference,
    # so candidates must cap their gpu_memory_utilization to fit in the
    # remaining 60%. vLLM treats --gpu-memory-utilization as % of TOTAL
    # GPU memory (not remainder), so the surrogate's default sweep
    # [0.80, 0.95] OOMs in 1-GPU mode without the cap.
    env["CUDA_VISIBLE_DEVICES"] = "0"
    if gpu_count == 1:
        # 0.55 = 60% of GPU minus a small safety margin (vLLM's own
        # auto-grow + cudagraph capture) so the candidate doesn't trip
        # the OOM watchdog when the reference is concurrently serving
        # gate prompts.
        env["AUTOINFER_L1_GMU_MAX"] = "0.55"
        log("AUTOINFER_L1_GMU_MAX=0.55 (1-GPU mode; candidate gmu clamped)")
    cmd = ["uv", "run", "autoinfer", "run", config]
    if max_trials is not None:
        cmd.extend(["--max-trials", str(max_trials)])
    for lt in layer_trials:
        cmd.extend(["--layer-trials", lt])
    log(f"candidate placement: gpu_count={gpu_count} CUDA_VISIBLE_DEVICES=0")
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

    gpu_count = detect_gpu_count()
    log(f"detected gpu_count={gpu_count}")
    if gpu_count == 0:
        log("ERROR: no GPUs detected — campaign cannot run")
        return 2
    ref_proc = start_reference(
        workdir, args.model, args.ref_port, gpu_count=gpu_count,
    )
    try:
        code = run_autoinfer(
            workdir, args.config, args.max_trials, args.layer_trials,
            gpu_count=gpu_count,
        )
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
