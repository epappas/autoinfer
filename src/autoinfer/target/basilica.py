"""Basilica deployment helper.

One deployment with two GPUs runs the entire autoinfer campaign: the
reference replica on GPU 1, candidate vLLM processes on GPU 0 (spawned
per trial by the autoinfer controller on-node). Dev-side orchestration
creates the deployment, tails its logs, fetches artifacts via a
built-in HTTP server, and deletes the deployment on completion.

This mirrors ../autoresearch-rl's Basilica integration in spirit (one
owner script on dev, heavy work on Basilica), but collapses into a
single long-running deployment because autoinfer trials share the
reference replica.
"""

from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CampaignSpec:
    """Everything dev-side code needs to build a Basilica campaign.

    Kept as plain data so tests can assert on the generated source and
    the deployment kwargs without an SDK round-trip.
    """

    repo_url: str = "https://github.com/epappas/autoinfer.git"
    branch: str = "main"
    autoinfer_config: str = "examples/qwen3-8b-l1-slice/config.yaml"
    model: str = "Qwen/Qwen3-8B"
    reference_port: int = 8001
    candidate_port: int = 8000
    artifacts_port: int = 9000
    max_trials: int | None = None
    extra_autoinfer_args: tuple[str, ...] = ()
    hf_token_env: str | None = None
    env: dict[str, str] = field(default_factory=dict)

    def build_source(self) -> str:
        """Return the Python source that runs inside the Basilica container."""
        trials_arg = (
            f'            "--max-trials", "{self.max_trials}",\n'
            if self.max_trials is not None
            else ""
        )
        extra_args = "".join(
            f'            "{a}",\n' for a in self.extra_autoinfer_args
        )
        hf_login = (
            '    if os.environ.get("HF_TOKEN"):\n'
            '        subprocess.run(\n'
            '            ["huggingface-cli", "login", "--token", os.environ["HF_TOKEN"], "--add-to-git-credential"],\n'
            '            check=False,\n'
            '        )\n'
            if self.hf_token_env
            else ""
        )
        return textwrap.dedent(
            f'''
"""Autoinfer campaign runner — runs inside the Basilica deployment."""
from __future__ import annotations

import http.server
import json
import os
import socket
import socketserver
import subprocess
import sys
import time
from pathlib import Path


REPO_URL = {self.repo_url!r}
BRANCH = {self.branch!r}
CONFIG = {self.autoinfer_config!r}
MODEL = {self.model!r}
REF_PORT = {self.reference_port}
CAND_PORT = {self.candidate_port}
ART_PORT = {self.artifacts_port}
WORKDIR = Path("/workspace/autoinfer")


def log(msg: str) -> None:
    print(f"[campaign] {{msg}}", flush=True)


def sh(cmd, **kw) -> subprocess.CompletedProcess:
    log("$ " + " ".join(cmd))
    return subprocess.run(cmd, check=True, **kw)


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


def install_uv() -> None:
    bin_dir = Path.home() / ".local" / "bin"
    subprocess.run(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        shell=True, check=True,
    )
    os.environ["PATH"] = f"{{bin_dir}}:" + os.environ.get("PATH", "")


def clone_repo() -> None:
    WORKDIR.parent.mkdir(parents=True, exist_ok=True)
    if WORKDIR.exists():
        log(f"repo already present at {{WORKDIR}}")
        return
    sh(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(WORKDIR)])


def install_deps() -> None:
    sh(["uv", "sync", "--extra", "dev", "--extra", "vllm"], cwd=str(WORKDIR))


def prepare_data() -> None:
    sh(
        ["uv", "run", "python", "scripts/fetch_sharegpt.py", "--out-dir", "."],
        cwd=str(WORKDIR),
    )


def start_reference() -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"
    log_path = WORKDIR / "runs" / "reference.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    out = log_path.open("wb")
    proc = subprocess.Popen(
        [
            "uv", "run", "vllm", "serve", MODEL,
            "--port", str(REF_PORT),
            "--dtype", "auto",
            "--gpu-memory-utilization", "0.85",
        ],
        env=env, cwd=str(WORKDIR), stdout=out, stderr=subprocess.STDOUT,
    )
    log(f"reference replica pid={{proc.pid}}")
    ready = wait_for_port("127.0.0.1", REF_PORT, timeout_s=900)
    if not ready:
        proc.terminate()
        tail = log_path.read_text(errors="replace")[-2000:]
        raise RuntimeError(f"reference replica not ready in 15min\\n{{tail}}")
    log("reference replica READY")
    return proc


def run_campaign() -> int:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    cmd = [
        "uv", "run", "autoinfer", "run", CONFIG,
{trials_arg}{extra_args}    ]
    log("starting autoinfer run")
    result = subprocess.run(cmd, env=env, cwd=str(WORKDIR))
    log(f"autoinfer exit code: {{result.returncode}}")
    return result.returncode


def serve_artifacts_forever() -> None:
    os.chdir(str(WORKDIR / "runs"))
    class Q(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a, **kw): pass
    class S(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
    log(f"artifacts HTTP on 0.0.0.0:{{ART_PORT}} (runs/)")
    S(("0.0.0.0", ART_PORT), Q).serve_forever()


def summarize() -> None:
    runs = WORKDIR / "runs"
    if not runs.exists():
        log("no runs/ directory to summarize")
        return
    trial_files = sorted(runs.rglob("*.json"))
    log("--- SUMMARY ---")
    log(f"trial json files: {{len(trial_files)}}")
    for f in trial_files[:60]:
        try:
            payload = json.loads(f.read_text())
        except Exception:
            continue
        meas = "yes" if payload.get("measurement") else "no"
        fail = payload.get("failure")
        fail_kind = fail.get("kind") if isinstance(fail, dict) else "no"
        log(f"  {{f.relative_to(runs)}}: layer={{payload.get('layer')}} measurement={{meas}} failure={{fail_kind}}")
    log("--- END SUMMARY ---")


def main() -> int:
    os.makedirs("/workspace", exist_ok=True)
    os.chdir("/workspace")
    log("== autoinfer iteration-zero campaign ==")
    log(f"pwd={{os.getcwd()}}  python={{sys.version.split()[0]}}")
{hf_login}
    install_uv()
    clone_repo()
    install_deps()
    prepare_data()
    ref_proc = start_reference()
    try:
        code = run_campaign()
        log(f"campaign finished rc={{code}}")
        summarize()
    finally:
        log("terminating reference replica")
        try:
            ref_proc.terminate()
            ref_proc.wait(timeout=30)
        except Exception as e:
            log(f"reference teardown error: {{e}}")
    serve_artifacts_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
            '''
        ).strip()

    def build_deploy_kwargs(
        self,
        name: str,
        image: str = "pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel",
        gpu_count: int = 2,
        memory: str = "128Gi",
        storage: bool = True,
        ttl_seconds: int = 43200,
        timeout: int = 1800,
        min_gpu_memory_gb: int = 40,
    ) -> dict[str, Any]:
        """Return kwargs for ``BasilicaClient.deploy``.

        HF_TOKEN is passed through from the named env var if set.
        """
        env = dict(self.env)
        if self.hf_token_env is not None:
            hf_token = os.environ.get(self.hf_token_env)
            if hf_token:
                env["HF_TOKEN"] = hf_token
        return {
            "name": name,
            "source": self.build_source(),
            "image": image,
            "port": self.artifacts_port,
            "gpu_count": gpu_count,
            "min_gpu_memory_gb": min_gpu_memory_gb,
            "memory": memory,
            "storage": storage,
            "ttl_seconds": ttl_seconds,
            "timeout": timeout,
            "env": env,
        }
