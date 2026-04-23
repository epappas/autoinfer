"""Basilica deployment helper.

One deployment with two GPUs runs the entire autoinfer campaign: the
reference replica on GPU 1, candidate vLLM processes on GPU 0 (spawned
per trial by the autoinfer controller on-node). Dev-side orchestration
creates the deployment, tails its logs, fetches artifacts via a
built-in HTTP server, and deletes the deployment on completion.

The HTTP server on ``artifacts_port`` starts BEFORE the campaign so
Basilica's startup health check passes. The campaign runs in the main
thread after the server is up, updates a ``PHASE`` dict visible at
``/status``, and the ``/`` endpoint serves a flat recursive listing of
``runs/*.json`` once the campaign populates the directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

_CONTAINER_TEMPLATE = '''"""Autoinfer campaign runner — runs inside the Basilica deployment."""
from __future__ import annotations

import http.server
import json
import os
import socket
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path


REPO_URL = __REPO_URL__
BRANCH = __BRANCH__
CONFIG = __CONFIG__
MODEL = __MODEL__
REF_PORT = __REF_PORT__
CAND_PORT = __CAND_PORT__
ART_PORT = __ART_PORT__
WORKDIR = Path("/workspace/autoinfer")

PHASE = {"stage": "starting", "rc": None}


def log(msg):
    print("[campaign] " + msg, flush=True)


def sh(cmd, **kw):
    log("$ " + " ".join(cmd))
    return subprocess.run(cmd, check=True, **kw)


def wait_for_port(host, port, timeout_s):
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


class ArtifactsHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self._reply(200, b"ok", "text/plain")
            return
        if self.path == "/status":
            self._reply(200, json.dumps(PHASE).encode(), "application/json")
            return
        runs_root = WORKDIR / "runs"
        if not runs_root.exists():
            body = b"<html><body><p>runs/ not yet populated; see /status</p></body></html>"
            self._reply(200, body, "text/html")
            return
        rel = self.path.lstrip("/")
        if rel in ("", "/"):
            jsons = sorted(runs_root.rglob("*.json"))
            items = []
            for p in jsons:
                r = p.relative_to(runs_root).as_posix()
                items.append('<li><a href="' + r + '">' + r + '</a></li>')
            body = ("<html><body><h1>runs</h1><ul>" + "".join(items) + "</ul></body></html>").encode()
            self._reply(200, body, "text/html")
            return
        target = (runs_root / rel).resolve()
        root_real = runs_root.resolve()
        if not str(target).startswith(str(root_real)):
            self._reply(403, b"forbidden", "text/plain")
            return
        if not target.exists() or not target.is_file():
            self._reply(404, b"not found", "text/plain")
            return
        data = target.read_bytes()
        ctype = "application/json" if target.suffix == ".json" else "application/octet-stream"
        self._reply(200, data, ctype)

    def _reply(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a, **kw):
        pass


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def start_status_server():
    """Start before any slow step so Basilica's startup health check succeeds."""
    srv = _Server(("0.0.0.0", ART_PORT), ArtifactsHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    log("artifacts HTTP on 0.0.0.0:" + str(ART_PORT))


def install_uv():
    bin_dir = Path.home() / ".local" / "bin"
    subprocess.run("curl -LsSf https://astral.sh/uv/install.sh | sh", shell=True, check=True)
    os.environ["PATH"] = str(bin_dir) + ":" + os.environ.get("PATH", "")


def clone_repo():
    WORKDIR.parent.mkdir(parents=True, exist_ok=True)
    if WORKDIR.exists():
        log("repo already present at " + str(WORKDIR))
        return
    sh(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(WORKDIR)])


def install_deps():
    sh(["uv", "sync", "--extra", "dev", "--extra", "vllm"], cwd=str(WORKDIR))


def prepare_data():
    sh(["uv", "run", "python", "scripts/fetch_sharegpt.py", "--out-dir", "."], cwd=str(WORKDIR))


def start_reference():
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
    log("reference replica pid=" + str(proc.pid))
    if not wait_for_port("127.0.0.1", REF_PORT, timeout_s=1200):
        proc.terminate()
        tail = log_path.read_text(errors="replace")[-2000:]
        raise RuntimeError("reference replica not ready in 20min\\n" + tail)
    log("reference replica READY")
    return proc


def run_campaign():
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    cmd = [
        "uv", "run", "autoinfer", "run", CONFIG,
__TRIALS_ARG____EXTRA_ARGS__    ]
    log("starting autoinfer run")
    result = subprocess.run(cmd, env=env, cwd=str(WORKDIR))
    log("autoinfer exit code: " + str(result.returncode))
    return result.returncode


def summarize():
    runs = WORKDIR / "runs"
    if not runs.exists():
        log("no runs/ directory to summarize")
        return
    trial_files = sorted(runs.rglob("*.json"))
    log("--- SUMMARY ---")
    log("trial json files: " + str(len(trial_files)))
    for f in trial_files[:60]:
        try:
            payload = json.loads(f.read_text())
        except Exception:
            continue
        meas = "yes" if payload.get("measurement") else "no"
        fail = payload.get("failure")
        fail_kind = fail.get("kind") if isinstance(fail, dict) else "no"
        log("  " + str(f.relative_to(runs)) + ": layer=" + str(payload.get("layer")) + " measurement=" + meas + " failure=" + str(fail_kind))
    log("--- END SUMMARY ---")


def main():
    os.makedirs("/workspace", exist_ok=True)
    os.chdir("/workspace")
    log("== autoinfer iteration-zero campaign ==")
    log("pwd=" + os.getcwd() + "  python=" + sys.version.split()[0])
    start_status_server()
__HF_LOGIN__    PHASE["stage"] = "install_uv"
    install_uv()
    PHASE["stage"] = "clone_repo"
    clone_repo()
    PHASE["stage"] = "install_deps"
    install_deps()
    PHASE["stage"] = "prepare_data"
    prepare_data()
    PHASE["stage"] = "start_reference"
    ref_proc = start_reference()
    try:
        PHASE["stage"] = "run_campaign"
        code = run_campaign()
        PHASE["rc"] = code
        log("campaign finished rc=" + str(code))
        summarize()
        PHASE["stage"] = "done"
    finally:
        log("terminating reference replica")
        try:
            ref_proc.terminate()
            ref_proc.wait(timeout=30)
        except Exception as e:
            log("reference teardown error: " + str(e))
    log("campaign complete; keeping HTTP server alive for artifact download")
    while True:
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
'''


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
        """Return the Python source run inside the Basilica container."""
        trials_arg = (
            '        "--max-trials", "' + str(self.max_trials) + '",\n'
            if self.max_trials is not None
            else ""
        )
        extra_args = "".join(
            '        "' + a + '",\n' for a in self.extra_autoinfer_args
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
        return (
            _CONTAINER_TEMPLATE
            .replace("__REPO_URL__", repr(self.repo_url))
            .replace("__BRANCH__", repr(self.branch))
            .replace("__CONFIG__", repr(self.autoinfer_config))
            .replace("__MODEL__", repr(self.model))
            .replace("__REF_PORT__", str(self.reference_port))
            .replace("__CAND_PORT__", str(self.candidate_port))
            .replace("__ART_PORT__", str(self.artifacts_port))
            .replace("__TRIALS_ARG__", trials_arg)
            .replace("__EXTRA_ARGS__", extra_args)
            .replace("__HF_LOGIN__", hf_login)
        )

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
