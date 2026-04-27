"""Basilica deployment helper.

One deployment with two GPUs runs the entire autoinfer campaign: the
reference replica on GPU 1, candidate vLLM processes on GPU 0. Dev-side
orchestration creates the deployment, tails its logs, fetches artifacts
via a built-in HTTP server, and deletes the deployment on completion.

The Basilica-side code is split in two layers:

- This module generates a small bootstrap (well under 2 KB, ASCII-only,
  no complex patterns) that Basilica's deploy-time validator accepts.
  The bootstrap starts an HTTP server on ``artifacts_port`` immediately
  (so the scheduler's health check passes), installs uv, clones the
  autoinfer repository, and execs ``scripts/campaign_runner.py`` from
  the cloned repo.
- ``scripts/campaign_runner.py`` lives IN the repo and contains the
  full campaign logic (uv sync, trace fetch, reference replica, the
  autoinfer run itself, summarization). Changes to the campaign logic
  do not require regenerating the bootstrap.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

_BOOTSTRAP_TEMPLATE = """import http.server
import os
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path

ART_PORT = __ART_PORT__
REPO_URL = __REPO_URL__
BRANCH = __BRANCH__
CONFIG = __CONFIG__
MODEL = __MODEL__
MAX_TRIALS = __MAX_TRIALS__
LAYER_TRIALS = __LAYER_TRIALS__
REF_PORT = __REF_PORT__
WORKDIR = Path("/workspace/autoinfer")

STATE = {"stage": "booting", "error": None}


def log(msg):
    print("[boot] " + msg, flush=True)


class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        runs = WORKDIR / "runs"
        rel = self.path.lstrip("/")
        if runs.exists() and rel and rel not in ("/",):
            target = (runs / rel).resolve()
            if str(target).startswith(str(runs.resolve())) and target.is_file():
                data = target.read_bytes()
                ctype = "application/json" if target.suffix == ".json" else "text/plain"
                self._r(200, data, ctype)
                return
        items = []
        if runs.exists():
            for p in sorted(runs.rglob("*")):
                if p.is_file() and p.suffix[1:] in {"json","jsonl","tsv","log","out","err"}:
                    r = p.relative_to(runs).as_posix()
                    items.append('<li><a href="' + r + '">' + r + "</a></li>")
        body = (
            "<html><body>stage=" + str(STATE.get("stage")) + " err=" + str(STATE.get("error"))
            + "<ul>" + "".join(items) + "</ul></body></html>"
        ).encode()
        self._r(200, body, "text/html")

    def do_HEAD(self):
        self._r(200, b"", "text/plain")

    def _r(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def log_message(self, *a, **kw):
        pass


class S(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def run_campaign():
    try:
        STATE["stage"] = "apt_install_git"
        log("apt-get install git")
        env = dict(os.environ, DEBIAN_FRONTEND="noninteractive")
        r = subprocess.run(["apt-get", "update", "-qq"], env=env)
        if r.returncode != 0:
            log("apt-get update rc=" + str(r.returncode) + " (proceeding anyway)")
        r = subprocess.run(
            ["apt-get", "install", "-yqq", "git", "ca-certificates", "build-essential"],
            env=env,
        )
        if r.returncode != 0:
            STATE["stage"] = "apt_install_failed"
            STATE["error"] = "apt install git rc=" + str(r.returncode)
            log(STATE["error"])
            return

        STATE["stage"] = "pip_install_uv"
        log("installing uv")
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir", "uv"],
        )
        if r.returncode != 0:
            STATE["stage"] = "pip_failed"
            STATE["error"] = "pip install uv rc=" + str(r.returncode)
            log(STATE["error"])
            return

        STATE["stage"] = "git_clone"
        log("cloning " + REPO_URL + " branch=" + BRANCH)
        WORKDIR.parent.mkdir(parents=True, exist_ok=True)
        if not WORKDIR.exists():
            r = subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(WORKDIR)],
            )
            if r.returncode != 0:
                STATE["stage"] = "git_failed"
                STATE["error"] = "git clone rc=" + str(r.returncode)
                log(STATE["error"])
                return

        import shutil
        uv_bin = shutil.which("uv") or "uv"
        cmd = [
            uv_bin,
            "run", "python", "scripts/campaign_runner.py",
            "--config", CONFIG,
            "--model", MODEL,
            "--ref-port", str(REF_PORT),
        ]
        if MAX_TRIALS is not None:
            cmd.extend(["--max-trials", str(MAX_TRIALS)])
        for lt in LAYER_TRIALS:
            cmd.extend(["--layer-trials", lt])

        STATE["stage"] = "campaign_running"
        log("running: " + " ".join(cmd))
        result = subprocess.run(cmd, cwd=str(WORKDIR))
        STATE["stage"] = "campaign_done"
        print("campaign finished rc=" + str(result.returncode), flush=True)
    except Exception as e:
        STATE["stage"] = "bootstrap_exception"
        STATE["error"] = repr(e)
        log("bootstrap error: " + repr(e))


# Start HTTP server in background thread; main thread runs campaign.
threading.Thread(
    target=lambda: S(("0.0.0.0", ART_PORT), H).serve_forever(),
    daemon=True,
).start()
log("http server on 0.0.0.0:" + str(ART_PORT))
time.sleep(2.0)  # let the server bind before we do anything slow

if os.environ.get("HF_TOKEN"):
    log("HF_TOKEN present")

run_campaign()

log("keeping http server alive for artifact download and log inspection")
while True:
    time.sleep(60)
"""


@dataclass
class CampaignSpec:
    """Everything dev-side code needs to build a Basilica campaign."""

    repo_url: str = "https://github.com/epappas/autoinfer.git"
    branch: str = "main"
    autoinfer_config: str = "examples/qwen3-8b-l1-slice/config.yaml"
    model: str = "Qwen/Qwen3-8B"
    reference_port: int = 8001
    candidate_port: int = 8000
    artifacts_port: int = 9000
    max_trials: int | None = None
    layer_trials: list[str] = field(default_factory=list)
    hf_token_env: str | None = None
    env: dict[str, str] = field(default_factory=dict)

    def build_source(self) -> str:
        """Return the bootstrap Python source for the Basilica deployment."""
        return (
            _BOOTSTRAP_TEMPLATE
            .replace("__ART_PORT__", str(self.artifacts_port))
            .replace("__REPO_URL__", repr(self.repo_url))
            .replace("__BRANCH__", repr(self.branch))
            .replace("__CONFIG__", repr(self.autoinfer_config))
            .replace("__MODEL__", repr(self.model))
            .replace(
                "__MAX_TRIALS__",
                "None" if self.max_trials is None else str(self.max_trials),
            )
            .replace("__LAYER_TRIALS__", repr(list(self.layer_trials)))
            .replace("__REF_PORT__", str(self.reference_port))
        )

    def build_deploy_kwargs(
        self,
        name: str,
        image: str = "pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime",
        gpu_count: int = 2,
        memory: str = "64Gi",
        storage: bool = True,
        ttl_seconds: int = 43200,
        timeout: int = 1800,
        min_gpu_memory_gb: int = 40,
        gpu_models: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return kwargs for ``BasilicaClient.deploy``.

        Includes an explicit startup HealthCheckConfig with generous
        timeouts because Basilica's default probe times out before a
        large CUDA image has finished pulling + booting on a fresh node.
        """
        import basilica

        env = dict(self.env)
        if self.hf_token_env is not None:
            hf_token = os.environ.get(self.hf_token_env)
            if hf_token:
                env["HF_TOKEN"] = hf_token
        # Startup probe gives ~10 min for apt + pip + git to finish; liveness/
        # readiness probes tolerate short HTTP hiccups once the server is up
        # (avoids container restarts during campaign).
        health = basilica.HealthCheckConfig(
            startup=basilica.ProbeConfig(
                path="/",
                initial_delay_seconds=90,
                period_seconds=15,
                timeout_seconds=10,
                failure_threshold=40,
            ),
            liveness=basilica.ProbeConfig(
                path="/",
                initial_delay_seconds=600,
                period_seconds=60,
                timeout_seconds=15,
                failure_threshold=5,
            ),
            readiness=basilica.ProbeConfig(
                path="/",
                initial_delay_seconds=60,
                period_seconds=30,
                timeout_seconds=10,
                failure_threshold=5,
            ),
        )
        kwargs: dict[str, Any] = {
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
            "health_check": health,
        }
        if gpu_models:
            kwargs["gpu_models"] = list(gpu_models)
        return kwargs
