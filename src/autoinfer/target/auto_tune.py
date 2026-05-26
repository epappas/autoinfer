"""Basilica deployment helper for the vLLM ``auto_tune.sh`` baseline (T-37).

Sister to ``target/basilica.py``. Same architecture: dev-side code
generates a small bootstrap source (under 2 KB, ASCII-only) that
Basilica's deploy-time validator accepts; the bootstrap starts an HTTP
server immediately so the scheduler's health check passes, then runs
``vllm/benchmarks/auto_tune/auto_tune.sh`` and serves the resulting
``result.txt`` + vLLM/bench logs back to dev for archival.

Why a separate spec instead of extending ``CampaignSpec``:

- Auto-tune is single-GPU and runs a different in-container command
  than autoinfer's controller. Mixing the two surfaces would hide the
  baseline run's distinct contract.
- The campaign-done marker the orchestrator scans for ("campaign
  finished rc=") is identical so we reuse ``orchestrate_iteration_zero``'s
  log streamer and artifact fetcher.
- Field set is auto_tune-specific (``INPUT_LEN`` / ``OUTPUT_LEN`` /
  ``MAX_LATENCY_ALLOWED_MS`` / etc.); ``CampaignSpec`` doesn't know
  about them and shouldn't.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

# Bootstrap is a single Python string with __PLACEHOLDER__ tokens. ASCII-
# only and within ~6 KB so Basilica's deploy-time validator accepts it
# (the sibling CampaignSpec template, known-good in production, is
# ~5 KB). State + HTTP-server pattern mirrors ``target/basilica.py``.
_BOOTSTRAP_TEMPLATE = """import http.server
import os
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path

ART_PORT = __ART_PORT__
VLLM_VERSION = __VLLM_VERSION__
MODEL = __MODEL__
INPUT_LEN = __INPUT_LEN__
OUTPUT_LEN = __OUTPUT_LEN__
MAX_MODEL_LEN = __MAX_MODEL_LEN__
MAX_LATENCY_ALLOWED_MS = __MAX_LATENCY_ALLOWED_MS__
MIN_CACHE_HIT_PCT = __MIN_CACHE_HIT_PCT__
NUM_SEQS_LIST = __NUM_SEQS_LIST__
NUM_BATCHED_TOKENS_LIST = __NUM_BATCHED_TOKENS_LIST__
TP = __TP__
WORKBASE = Path("/workspace")
VLLM_DIR = WORKBASE / "vllm"
RESULTS_DIR = WORKBASE / "auto-benchmark"

STATE = {"stage": "booting", "error": None}


def log(msg):
    print("[boot] " + msg, flush=True)


class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        rel = self.path.lstrip("/")
        if RESULTS_DIR.exists() and rel:
            target = (RESULTS_DIR / rel).resolve()
            if str(target).startswith(str(RESULTS_DIR.resolve())) and target.is_file():
                data = target.read_bytes()
                ctype = "text/plain"
                self._r(200, data, ctype)
                return
        items = []
        if RESULTS_DIR.exists():
            for p in sorted(RESULTS_DIR.rglob("*")):
                if p.is_file():
                    r = p.relative_to(RESULTS_DIR).as_posix()
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

    def log_message(self, fmt, *args):
        return


class S(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


def run_baseline():
    try:
        STATE["stage"] = "apt_install"
        log("apt-get install git")
        env = dict(os.environ, DEBIAN_FRONTEND="noninteractive")
        r = subprocess.run(["apt-get", "update", "-qq"], env=env)
        if r.returncode != 0:
            log("apt-get update rc=" + str(r.returncode) + " (proceeding anyway)")
        # ``bc`` is required by ``auto_tune.sh``'s GMU-find decrement loop
        # (line 255: ``while (( $(echo "$gpu_memory_utilization >= 0.9" |
        # bc -l) ))``). The ``vllm/vllm-openai:v0.21.0`` base image
        # omits bc by default; without it the loop exits immediately
        # with "Cannot find a proper gpu_memory_utilization over 0.9".
        # Empirically confirmed by the first T-37 attempt (2026-05-26).
        r = subprocess.run(
            ["apt-get", "install", "-yqq", "git", "ca-certificates", "bc"],
            env=env,
        )
        if r.returncode != 0:
            STATE["stage"] = "apt_install_failed"
            STATE["error"] = "apt install git rc=" + str(r.returncode)
            log(STATE["error"])
            return

        STATE["stage"] = "pip_install_vllm"
        log("installing vllm==" + VLLM_VERSION + " + datasets")
        r = subprocess.run(
            [
                sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir",
                "vllm==" + VLLM_VERSION, "datasets",
            ],
        )
        if r.returncode != 0:
            STATE["stage"] = "pip_failed"
            STATE["error"] = "pip install vllm rc=" + str(r.returncode)
            log(STATE["error"])
            return

        STATE["stage"] = "git_clone"
        log("cloning vllm@v" + VLLM_VERSION)
        WORKBASE.mkdir(parents=True, exist_ok=True)
        if not VLLM_DIR.exists():
            r = subprocess.run(
                [
                    "git", "clone", "--depth", "1",
                    "--branch", "v" + VLLM_VERSION,
                    "https://github.com/vllm-project/vllm.git", str(VLLM_DIR),
                ],
            )
            if r.returncode != 0:
                STATE["stage"] = "git_failed"
                STATE["error"] = "git clone rc=" + str(r.returncode)
                log(STATE["error"])
                return

        # auto_tune.sh cds to $BASE/vllm. Python prepends cwd to
        # sys.path so the cloned vllm/ dir would shadow the installed
        # wheel and ``import vllm._C`` (compiled extension) fails.
        # Rename it; benchmarks/auto_tune/ + .git stay intact.
        # T-37 attempt 2 confirmed (2026-05-26).
        shadow_pkg = VLLM_DIR / "vllm"
        if shadow_pkg.exists():
            STATE["stage"] = "shadow_rename"
            log("renaming " + str(shadow_pkg) + " to avoid import shadow")
            r = subprocess.run(
                ["mv", str(shadow_pkg), str(VLLM_DIR / ".vllm_src_shadow_moved")],
            )
            if r.returncode != 0:
                STATE["stage"] = "shadow_rename_failed"
                STATE["error"] = "shadow rename rc=" + str(r.returncode)
                log(STATE["error"])
                return

        # auto_tune.sh line 21 ``HOSTNAME=$(hostname)`` picks the pod
        # name on Basilica; ``vllm bench serve --host POD_NAME`` then
        # fails to connect from inside the same pod. Force localhost
        # via sed; verify the substitution fired (sed returns 0 even
        # on no-match, so check the file). T-37 attempt 3 confirmed.
        STATE["stage"] = "patch_hostname"
        auto_tune_sh = VLLM_DIR / "benchmarks" / "auto_tune" / "auto_tune.sh"
        log("patching HOSTNAME=$(hostname) -> HOSTNAME=localhost")
        r = subprocess.run(
            [
                "sed", "-i",
                r"s|^HOSTNAME=$(hostname)$|HOSTNAME=localhost|",
                str(auto_tune_sh),
            ],
        )
        if r.returncode != 0:
            STATE["stage"] = "patch_hostname_failed"
            STATE["error"] = "sed patch rc=" + str(r.returncode)
            log(STATE["error"])
            return
        verify = subprocess.run(
            ["grep", "-cF", "HOSTNAME=localhost", str(auto_tune_sh)],
            capture_output=True, text=True,
        )
        if verify.returncode != 0 or verify.stdout.strip() == "0":
            STATE["stage"] = "patch_hostname_verify_failed"
            STATE["error"] = "HOSTNAME=localhost missing after sed"
            log(STATE["error"])
            return

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        auto_tune_env = dict(os.environ)
        auto_tune_env.update({
            "BASE": str(WORKBASE),
            "MODEL": MODEL,
            "SYSTEM": "GPU",
            "TP": str(TP),
            "DOWNLOAD_DIR": "",
            "INPUT_LEN": str(INPUT_LEN),
            "OUTPUT_LEN": str(OUTPUT_LEN),
            "MAX_MODEL_LEN": str(MAX_MODEL_LEN),
            "MIN_CACHE_HIT_PCT": str(MIN_CACHE_HIT_PCT),
            "MAX_LATENCY_ALLOWED_MS": str(MAX_LATENCY_ALLOWED_MS),
            "NUM_SEQS_LIST": NUM_SEQS_LIST,
            "NUM_BATCHED_TOKENS_LIST": NUM_BATCHED_TOKENS_LIST,
            "VLLM_LOGGING_LEVEL": "INFO",
        })
        STATE["stage"] = "auto_tune_running"
        cmd = ["bash", str(VLLM_DIR / "benchmarks" / "auto_tune" / "auto_tune.sh")]
        log("running: " + " ".join(cmd))
        log("env: MODEL=" + MODEL + " INPUT_LEN=" + str(INPUT_LEN) + " OUTPUT_LEN=" + str(OUTPUT_LEN))
        log("env: MAX_LATENCY_ALLOWED_MS=" + str(MAX_LATENCY_ALLOWED_MS) + " TP=" + str(TP))
        result = subprocess.run(
            cmd,
            cwd=str(VLLM_DIR / "benchmarks" / "auto_tune"),
            env=auto_tune_env,
        )
        STATE["stage"] = "auto_tune_done"
        # Echo the campaign-done marker the orchestrator's _stream_logs
        # scans for; rc=0 means a clean run, non-zero documented in
        # result.txt under the same dir.
        print("campaign finished rc=" + str(result.returncode), flush=True)
    except Exception as e:
        STATE["stage"] = "bootstrap_exception"
        STATE["error"] = repr(e)
        log("bootstrap error: " + repr(e))


threading.Thread(
    target=lambda: S(("0.0.0.0", ART_PORT), H).serve_forever(),
    daemon=True,
).start()
log("http server on 0.0.0.0:" + str(ART_PORT))
time.sleep(2.0)

if os.environ.get("HF_TOKEN"):
    log("HF_TOKEN present")

run_baseline()

log("keeping http server alive for artifact download and log inspection")
while True:
    time.sleep(60)
"""


@dataclass
class AutoTuneBaselineSpec:
    """Everything dev-side code needs to deploy the auto_tune.sh baseline.

    Maps 1:1 to ``benchmarks/auto_tune/auto_tune.sh``'s environment
    contract (see vllm-project/vllm + this repo's C04 recon
    addendum). Defaults match the C04-recon-recommended Llama-3.1-8B /
    1×A100 / latency-bounded throughput tuple.
    """

    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    vllm_version: str = "0.21.0"
    tp: int = 1
    input_len: int = 1800
    output_len: int = 20
    max_model_len: int = 2048
    max_latency_allowed_ms: int = 500
    min_cache_hit_pct: int = 0
    num_seqs_list: str = "128 256"
    num_batched_tokens_list: str = "512 1024 2048 4096"
    artifacts_port: int = 9000
    hf_token_env: str | None = None
    env: dict[str, str] = field(default_factory=dict)

    def build_source(self) -> str:
        """Return the bootstrap Python source for the Basilica deployment.

        ``repr()`` is used on every string so quotes / backslashes
        survive template substitution intact. Numeric values stringify
        directly.
        """
        return (
            _BOOTSTRAP_TEMPLATE
            .replace("__ART_PORT__", str(self.artifacts_port))
            .replace("__VLLM_VERSION__", repr(self.vllm_version))
            .replace("__MODEL__", repr(self.model))
            .replace("__INPUT_LEN__", str(self.input_len))
            .replace("__OUTPUT_LEN__", str(self.output_len))
            .replace("__MAX_MODEL_LEN__", str(self.max_model_len))
            .replace("__MAX_LATENCY_ALLOWED_MS__", str(self.max_latency_allowed_ms))
            .replace("__MIN_CACHE_HIT_PCT__", str(self.min_cache_hit_pct))
            .replace("__NUM_SEQS_LIST__", repr(self.num_seqs_list))
            .replace("__NUM_BATCHED_TOKENS_LIST__", repr(self.num_batched_tokens_list))
            .replace("__TP__", str(self.tp))
        )

    def build_deploy_kwargs(
        self,
        name: str,
        # Pin to the explicit vllm-openai tag matching ``vllm_version`` —
        # the ``:latest`` floating tag was flagged by the C04 recon as a
        # reproducibility risk. The bootstrap also installs vllm by pip
        # at the same version, but using the matching base image avoids
        # CUDA/driver mismatches at container boot.
        image: str | None = None,
        gpu_count: int = 1,
        memory: str = "64Gi",
        storage: bool = True,
        ttl_seconds: int = 14400,
        timeout: int = 1800,
        min_gpu_memory_gb: int = 40,
        gpu_models: list[str] | None = None,
        spot: bool | None = None,
        cpu: str = "4",
    ) -> dict[str, Any]:
        """Return kwargs for ``BasilicaClient.deploy``.

        Defaults: 1× GPU, 40 GB min HBM (Llama-3.1-8B fp16 weights ~16
        GB + KV cache headroom). ``ttl_seconds`` lower than the
        campaign default (14400 = 4 h) because auto_tune.sh's worst
        case is ~90 min — anything longer indicates a hung run.
        """
        import basilica

        if image is None:
            image = f"vllm/vllm-openai:v{self.vllm_version}"

        env = dict(self.env)
        if self.hf_token_env is not None:
            hf_token = os.environ.get(self.hf_token_env)
            if hf_token:
                env["HF_TOKEN"] = hf_token

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
            "cpu": cpu,
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
        if spot is not None:
            kwargs["spot"] = spot
        return kwargs


__all__ = ["AutoTuneBaselineSpec"]
