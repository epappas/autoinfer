"""FP16 reference replica lifecycle (P8).

Manages a local ``vllm serve`` subprocess that the quality gate queries
as ground truth. Real subprocess; no fakes. The process lifecycle state
machine and port-readiness check are testable without GPU; full
start/stop requires the vllm runtime and is marked ``gpu`` in tests.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from types import TracebackType

from typing_extensions import Self


class ReferenceReplica:
    def __init__(
        self,
        model: str,
        port: int = 8001,
        dtype: str = "float16",
        extra_args: list[str] | None = None,
        *,
        seed: int | None = None,
        multiprocessing_v1: bool = True,
    ) -> None:
        self._model = model
        self._port = port
        self._dtype = dtype
        self._extra_args = list(extra_args) if extra_args else []
        self._seed = seed
        self._multiprocessing_v1 = multiprocessing_v1
        self._process: subprocess.Popen[bytes] | None = None

    @property
    def uri(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    @property
    def seed(self) -> int | None:
        return self._seed

    @property
    def multiprocessing_v1(self) -> bool:
        return self._multiprocessing_v1

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def build_cmd_env(
        self, base_env: dict[str, str] | None = None
    ) -> tuple[list[str], dict[str, str]]:
        """Assemble the ``vllm serve`` argv + env for this replica. Pure.

        Extracted from ``start`` so tests can pin the determinism wiring
        (T-36: ``--seed N`` when seed set; ``VLLM_ENABLE_V1_MULTIPROCESSING=0``
        when ``multiprocessing_v1=False``) without spawning a real
        subprocess. ``base_env`` defaults to a copy of the current process
        environment; passing an explicit dict keeps tests hermetic.
        """
        cmd = [
            "vllm", "serve", self._model,
            "--port", str(self._port),
            "--dtype", self._dtype,
            *self._extra_args,
        ]
        if self._seed is not None:
            cmd.extend(["--seed", str(self._seed)])
        env = dict(base_env) if base_env is not None else os.environ.copy()
        if not self._multiprocessing_v1:
            env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        return cmd, env

    def start(self, timeout_s: int = 300) -> None:
        if self._process is not None:
            raise RuntimeError("replica already started")
        cmd, env = self.build_cmd_env()
        self._process = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        try:
            self._wait_ready(timeout_s)
        except BaseException:
            self.stop()
            raise

    def _wait_ready(self, timeout_s: int) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                code = self._process.returncode
                raise RuntimeError(f"replica exited during startup (code {code})")
            if self._tcp_open("127.0.0.1", self._port, timeout_s=1.0):
                return
            time.sleep(2.0)
        raise TimeoutError(f"replica not ready after {timeout_s}s on port {self._port}")

    @staticmethod
    def _tcp_open(host: str, port: int, timeout_s: float) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout_s)
            try:
                s.connect((host, port))
            except (OSError, TimeoutError):
                return False
            return True

    def stop(self, timeout_s: int = 30) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self._process = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()
