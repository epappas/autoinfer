"""L2 topology adapter — per-trial Basilica deployments with varied GPU classes.

Each ``run(trial)`` call:

1. Translates the trial config into ``BasilicaClient.deploy_vllm`` kwargs
   (see ``l2_topology.surface.config_to_deploy_kwargs``).
2. Spawns a fresh Basilica deployment, waits for its public URL to be
   ready.
3. Drives ``vllm bench serve`` against the remote URL (run_driver).
4. Runs the quality gate: candidate=remote URL,
   reference=self.reference_uri (a pre-deployed long-lived reference).
5. Tears the deployment down (always, on the ``finally`` path).

All failures convert to typed ``FailureRecord`` — the L2 adapter never
raises out of ``run``.
"""

from __future__ import annotations

import os
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autoinfer.harness.driver import run_driver
from autoinfer.harness.failure import FailureKind, FailureRecord
from autoinfer.harness.gate import run_gate
from autoinfer.layers import TrialInput, TrialOutput
from autoinfer.layers.l1_engine.adapter import compose_measurement
from autoinfer.layers.l2_topology.surface import (
    L2Catalog,
    config_to_deploy_kwargs,
    to_surrogate_surface,
)


@dataclass
class L2TopologyAdapter:
    model: str
    catalog: L2Catalog
    trace_path: Path
    reference_uri: str
    quality_prompts: list[str]
    max_kl: float
    result_dir: Path
    basilica_client: Any  # basilica.BasilicaClient; Any to avoid hard dep
    layer_name: str = "l2_topology"
    batch_sizes: tuple[int, ...] = (1,)
    dataset_name: str = "random"
    num_prompts: int = 64
    gate_concurrency: int = 4
    driver_timeout_s: int = 1800
    # 30 min gives enough slack for Basilica scheduler + pod bring-up +
    # ~16 GB model pull on a cold spot node.
    deploy_timeout_s: int = 1800
    ttl_seconds: int = 3600
    memory: str = "64Gi"
    storage: bool = True
    pass_through_env: dict[str, str] = field(default_factory=dict)

    def surface(self) -> dict[str, Any]:
        return to_surrogate_surface(self.catalog)

    def run(self, trial: TrialInput) -> TrialOutput:
        deploy_kwargs = self._build_deploy_kwargs(trial)
        deployment = None
        instance_name: str | None = None
        try:
            deployment = self.basilica_client.deploy_vllm(**deploy_kwargs)
        except Exception as e:  # noqa: BLE001
            # deploy_vllm internally calls wait_until_ready; on TIMEOUT the
            # deployment was created on Basilica but never returned to us.
            # The exception message contains the instance UUID — extract
            # and delete so we don't leak spot-GPU quota.
            msg = str(e)
            import re as _re
            m = _re.search(r"'([0-9a-f-]{36})'", msg)
            if m:
                instance_name = m.group(1)
                import contextlib as _cl
                with _cl.suppress(Exception):
                    self.basilica_client.delete_deployment(instance_name)
            return TrialOutput(
                measurement=None,
                failure=self._fail(trial, FailureKind.STARTUP, f"deploy_vllm: {e}"),
            )
        try:
            return self._run_benchmarks(trial, deployment)
        finally:
            self._safe_delete(deployment)

    def teardown(self) -> None:
        return None

    def _build_deploy_kwargs(self, trial: TrialInput) -> dict[str, Any]:
        kwargs = config_to_deploy_kwargs(trial.config, self.catalog)
        kwargs.setdefault("model", self.model)
        kwargs.setdefault("name", f"ai-l2-{trial.trial_id}-{int(time.time())}")
        kwargs.setdefault("memory", self.memory)
        kwargs.setdefault("storage", self.storage)
        kwargs.setdefault("ttl_seconds", self.ttl_seconds)
        kwargs.setdefault("timeout", self.deploy_timeout_s)
        # Generous startup probe: 2 min initial delay + 60 failures x 15s
        # = 17 min tolerance for model download + vllm warmup after the
        # container is alive. Liveness has even longer head-room so an
        # already-running candidate isn't nuked during bench runs.
        import basilica
        kwargs.setdefault(
            "health_check",
            basilica.HealthCheckConfig(
                startup=basilica.ProbeConfig(
                    path="/health",
                    initial_delay_seconds=120,
                    period_seconds=15,
                    timeout_seconds=10,
                    failure_threshold=60,
                ),
                liveness=basilica.ProbeConfig(
                    path="/health",
                    initial_delay_seconds=1200,
                    period_seconds=60,
                    timeout_seconds=15,
                    failure_threshold=5,
                ),
                readiness=basilica.ProbeConfig(
                    path="/health",
                    initial_delay_seconds=60,
                    period_seconds=30,
                    timeout_seconds=10,
                    failure_threshold=10,
                ),
            ),
        )
        env = dict(self.pass_through_env)
        for var in ("HF_TOKEN",):
            val = os.environ.get(var)
            if val:
                env[var] = val
        if env:
            kwargs.setdefault("env", env)
        return kwargs

    def _run_benchmarks(self, trial: TrialInput, deployment: Any) -> TrialOutput:
        endpoint = str(deployment.url).rstrip("/")
        try:
            driver = run_driver(
                endpoint=endpoint,
                trace_path=self.trace_path,
                model=self.model,
                result_dir=self.result_dir,
                result_name=f"{trial.trial_id}_bench.json",
                timeout_s=self.driver_timeout_s,
                dataset_name=self.dataset_name,
                num_prompts=self.num_prompts,
            )
        except Exception as e:  # noqa: BLE001
            return TrialOutput(
                measurement=None,
                failure=self._fail(trial, FailureKind.HANG, f"driver: {e}"),
            )
        try:
            gate = run_gate(
                candidate_endpoint=endpoint,
                reference_endpoint=self.reference_uri,
                model=self.model,
                prompts=self.quality_prompts,
                batch_sizes=self.batch_sizes,
                concurrency=self.gate_concurrency,
            )
        except Exception as e:  # noqa: BLE001
            return TrialOutput(
                measurement=None,
                failure=self._fail(trial, FailureKind.UNKNOWN, f"gate: {e}"),
            )
        if not gate.passes(self.max_kl):
            kind = (
                FailureKind.QUALITY_INVARIANCE
                if not gate.batch_invariant
                else FailureKind.QUALITY_KL
            )
            return TrialOutput(
                measurement=None,
                failure=self._fail(
                    trial,
                    kind,
                    f"gate rejected mean_kl={gate.mean_kl:.4f} invariant={gate.batch_invariant}",
                ),
            )
        return TrialOutput(
            measurement=compose_measurement(driver, gate, peak_hbm_gb=0.0),
            failure=None,
        )

    def _safe_delete(self, deployment: Any) -> None:
        if deployment is None:
            return
        try:
            deployment.delete()
        except Exception as e:  # noqa: BLE001
            traceback.print_exception(type(e), e, e.__traceback__)

    def _fail(self, trial: TrialInput, kind: FailureKind, msg: str) -> FailureRecord:
        return FailureRecord(
            kind=kind,
            message=msg[:500],
            trial_id=trial.trial_id,
            layer=self.layer_name,
        )
