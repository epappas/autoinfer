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
    deploy_timeout_s: int = 1200
    ttl_seconds: int = 3600
    memory: str = "64Gi"
    storage: bool = True
    pass_through_env: dict[str, str] = field(default_factory=dict)

    def surface(self) -> dict[str, Any]:
        return to_surrogate_surface(self.catalog)

    def run(self, trial: TrialInput) -> TrialOutput:
        deploy_kwargs = self._build_deploy_kwargs(trial)
        deployment = None
        try:
            deployment = self.basilica_client.deploy_vllm(**deploy_kwargs)
        except Exception as e:  # noqa: BLE001
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
