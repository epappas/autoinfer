from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from autoinfer.harness.failure import FailureKind
from autoinfer.layers import TrialInput
from autoinfer.layers.l2_topology.adapter import L2TopologyAdapter
from autoinfer.layers.l2_topology.surface import load_catalog

_REPO_CATALOG = Path(__file__).parent.parent / "src/autoinfer/layers/l2_topology/knobs.yaml"


class _FakeClient:
    """Real alternate BasilicaClient for CPU-only tests.

    Not a mock — behaves like a stub-dev client: records the last
    deploy_vllm call and returns a rigged deployment object.
    """

    def __init__(self, *, raise_on_deploy: bool = False, deployment_url: str = "http://example/") -> None:
        self.raise_on_deploy = raise_on_deploy
        self.deployment_url = deployment_url
        self.calls: list[dict[str, Any]] = []
        self.deleted: list[str] = []

    def deploy_vllm(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.raise_on_deploy:
            raise RuntimeError("deploy failed")
        return _FakeDeployment(url=self.deployment_url, on_delete=self.deleted.append)


class _FakeDeployment:
    def __init__(self, url: str, on_delete: Any) -> None:
        self.url = url
        self._on_delete = on_delete

    def delete(self) -> None:
        self._on_delete(self.url)


def _mk_adapter(client: _FakeClient) -> L2TopologyAdapter:
    return L2TopologyAdapter(
        model="Qwen/Qwen3-8B",
        catalog=load_catalog(_REPO_CATALOG),
        trace_path=Path("/tmp/t"),
        reference_uri="http://ref.example",
        quality_prompts=["hi"],
        max_kl=2.0,
        result_dir=Path("/tmp/runs"),
        basilica_client=client,
    )


def test_surface_loads_from_repo_catalog() -> None:
    adapter = _mk_adapter(_FakeClient())
    surface = adapter.surface()
    assert "gpu_type" in surface
    assert surface["gpu_type"]["type"] == "categorical"


def test_deploy_failure_becomes_startup_record() -> None:
    client = _FakeClient(raise_on_deploy=True)
    adapter = _mk_adapter(client)
    out = adapter.run(TrialInput(trial_id="l2_t0000", config={"gpu_type": "RTX A6000"}))
    assert out.measurement is None
    assert out.failure is not None
    assert out.failure.kind is FailureKind.STARTUP
    assert len(client.calls) == 1  # attempted exactly once
    assert "deploy_vllm" in out.failure.message


def test_deploy_kwargs_include_gpu_models_and_tp() -> None:
    client = _FakeClient(raise_on_deploy=True)
    adapter = _mk_adapter(client)
    adapter.run(TrialInput(trial_id="l2_t0001", config={"gpu_type": "L40", "gpu_count": 2}))
    call = client.calls[0]
    assert call["gpu_models"] == ["L40"]
    assert call["gpu_count"] == 2
    assert call["tensor_parallel_size"] == 2
    assert call["model"] == "Qwen/Qwen3-8B"
    assert call["memory"] == "64Gi"


def test_deploy_pass_through_env_from_os(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")
    client = _FakeClient(raise_on_deploy=True)
    adapter = _mk_adapter(client)
    adapter.run(TrialInput(trial_id="l2_t0002", config={"gpu_type": "RTX A6000"}))
    assert client.calls[0]["env"]["HF_TOKEN"] == "hf_xxx"
