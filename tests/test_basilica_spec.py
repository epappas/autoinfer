from __future__ import annotations

from autoinfer.target.basilica import CampaignSpec


def test_default_spec_source_contains_expected_phases() -> None:
    src = CampaignSpec().build_source()
    # phases the dev-side orchestrator relies on
    assert "install_uv()" in src
    assert "clone_repo()" in src
    assert "install_deps()" in src
    assert "prepare_data()" in src
    assert "start_reference()" in src
    assert "run_campaign()" in src
    assert "serve_artifacts_forever()" in src
    assert "summarize()" in src
    # completion marker the orchestrator tails for
    assert "campaign finished rc=" in src


def test_source_pins_config_and_model() -> None:
    spec = CampaignSpec(
        autoinfer_config="examples/qwen3-8b-l1-slice/config.yaml",
        model="Qwen/Qwen3-8B",
    )
    src = spec.build_source()
    assert "Qwen/Qwen3-8B" in src
    assert "examples/qwen3-8b-l1-slice/config.yaml" in src


def test_source_injects_max_trials_when_set() -> None:
    src = CampaignSpec(max_trials=3).build_source()
    assert '"--max-trials", "3",' in src


def test_source_omits_max_trials_when_none() -> None:
    src = CampaignSpec(max_trials=None).build_source()
    # no --max-trials flag in the command
    assert '"--max-trials"' not in src


def test_source_injects_extra_autoinfer_args() -> None:
    src = CampaignSpec(extra_autoinfer_args=("--foo", "bar")).build_source()
    assert '"--foo",' in src
    assert '"bar",' in src


def test_source_includes_hf_login_only_when_token_env_is_set() -> None:
    no_hf = CampaignSpec().build_source()
    with_hf = CampaignSpec(hf_token_env="HF_TOKEN").build_source()
    assert "huggingface-cli" not in no_hf
    assert "huggingface-cli" in with_hf
    assert 'os.environ["HF_TOKEN"]' in with_hf


def test_deploy_kwargs_minimal() -> None:
    spec = CampaignSpec()
    kw = spec.build_deploy_kwargs(name="test-deploy")
    assert kw["name"] == "test-deploy"
    assert kw["image"] == "pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel"
    assert kw["gpu_count"] == 2
    assert kw["memory"] == "128Gi"
    assert kw["storage"] is True
    assert kw["port"] == 9000
    assert kw["ttl_seconds"] == 43200
    assert kw["min_gpu_memory_gb"] == 40
    assert isinstance(kw["source"], str) and len(kw["source"]) > 100


def test_deploy_kwargs_custom_overrides() -> None:
    spec = CampaignSpec()
    kw = spec.build_deploy_kwargs(
        name="custom",
        image="nvidia/cuda:12.4.0-devel-ubuntu22.04",
        gpu_count=1,
        memory="64Gi",
        storage=False,
        ttl_seconds=3600,
        timeout=600,
        min_gpu_memory_gb=24,
    )
    assert kw["image"] == "nvidia/cuda:12.4.0-devel-ubuntu22.04"
    assert kw["gpu_count"] == 1
    assert kw["memory"] == "64Gi"
    assert kw["storage"] is False
    assert kw["ttl_seconds"] == 3600
    assert kw["timeout"] == 600
    assert kw["min_gpu_memory_gb"] == 24


def test_deploy_kwargs_passes_hf_token_when_env_set(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_xxxxxxxxxxxx")
    spec = CampaignSpec(hf_token_env="HF_TOKEN")
    kw = spec.build_deploy_kwargs(name="n")
    assert kw["env"]["HF_TOKEN"] == "hf_xxxxxxxxxxxx"


def test_deploy_kwargs_omits_hf_token_when_env_unset(monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    spec = CampaignSpec(hf_token_env="HF_TOKEN")
    kw = spec.build_deploy_kwargs(name="n")
    assert "HF_TOKEN" not in kw["env"]


def test_source_compiles_as_valid_python() -> None:
    """The generated container source must be parseable by the container's Python."""
    import ast

    src = CampaignSpec(max_trials=3, hf_token_env="HF_TOKEN").build_source()
    ast.parse(src)


def test_ports_flow_into_source_text() -> None:
    spec = CampaignSpec(
        reference_port=9901, candidate_port=9900, artifacts_port=9999
    )
    src = spec.build_source()
    assert "REF_PORT = 9901" in src
    assert "CAND_PORT = 9900" in src
    assert "ART_PORT = 9999" in src
