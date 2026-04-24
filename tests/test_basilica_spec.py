from __future__ import annotations

from autoinfer.target.basilica import CampaignSpec


def test_bootstrap_source_is_ascii_only() -> None:
    """Basilica's validator rejects sources with non-ASCII chars."""
    src = CampaignSpec().build_source()
    for i, ch in enumerate(src):
        assert ord(ch) < 128, f"non-ASCII at index {i}: {ch!r}"


def test_bootstrap_source_is_small() -> None:
    """Bootstrap must stay small so the deploy-time validator accepts it."""
    src = CampaignSpec().build_source()
    assert len(src) < 5000, f"bootstrap too large: {len(src)} chars"


def test_bootstrap_starts_http_server_thread_before_calling_run_campaign() -> None:
    """Regression: HTTP must be up before any pip install / git clone so
    basilica's startup probe succeeds even if those steps are slow/fail."""
    src = CampaignSpec().build_source()
    http_idx = src.index("serve_forever")
    # find the top-level call (preceded by newline), not the `def` line
    run_idx = src.index("\nrun_campaign()")
    assert http_idx < run_idx, "HTTP server thread must start before run_campaign()"


def test_bootstrap_handler_always_returns_200_for_get() -> None:
    """The handler must 200 on every GET during bootstrap so any
    health-check path Basilica probes succeeds."""
    src = CampaignSpec().build_source()
    # must implement do_GET and do_HEAD; must not have 404 or 403 fast-paths
    assert "do_GET" in src
    assert "do_HEAD" in src
    # runs-not-found path must still yield 200
    assert 'self._r(200, body, "text/html"' in src


def test_bootstrap_uses_pip_not_curl_pipe() -> None:
    """Bootstrap installs uv via pip, not `curl | sh`, to keep the source safe."""
    src = CampaignSpec().build_source()
    assert "curl" not in src
    assert "pip" in src


def test_bootstrap_clones_then_execs_campaign_runner() -> None:
    src = CampaignSpec().build_source()
    assert "git" in src and "clone" in src
    assert "scripts/campaign_runner.py" in src
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
    assert "MAX_TRIALS = 3" in src
    assert '"--max-trials"' in src


def test_source_omits_max_trials_flag_when_none() -> None:
    src = CampaignSpec(max_trials=None).build_source()
    # MAX_TRIALS = None so the --max-trials cmd extension is skipped at runtime
    assert "MAX_TRIALS = None" in src


def test_source_injects_layer_trials() -> None:
    src = CampaignSpec(
        layer_trials=["l1_engine=3", "l2_topology=1"]
    ).build_source()
    assert "LAYER_TRIALS = ['l1_engine=3', 'l2_topology=1']" in src
    assert '"--layer-trials"' in src


def test_source_empty_layer_trials_when_unset() -> None:
    src = CampaignSpec().build_source()
    assert "LAYER_TRIALS = []" in src


def test_deploy_kwargs_minimal() -> None:
    spec = CampaignSpec()
    kw = spec.build_deploy_kwargs(name="test-deploy")
    assert kw["name"] == "test-deploy"
    assert kw["image"] == "pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime"
    assert kw["gpu_count"] == 2
    assert kw["memory"] == "64Gi"
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
    spec = CampaignSpec(reference_port=9901, artifacts_port=9999)
    src = spec.build_source()
    assert "REF_PORT = 9901" in src
    assert "ART_PORT = 9999" in src
