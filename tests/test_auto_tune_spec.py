"""AutoTuneBaselineSpec tests (T-37).

Mirrors ``test_basilica_spec.py`` in pattern — exercise the pure parts
of the spec (source generation, deploy-kwargs assembly) without
touching the Basilica SDK or network. Validates that the bootstrap
echoes the right auto_tune.sh env vars, the campaign-done marker is
the same string the orchestrator scans for, and HF_TOKEN passthrough
gates on the env-var name.
"""

from __future__ import annotations

import pytest

from autoinfer.target.auto_tune import AutoTuneBaselineSpec


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip HF_TOKEN by default so tests opt-in to the present-token path."""
    monkeypatch.delenv("HF_TOKEN", raising=False)


def test_default_spec_bootstrap_contains_canonical_auto_tune_invocation() -> None:
    spec = AutoTuneBaselineSpec()
    src = spec.build_source()
    # Canonical env vars (matching auto_tune.sh contract) appear by name
    # so an in-container env dump explains what was requested.
    assert "MODEL=" in src
    assert "INPUT_LEN=" in src
    assert "OUTPUT_LEN=" in src
    assert "MAX_LATENCY_ALLOWED_MS=" in src
    # The script invocation is anchored to vllm/benchmarks/auto_tune/
    assert "benchmarks" in src
    assert "auto_tune.sh" in src
    # Campaign-done marker matches orchestrator's expectation.
    assert "campaign finished rc=" in src


def test_default_spec_uses_c04_recon_recommended_values() -> None:
    spec = AutoTuneBaselineSpec()
    assert spec.model == "meta-llama/Llama-3.1-8B-Instruct"
    assert spec.vllm_version == "0.21.0"
    assert spec.tp == 1
    assert spec.input_len == 1800
    assert spec.output_len == 20
    assert spec.max_model_len == 2048
    assert spec.max_latency_allowed_ms == 500
    assert spec.min_cache_hit_pct == 0
    # auto_tune.sh defaults.
    assert spec.num_seqs_list == "128 256"
    assert spec.num_batched_tokens_list == "512 1024 2048 4096"


def test_build_source_substitutes_numeric_fields_without_quotes() -> None:
    """Integers must end up as Python literals (no repr/quote wrapping)."""
    spec = AutoTuneBaselineSpec(input_len=1234, output_len=56, max_model_len=4321)
    src = spec.build_source()
    assert "INPUT_LEN = 1234" in src
    assert "OUTPUT_LEN = 56" in src
    assert "MAX_MODEL_LEN = 4321" in src
    # No accidental quoting.
    assert "INPUT_LEN = '1234'" not in src
    assert 'INPUT_LEN = "1234"' not in src


def test_build_source_quotes_string_fields() -> None:
    """Model name and list strings must survive as Python string literals."""
    spec = AutoTuneBaselineSpec(
        model="meta-llama/Llama-3.1-8B-Instruct",
        num_seqs_list="64 128 256",
    )
    src = spec.build_source()
    assert "MODEL = 'meta-llama/Llama-3.1-8B-Instruct'" in src
    assert "NUM_SEQS_LIST = '64 128 256'" in src


def test_build_source_includes_vllm_pin_for_git_clone_and_pip() -> None:
    """vllm_version flows into both the pip install and the git clone tag."""
    spec = AutoTuneBaselineSpec(vllm_version="0.21.0")
    src = spec.build_source()
    assert "VLLM_VERSION = '0.21.0'" in src
    # pip install vllm==<VERSION>
    assert 'vllm==" + VLLM_VERSION' in src
    # git clone --branch v<VERSION>
    assert '"v" + VLLM_VERSION' in src


def test_build_source_under_size_cap() -> None:
    """Basilica validator accepts the sibling CampaignSpec template at
    ~5 KB; keep this one within the same order of magnitude (under 8 KB).
    """
    spec = AutoTuneBaselineSpec()
    assert len(spec.build_source()) < 8000


def test_build_source_pure_ascii() -> None:
    """The validator also rejects non-ASCII bootstrap source."""
    spec = AutoTuneBaselineSpec()
    spec.build_source().encode("ascii")  # raises UnicodeEncodeError on failure


def test_build_deploy_kwargs_default_image_pins_to_vllm_version() -> None:
    spec = AutoTuneBaselineSpec(vllm_version="0.21.0")
    kwargs = spec.build_deploy_kwargs(name="run-1")
    assert kwargs["image"] == "vllm/vllm-openai:v0.21.0"


def test_build_deploy_kwargs_respects_explicit_image() -> None:
    spec = AutoTuneBaselineSpec()
    kwargs = spec.build_deploy_kwargs(name="run-1", image="custom:tag")
    assert kwargs["image"] == "custom:tag"


def test_build_deploy_kwargs_minimal_shape() -> None:
    spec = AutoTuneBaselineSpec()
    kwargs = spec.build_deploy_kwargs(name="run-1")
    assert kwargs["name"] == "run-1"
    assert kwargs["gpu_count"] == 1
    assert kwargs["min_gpu_memory_gb"] == 40
    assert kwargs["cpu"] == "4"
    assert kwargs["memory"] == "64Gi"
    assert kwargs["storage"] is True
    assert kwargs["ttl_seconds"] == 14400
    assert kwargs["timeout"] == 1800
    assert "health_check" in kwargs
    assert kwargs["env"] == {}
    # gpu_models / spot left off until explicitly requested.
    assert "gpu_models" not in kwargs
    assert "spot" not in kwargs


def test_build_deploy_kwargs_passes_hf_token_when_env_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "secret-hf-token")
    spec = AutoTuneBaselineSpec(hf_token_env="HF_TOKEN")
    kwargs = spec.build_deploy_kwargs(name="run-1")
    assert kwargs["env"]["HF_TOKEN"] == "secret-hf-token"


def test_build_deploy_kwargs_omits_hf_token_when_env_unset() -> None:
    spec = AutoTuneBaselineSpec(hf_token_env="HF_TOKEN")
    kwargs = spec.build_deploy_kwargs(name="run-1")
    assert "HF_TOKEN" not in kwargs["env"]


def test_build_deploy_kwargs_extra_env_passes_through() -> None:
    spec = AutoTuneBaselineSpec(env={"FOO": "bar"})
    kwargs = spec.build_deploy_kwargs(name="run-1")
    assert kwargs["env"]["FOO"] == "bar"


def test_build_deploy_kwargs_gpu_models_and_spot_threaded() -> None:
    spec = AutoTuneBaselineSpec()
    kwargs = spec.build_deploy_kwargs(
        name="run-1", gpu_models=["A100", "H100"], spot=True,
    )
    assert kwargs["gpu_models"] == ["A100", "H100"]
    assert kwargs["spot"] is True


def test_build_deploy_kwargs_health_probes_present() -> None:
    """Generous startup probe matches the campaign-side rationale: large
    CUDA image + vllm install + model download all happen before the
    HTTP server's first response would otherwise pass health checks."""
    import basilica

    spec = AutoTuneBaselineSpec()
    kwargs = spec.build_deploy_kwargs(name="run-1")
    health = kwargs["health_check"]
    assert isinstance(health, basilica.HealthCheckConfig)
    assert health.startup is not None
    # ~10 min startup grace; same shape as CampaignSpec.
    assert health.startup.failure_threshold * health.startup.period_seconds >= 600


def test_field_overrides_thread_through() -> None:
    spec = AutoTuneBaselineSpec(
        model="meta-llama/Llama-3.1-70B-Instruct",
        vllm_version="0.20.0",
        tp=4,
        input_len=4000,
        output_len=200,
        max_model_len=4300,
        max_latency_allowed_ms=2000,
        num_seqs_list="64 128",
        num_batched_tokens_list="8192 16384",
        min_cache_hit_pct=60,
    )
    src = spec.build_source()
    assert "MODEL = 'meta-llama/Llama-3.1-70B-Instruct'" in src
    assert "VLLM_VERSION = '0.20.0'" in src
    assert "TP = 4" in src
    assert "INPUT_LEN = 4000" in src
    assert "OUTPUT_LEN = 200" in src
    assert "MAX_MODEL_LEN = 4300" in src
    assert "MAX_LATENCY_ALLOWED_MS = 2000" in src
    assert "NUM_SEQS_LIST = '64 128'" in src
    assert "NUM_BATCHED_TOKENS_LIST = '8192 16384'" in src
    assert "MIN_CACHE_HIT_PCT = 60" in src
