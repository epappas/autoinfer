from __future__ import annotations

from pathlib import Path

from autoinfer.harness.driver import DriverResult
from autoinfer.harness.gate import GateResult
from autoinfer.layers.l1_engine.adapter import compose_measurement


def test_compose_measurement_maps_all_fields() -> None:
    driver = DriverResult(
        tokens_per_sec=1234.5,
        request_throughput=10.0,
        ttft_ms={"p50": 100.0, "p95": 150.0, "p99": 200.0},
        tpot_ms={"p50": 20.0, "p95": 30.0, "p99": 45.0},
        goodput_req_per_sec=9.0,
        raw={},
    )
    gate = GateResult(
        mean_kl=0.01,
        max_kl=0.02,
        per_prompt_kl=(0.01, 0.01),
        batch_invariant=True,
    )
    m = compose_measurement(driver, gate, peak_hbm_gb=42.5)
    assert m.tokens_per_sec == 1234.5
    assert m.ttft_p99_ms == 200.0
    assert m.tpot_p99_ms == 45.0
    assert m.peak_hbm_gb == 42.5
    assert m.kl_divergence == 0.01
    assert m.extra["ttft_p50_ms"] == 100.0
    assert m.extra["tpot_p50_ms"] == 20.0
    assert m.extra["goodput"] == 9.0
    assert m.extra["max_kl"] == 0.02


def test_compose_measurement_handles_missing_percentiles() -> None:
    driver = DriverResult(
        tokens_per_sec=500.0,
        request_throughput=1.0,
        ttft_ms={},
        tpot_ms={},
        goodput_req_per_sec=1.0,
        raw={},
    )
    gate = GateResult(
        mean_kl=0.0, max_kl=0.0, per_prompt_kl=(0.0,), batch_invariant=True
    )
    m = compose_measurement(driver, gate, peak_hbm_gb=0.0)
    assert m.ttft_p99_ms == 0.0
    assert m.tpot_p99_ms == 0.0


def test_compose_measurement_includes_kl_percentiles() -> None:
    """T-25: per-prompt KL percentiles in extra so post-run analysis
    can spot outlier-driven mean drift."""
    driver = DriverResult(
        tokens_per_sec=100.0,
        request_throughput=1.0,
        ttft_ms={"p99": 100.0, "p50": 50.0},
        tpot_ms={"p99": 30.0, "p50": 20.0},
        goodput_req_per_sec=1.0,
        raw={},
    )
    # 20 small KLs + 1 outlier
    per_prompt = tuple([0.1] * 19 + [10.0])
    gate = GateResult(
        mean_kl=sum(per_prompt) / len(per_prompt),
        max_kl=10.0,
        per_prompt_kl=per_prompt,
        batch_invariant=True,
    )
    m = compose_measurement(driver, gate, peak_hbm_gb=10.0)
    assert m.extra["kl_min"] == 0.1
    assert m.extra["kl_p50"] == 0.1
    assert m.extra["kl_p99"] == 10.0
    # max_kl already there from prior code; check it survived
    assert m.extra["max_kl"] == 10.0


def test_compose_measurement_kl_percentiles_handle_empty() -> None:
    """If gate produced no per-prompt KL, percentile fields are absent;
    other extras still present."""
    driver = DriverResult(
        tokens_per_sec=500.0,
        request_throughput=1.0,
        ttft_ms={"p99": 100.0},
        tpot_ms={"p99": 30.0},
        goodput_req_per_sec=1.0,
        raw={},
    )
    gate = GateResult(mean_kl=0.0, max_kl=0.0, per_prompt_kl=(), batch_invariant=True)
    m = compose_measurement(driver, gate, peak_hbm_gb=0.0)
    assert "kl_p99" not in m.extra
    assert "max_kl" in m.extra


def test_query_gpu_memory_returns_none_or_float() -> None:
    """nvidia-smi may not exist on CI; we just assert the function doesn't crash."""
    from autoinfer.layers.l1_engine.adapter import query_gpu_memory_used_gb

    result = query_gpu_memory_used_gb()
    assert result is None or isinstance(result, float)


def test_l1_adapter_exports_and_constructs() -> None:
    """L1EngineAdapter can be constructed without hitting GPU. Startup is separate."""
    from autoinfer.layers.l1_engine import L1EngineAdapter, load_catalog

    catalog = load_catalog(
        Path(__file__).parent.parent / "src/autoinfer/layers/l1_engine/knobs.yaml"
    )
    adapter = L1EngineAdapter(
        model="Qwen/Qwen3-8B",
        catalog=catalog,
        trace_path=Path("/tmp/trace.jsonl"),
        reference_uri="http://localhost:8001",
        quality_prompts=["hi"],
        max_kl=0.05,
        result_dir=Path("/tmp/runs"),
    )
    assert adapter.layer_name == "l1_engine"
    surface = adapter.surface()
    assert "max_num_batched_tokens" in surface


def test_l1_adapter_rejects_constraint_violation_without_subprocess() -> None:
    """Canary: a compat-rule violation is classified as STARTUP failure pre-spawn.

    This is the step-5 canary from docs/runbook-iteration-zero.md.
    The adapter must reject without ever calling ``subprocess.Popen``,
    so it runs on CPU with no vllm dependency and is safe to use as
    the first CI-gate against regressions in the constraint pipeline.
    """
    from autoinfer.harness.failure import FailureKind
    from autoinfer.layers import TrialInput
    from autoinfer.layers.l1_engine import L1EngineAdapter, load_catalog

    catalog = load_catalog(
        Path(__file__).parent.parent / "src/autoinfer/layers/l1_engine/knobs.yaml"
    )
    adapter = L1EngineAdapter(
        model="Qwen/Qwen3-8B",
        catalog=catalog,
        trace_path=Path("/tmp/trace.jsonl"),
        reference_uri="http://localhost:8001",
        quality_prompts=["hi"],
        max_kl=0.05,
        result_dir=Path("/tmp/runs"),
    )

    bad_config = {
        "max_num_batched_tokens": 2048,
        "max_num_seqs": 128,
        "enable_chunked_prefill": True,
        "block_size": 16,
        "kv_cache_dtype": "fp8",
        "gpu_memory_utilization": 0.90,
        "enable_prefix_caching": False,
        "attention_backend": "XFORMERS",
        "num_scheduler_steps": 1,
        "swap_space": 4,
        "dtype": "auto",
        "quantization": "none",
    }
    out = adapter.run(TrialInput(trial_id="canary-0", config=bad_config))

    assert out.measurement is None
    assert out.failure is not None
    assert out.failure.kind is FailureKind.STARTUP
    assert "kv_fp8_requires_compatible_backend" in out.failure.message
