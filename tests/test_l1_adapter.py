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
