from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from autoinfer.config import RunConfig, load_config


def _minimal_raw() -> dict[str, object]:
    return {
        "name": "test-run",
        "harness": {
            "driver": {
                "trace_path": "trace.jsonl",
                "duration_s": 60,
                "slo_ttft_p99_ms": 500.0,
                "slo_tpot_p99_ms": 50.0,
            },
            "gate": {
                "replica_uri": "http://localhost:8001",
                "prompts_path": "prompts.jsonl",
                "max_kl": 0.05,
            },
            "ledger": {"output_dir": "./runs/test"},
        },
        "policy": {
            "warmstart": {"llm_model": "claude-opus-4-7"},
        },
        "layers": {
            "l1_engine": {
                "model": "Qwen/Qwen3-8B",
                "knobs_path": "knobs.yaml",
            },
        },
    }


def test_minimal_config_validates() -> None:
    run = RunConfig.model_validate(_minimal_raw())
    assert run.name == "test-run"
    assert run.layers.l1_engine is not None
    assert run.layers.l1_engine.model == "Qwen/Qwen3-8B"
    assert run.layers.l1_engine.max_trials == 200
    assert run.target.kind == "local"
    assert run.policy.surrogate.kind == "tpe"
    assert run.harness.gate.batch_sizes == (1, 8, 64)


def test_no_layers_rejected() -> None:
    raw = _minimal_raw()
    raw["layers"] = {}
    with pytest.raises(ValidationError):
        RunConfig.model_validate(raw)


def test_extra_field_rejected() -> None:
    raw = _minimal_raw()
    raw["unknown_top_level"] = 42
    with pytest.raises(ValidationError):
        RunConfig.model_validate(raw)


def test_negative_duration_rejected() -> None:
    raw = _minimal_raw()
    raw["harness"]["driver"]["duration_s"] = 0  # type: ignore[index]
    with pytest.raises(ValidationError):
        RunConfig.model_validate(raw)


def test_yaml_roundtrip(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(_minimal_raw()))
    run = load_config(cfg_path)
    assert run.name == "test-run"


def test_yaml_non_mapping_rejected(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("- a\n- b\n")
    with pytest.raises(ValueError):
        load_config(cfg_path)


def test_slo_e2e_p99_ms_defaults_to_none() -> None:
    """T-34: optional E2E SLO is None by default — preserves legacy
    throughput-mode runs."""
    run = RunConfig.model_validate(_minimal_raw())
    assert run.harness.driver.slo_e2e_p99_ms is None


def test_slo_e2e_p99_ms_accepts_positive_value() -> None:
    """T-34: E2E SLO set on the driver flips the run into goodput mode."""
    raw = _minimal_raw()
    raw["harness"]["driver"]["slo_e2e_p99_ms"] = 500.0  # type: ignore[index]
    run = RunConfig.model_validate(raw)
    assert run.harness.driver.slo_e2e_p99_ms == 500.0


def test_slo_e2e_p99_ms_zero_or_negative_rejected() -> None:
    """T-34: gt=0 constraint matches the other SLO fields."""
    raw = _minimal_raw()
    raw["harness"]["driver"]["slo_e2e_p99_ms"] = 0.0  # type: ignore[index]
    with pytest.raises(ValidationError):
        RunConfig.model_validate(raw)


def test_bench_seed_defaults_to_none() -> None:
    """T-35: deterministic bench seed opt-in (default unset)."""
    run = RunConfig.model_validate(_minimal_raw())
    assert run.harness.driver.bench_seed is None


def test_bench_seed_accepts_non_negative_int() -> None:
    raw = _minimal_raw()
    raw["harness"]["driver"]["bench_seed"] = 0  # type: ignore[index]
    assert RunConfig.model_validate(raw).harness.driver.bench_seed == 0
    raw["harness"]["driver"]["bench_seed"] = 42  # type: ignore[index]
    assert RunConfig.model_validate(raw).harness.driver.bench_seed == 42


def test_bench_seed_negative_rejected() -> None:
    raw = _minimal_raw()
    raw["harness"]["driver"]["bench_seed"] = -1  # type: ignore[index]
    with pytest.raises(ValidationError):
        RunConfig.model_validate(raw)
