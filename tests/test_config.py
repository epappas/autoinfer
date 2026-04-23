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
