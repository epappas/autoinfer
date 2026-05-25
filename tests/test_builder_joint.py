"""Joint multi-layer runner assembly tests.

Uses L1 + L3 pair — L3 has no external deps, L1 builds an adapter
from pure data (vllm subprocess is only spawned when ``.run()`` is
invoked, which these tests never do). Basilica-backed L2 is covered
separately.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from autoinfer.builder import build_runner
from autoinfer.config import RunConfig

_REPO_ROOT = Path(__file__).parent.parent
_L1_CATALOG = _REPO_ROOT / "src/autoinfer/layers/l1_engine/knobs.yaml"
_L3_CATALOG = _REPO_ROOT / "src/autoinfer/layers/l3_kernel/knobs.yaml"


def _raw_joint(tmp_path: Path) -> dict[str, Any]:
    trace = tmp_path / "trace.jsonl"
    trace.write_text('{"prompt": "hi"}\n')
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"prompt": "hello"}\n{"prompt": "world"}\n')
    return {
        "name": "joint-smoke",
        "harness": {
            "driver": {
                "trace_path": str(trace),
                "duration_s": 60,
                "slo_ttft_p99_ms": 500.0,
                "slo_tpot_p99_ms": 50.0,
            },
            "gate": {
                "replica_uri": "http://127.0.0.1:8001",
                "prompts_path": str(prompts),
                "smoke_prompts": 2,
                "max_kl": 2.0,
                "calibrate_self_kl": False,
            },
            "ledger": {
                "output_dir": str(tmp_path / "runs"),
                "pareto_axes": ["tokens_per_sec", "tpot_p99_ms"],
            },
        },
        "policy": {
            "warmstart": {
                "provider": "deterministic",
                "llm_model": "stub",
                "n_configs": 2,
            },
            "surrogate": {"kind": "tpe", "seed": 0},
        },
        "layers": {
            "l1_engine": {
                "model": "Qwen/Qwen3-8B",
                "knobs_path": str(_L1_CATALOG),
                "max_trials": 4,
            },
            "l3_kernel": {
                "knobs_path": str(_L3_CATALOG),
                "max_trials": 3,
            },
        },
    }


def test_joint_runner_registers_both_layers(tmp_path: Path) -> None:
    cfg = RunConfig.model_validate(_raw_joint(tmp_path))
    runner, ledger = build_runner(cfg)
    assert set(runner.scheduler.specs.keys()) == {"l1_engine", "l3_kernel"}
    # ledger is shared — same instance, one output dir
    assert ledger is runner.ledger
    assert ledger._dir == tmp_path / "runs"  # noqa: SLF001


def test_joint_runner_respects_per_layer_max_trials(tmp_path: Path) -> None:
    cfg = RunConfig.model_validate(_raw_joint(tmp_path))
    runner, _ = build_runner(cfg)
    assert runner.scheduler.specs["l1_engine"].max_trials == 4
    assert runner.scheduler.specs["l3_kernel"].max_trials == 3


def test_joint_override_caps_every_layer(tmp_path: Path) -> None:
    cfg = RunConfig.model_validate(_raw_joint(tmp_path))
    runner, _ = build_runner(cfg, max_trials_override=2)
    assert runner.scheduler.specs["l1_engine"].max_trials == 2
    assert runner.scheduler.specs["l3_kernel"].max_trials == 2


def test_joint_per_layer_override_wins_over_uniform(tmp_path: Path) -> None:
    cfg = RunConfig.model_validate(_raw_joint(tmp_path))
    runner, _ = build_runner(
        cfg,
        max_trials_override=10,
        per_layer_overrides={"l1_engine": 2},
    )
    assert runner.scheduler.specs["l1_engine"].max_trials == 2
    # uniform fallback for layers without explicit per-layer override
    assert runner.scheduler.specs["l3_kernel"].max_trials == 10


def test_joint_per_layer_override_alone(tmp_path: Path) -> None:
    cfg = RunConfig.model_validate(_raw_joint(tmp_path))
    runner, _ = build_runner(
        cfg, per_layer_overrides={"l1_engine": 1, "l3_kernel": 2},
    )
    assert runner.scheduler.specs["l1_engine"].max_trials == 1
    assert runner.scheduler.specs["l3_kernel"].max_trials == 2


def test_joint_emits_config_loaded_with_all_layers(tmp_path: Path) -> None:
    cfg = RunConfig.model_validate(_raw_joint(tmp_path))
    build_runner(cfg)
    events_file = tmp_path / "runs" / "events.jsonl"
    assert events_file.exists()
    import json

    lines = [json.loads(ln) for ln in events_file.read_text().splitlines() if ln.strip()]
    config_loaded = next(ln for ln in lines if ln.get("type") == "config_loaded")
    assert config_loaded["layers"] == ["l1_engine", "l3_kernel"]
    assert len(config_loaded["per_layer"]) == 2


def test_joint_no_operator_when_not_configured(tmp_path: Path) -> None:
    cfg = RunConfig.model_validate(_raw_joint(tmp_path))
    runner, _ = build_runner(cfg)
    assert runner.operator is None


def test_single_layer_l1_still_builds(tmp_path: Path) -> None:
    raw = _raw_joint(tmp_path)
    raw["layers"].pop("l3_kernel")  # type: ignore[attr-defined]
    cfg = RunConfig.model_validate(raw)
    runner, _ = build_runner(cfg)
    assert list(runner.scheduler.specs.keys()) == ["l1_engine"]


def test_runner_objective_axis_defaults_to_tokens_per_sec(tmp_path: Path) -> None:
    """T-34: without slo_e2e_p99_ms set, legacy throughput-axis path stays."""
    cfg = RunConfig.model_validate(_raw_joint(tmp_path))
    runner, _ = build_runner(cfg)
    assert runner.objective_axis == "tokens_per_sec"
    # L1 adapter should not have an SLO either.
    l1_adapter = runner.scheduler.specs["l1_engine"].adapter
    assert l1_adapter.goodput_slo_ms is None  # type: ignore[attr-defined]


def test_runner_objective_axis_switches_to_goodput_when_e2e_slo_set(
    tmp_path: Path,
) -> None:
    """T-34: setting harness.driver.slo_e2e_p99_ms flips the runner to
    goodput-axis and wires the L1 adapter with --goodput SLOs."""
    raw = _raw_joint(tmp_path)
    raw["harness"]["driver"]["slo_e2e_p99_ms"] = 500.0
    cfg = RunConfig.model_validate(raw)
    runner, _ = build_runner(cfg)
    assert runner.objective_axis == "goodput_req_per_sec"
    l1_adapter = runner.scheduler.specs["l1_engine"].adapter
    assert l1_adapter.goodput_slo_ms == {  # type: ignore[attr-defined]
        "TTFT": 500.0,
        "TPOT": 50.0,
        "E2E": 500.0,
    }


def test_l1_spec_event_records_objective_and_slo(tmp_path: Path) -> None:
    """Run-loaded event must surface the SLO configuration so post-hoc
    analysis can tell goodput-mode runs apart from throughput-mode runs."""
    raw = _raw_joint(tmp_path)
    raw["harness"]["driver"]["slo_e2e_p99_ms"] = 500.0
    cfg = RunConfig.model_validate(raw)
    build_runner(cfg)
    events_file = tmp_path / "runs" / "events.jsonl"
    import json

    lines = [json.loads(ln) for ln in events_file.read_text().splitlines() if ln.strip()]
    config_loaded = next(ln for ln in lines if ln.get("type") == "config_loaded")
    l1_event = next(
        e for e in config_loaded["per_layer"] if e["layer"] == "l1_engine"
    )
    assert l1_event["objective_axis"] == "goodput_req_per_sec"
    assert l1_event["goodput_slo_ms"] == {"TTFT": 500.0, "TPOT": 50.0, "E2E": 500.0}


def test_single_layer_l3_still_builds(tmp_path: Path) -> None:
    raw = _raw_joint(tmp_path)
    raw["layers"].pop("l1_engine")  # type: ignore[attr-defined]
    cfg = RunConfig.model_validate(raw)
    runner, _ = build_runner(cfg)
    assert list(runner.scheduler.specs.keys()) == ["l3_kernel"]


def test_joint_runs_end_to_end_with_l3_only_adapter(tmp_path: Path) -> None:
    """L1 adapter can't execute without vLLM, but L3 can — wire the L1
    spec but only let the scheduler pick L3 by exhausting L1's budget
    with max_trials=0... actually max_trials must be >=1 per config.

    Instead: build L3-only and confirm the runner completes trials, to
    verify the refactor didn't break the single-layer runnable path.
    L3 entries are pareto_eligible=False (kernel ops/sec aren't unit-
    comparable to token throughput) so the joint frontier excludes
    them; the per-layer frontier still includes the L3 best."""
    raw = _raw_joint(tmp_path)
    raw["layers"].pop("l1_engine")  # type: ignore[attr-defined]
    raw["layers"]["l3_kernel"]["max_trials"] = 3  # type: ignore[index]
    cfg = RunConfig.model_validate(raw)
    runner, ledger = build_runner(cfg)
    runner.run()
    entries = ledger.entries()
    assert len(entries) == 3
    # L3 reference kernels pass correctness; at least one kept
    kept = [e for e in entries if e.kept]
    assert len(kept) >= 1
    # joint Pareto excludes L3 (ineligible); per-layer frontier sees it
    by_layer = ledger.pareto_front_by_layer()
    assert "l3_kernel" in by_layer
    assert len(by_layer["l3_kernel"]) >= 1
