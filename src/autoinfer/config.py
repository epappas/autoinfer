"""Typed run-configuration schemas.

Loading flow: YAML file -> dict -> ``RunConfig.model_validate``. Every
adapter and policy consumes ``RunConfig`` or a sub-model; no module
parses YAML directly outside this file.

All models forbid unknown fields. Extra keys in config files fail
loudly rather than silently (P12: falsifiable).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

LayerName = Literal["l1_engine", "l2_topology", "l3_kernel"]
SurrogateKind = Literal["tpe", "cmaes", "bohb", "random"]
TargetKind = Literal["local", "basilica"]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)


class DriverConfig(_Base):
    """Workload driver that replays a trace through the engine."""

    trace_path: Path
    duration_s: int = Field(gt=0)
    slo_ttft_p99_ms: float = Field(gt=0.0)
    slo_tpot_p99_ms: float = Field(gt=0.0)
    dataset_name: Literal["random", "sharegpt", "custom", "sonnet"] = "random"
    num_prompts: int = Field(ge=1, default=64)


class QualityGateConfig(_Base):
    """Live reference-replica quality gate (P8, C9)."""

    replica_uri: str = Field(description="HTTP endpoint of the FP16 reference replica.")
    prompts_path: Path
    smoke_prompts: int = Field(ge=1, default=100)
    full_prompts: int = Field(ge=1, default=500)
    batch_sizes: tuple[int, ...] = (1, 8, 64)
    max_kl: float = Field(gt=0.0, description="Reject if per-token KL exceeds this.")
    calibrate_self_kl: bool = Field(
        default=False,
        description="If true, run reference-vs-self gate at startup and override max_kl.",
    )
    calibration_multiplier: float = Field(
        default=10.0,
        gt=1.0,
        description="max_kl = multiplier * self-self p95 after calibration.",
    )


class LedgerConfig(_Base):
    output_dir: Path
    pareto_axes: tuple[str, ...] = ("tokens_per_sec", "tpot_p99_ms", "peak_hbm_gb")


class HarnessConfig(_Base):
    """Shared substrate. Frozen per run (P10)."""

    driver: DriverConfig
    gate: QualityGateConfig
    ledger: LedgerConfig


class WarmstartConfig(_Base):
    provider: Literal["anthropic", "openai_compatible", "deterministic"] = "deterministic"
    llm_model: str = Field(description="Policy LLM id, e.g. 'claude-opus-4-7'.")
    base_url: str | None = Field(default=None, description="Used when provider='openai_compatible'.")
    api_key_env: str | None = Field(default=None, description="Env var name holding the API key.")
    n_configs: int = Field(ge=1, le=100, default=15)
    seed_configs: list[dict[str, Any]] | None = Field(
        default=None,
        description="Configs for provider='deterministic'. Required in that mode.",
    )
    hardware_notes: str | None = Field(
        default=None,
        description=(
            "Hardware + compatibility facts threaded into the LLM prompt. "
            "Freeform prose. Example: 'GPU: 2x RTX A6000 (Ampere, 48GB). "
            "FP8 KV unsupported. No NVLink; PCIe only.'"
        ),
    )


class SurrogateConfig(_Base):
    kind: SurrogateKind = "tpe"
    seed: int = 0
    feasibility_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "If > 0, wrap the perf surrogate in ConstrainedOptunaSurrogate "
            "with a FeasibilityModel that rejects candidates whose nearest-"
            "neighbor history is too failure-dense. 0.0 disables (default, "
            "preserves prior behavior). 0.4 is a sane starting value for "
            "infeasibility-rich surfaces like L1 engine knobs."
        ),
    )
    feasibility_max_resamples: int = Field(
        default=8,
        ge=1,
        description="Resampling budget per suggest() before falling back.",
    )
    feasibility_k: int = Field(
        default=3,
        ge=1,
        description="Nearest-neighbor count for the feasibility classifier.",
    )
    feasibility_min_observations: int = Field(
        default=4,
        ge=0,
        description=(
            "Below this many recorded trials, the classifier returns 1.0 "
            "(no constraint signal yet). Default matches typical warmstart "
            "batch size."
        ),
    )


class FidelityConfig(_Base):
    """Hyperband/BOHB-style multi-fidelity scheduler."""

    rungs: tuple[int, ...] = (100, 500)
    eta: int = Field(ge=2, default=3)


class OperatorConfig(_Base):
    """LLM proposal operator invoked on stall (P7)."""

    llm_model: str
    cadence: int = Field(ge=1, default=10)


class PolicyConfig(_Base):
    """Hybrid LLM + classical-surrogate policy stack (P7)."""

    warmstart: WarmstartConfig
    surrogate: SurrogateConfig = Field(default_factory=SurrogateConfig)
    fidelity: FidelityConfig = Field(default_factory=FidelityConfig)
    operator: OperatorConfig | None = None


class L1EngineConfig(_Base):
    model: str = "Qwen/Qwen3-8B"
    knobs_path: Path
    max_trials: int = Field(ge=1, default=200)
    reserve_cap: int = Field(
        ge=0,
        default=0,
        description=(
            "Extra trials granted on cross-layer stale invalidation. "
            "Default 0 preserves single-pass behavior; set >0 to enable "
            "second-pass re-search after a deeper layer dominates."
        ),
    )
    candidate_port: int = Field(ge=1024, le=65535, default=8000)
    startup_timeout_s: int = Field(ge=30, default=600)


class L2TopologyConfig(_Base):
    model: str = "Qwen/Qwen3-8B"
    knobs_path: Path
    max_trials: int = Field(ge=1, default=20)
    reserve_cap: int = Field(ge=0, default=0)
    memory: str = "64Gi"
    ttl_seconds: int = Field(ge=300, default=3600)
    deploy_timeout_s: int = Field(ge=60, default=1200)


class L3KernelConfig(_Base):
    model: str = "Qwen/Qwen3-8B"
    """Required for mode='vllm'. Ignored in mode='cpu'."""

    knobs_path: Path
    max_trials: int = Field(ge=1, default=6)
    reserve_cap: int = Field(ge=0, default=0)
    atol: float = Field(gt=0.0, default=1e-3)
    rtol: float = Field(gt=0.0, default=1e-3)
    perf_repeats: int = Field(ge=1, default=5)
    warmup_runs: int = Field(ge=0, default=2)
    mode: Literal["cpu", "vllm"] = Field(
        default="cpu",
        description=(
            "'cpu' uses L3KernelAdapter — times the kernel in isolation, "
            "produces ops/sec, sets pareto_eligible=False. Useful for "
            "fast correctness validation and dev workflows. "
            "'vllm' uses L3VllmKernelAdapter — ships the kernel into "
            "vLLM via runtime monkeypatch, runs vllm bench serve, and "
            "produces real end-to-end token throughput in the same "
            "units as L1/L2 (pareto_eligible=True). The vllm mode is "
            "the load-bearing thesis-grade L3 path. Default 'cpu' so "
            "tests + dev work without a GPU; campaign configs should "
            "opt into 'vllm'."
        ),
    )
    candidate_port: int = Field(ge=1024, le=65535, default=8200)
    """Distinct from L1's default 8000 so a joint campaign running both
    L1 and L3-vllm in the same container doesn't collide."""

    startup_timeout_s: int = Field(ge=30, default=600)
    extra_vllm_args: list[str] = Field(
        default_factory=list,
        description=(
            "Additional ``vllm serve`` CLI args appended after the "
            "canonical model + port prefix when mode='vllm'."
        ),
    )
    paired_control: bool = Field(
        default=False,
        description=(
            "T-27. When True, the L3 warmstart emits interleaved "
            "(reference, llm-novel) pairs at identical "
            "(target_op, dtype, shape_regime) cells so each LLM-novel "
            "kernel has a same-cell reference control to compare against. "
            "Requires a non-deterministic warmstart provider (the LLM "
            "must actually generate the novel half of each pair). "
            "Default False to keep existing campaigns reproducible."
        ),
    )
    paired_control_cells: list[tuple[str, str, str]] | None = Field(
        default=None,
        description=(
            "T-26-followup. When ``paired_control=True``, override the "
            "default 6-cell breadth list with an explicit list of "
            "(target_op, dtype, shape_regime) tuples. Campaign 03 uses "
            "this to narrow to 3 cells (+14.7% winner, −11% loser, tie "
            "from C02) for a focused replication study."
        ),
    )
    paired_control_replicates: int = Field(
        default=1,
        ge=1,
        le=20,
        description=(
            "T-26-followup. Number of paired (ref, novel) replicates "
            "per cell. Default 1 = same as Campaign 02 (no replication). "
            "Setting > 1 enables a real N>1 paired A/B per cell so "
            "single-shot LLM emission luck can be distinguished from "
            "robust kernel-novelty wins."
        ),
    )
    warmstart_n: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description=(
            "Per-layer warmstart batch size override. When None, falls "
            "back to ``policy.warmstart.n_configs``. Useful when "
            "paired_control=True so L3 can warmstart 12 trials (6 "
            "ref/novel pairs) without forcing L1 / L2 to also expand "
            "their warmstart batch. T-27."
        ),
    )


class LayersConfig(_Base):
    l1_engine: L1EngineConfig | None = None
    l2_topology: L2TopologyConfig | None = None
    l3_kernel: L3KernelConfig | None = None

    @model_validator(mode="after")
    def _at_least_one_enabled(self) -> LayersConfig:
        enabled = [self.l1_engine, self.l2_topology, self.l3_kernel]
        if not any(e is not None for e in enabled):
            raise ValueError("at least one of l1_engine/l2_topology/l3_kernel must be configured")
        return self


class TargetConfig(_Base):
    kind: TargetKind = "local"
    basilica_tier: str | None = None


class RunConfig(_Base):
    """Top-level run configuration."""

    name: str = Field(min_length=1)
    harness: HarnessConfig
    policy: PolicyConfig
    layers: LayersConfig
    target: TargetConfig = Field(default_factory=TargetConfig)


def load_config(path: Path) -> RunConfig:
    """Load YAML and validate into a ``RunConfig``."""
    import yaml

    with path.open() as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"config file {path} must contain a YAML mapping at top level")
    return RunConfig.model_validate(raw)
