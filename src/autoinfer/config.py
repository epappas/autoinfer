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


class QualityGateConfig(_Base):
    """Live reference-replica quality gate (P8, C9)."""

    replica_uri: str = Field(description="HTTP endpoint of the FP16 reference replica.")
    prompts_path: Path
    smoke_prompts: int = Field(ge=1, default=100)
    full_prompts: int = Field(ge=1, default=500)
    batch_sizes: tuple[int, ...] = (1, 8, 64)
    max_kl: float = Field(gt=0.0, description="Reject if per-token KL exceeds this.")


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


class SurrogateConfig(_Base):
    kind: SurrogateKind = "tpe"
    seed: int = 0


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
    candidate_port: int = Field(ge=1024, le=65535, default=8000)
    startup_timeout_s: int = Field(ge=30, default=600)


class L2TopologyConfig(_Base):
    gpu_classes: tuple[str, ...] = ("H100",)
    max_trials: int = Field(ge=0, default=0)


class L3KernelConfig(_Base):
    max_trials: int = Field(ge=0, default=0)


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
