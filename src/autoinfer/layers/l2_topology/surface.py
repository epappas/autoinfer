"""L2 topology search-surface loader and deploy_vllm kwarg materializer.

Same shape as ``l1_engine.surface`` but maps knobs to
``BasilicaClient.deploy_vllm`` kwargs instead of vLLM CLI flags. Pure
logic; no SDK calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_KNOB_TYPES = {"int", "float", "categorical", "bool"}


@dataclass(frozen=True)
class L2KnobSpec:
    name: str
    type: str
    default: Any
    low: float | int | None = None
    high: float | int | None = None
    values: tuple[Any, ...] | None = None
    basilica_deploy_vllm_kwarg: str | None = None
    coupled_with: tuple[str, ...] = ()
    axis: str | None = None
    description: str = ""


@dataclass(frozen=True)
class L2CompatRule:
    rule: str
    description: str
    when_knob: str
    when_values: tuple[Any, ...]
    requires_knob: str
    requires_values: tuple[Any, ...]


@dataclass(frozen=True)
class L2Catalog:
    knobs: dict[str, L2KnobSpec]
    constraints: tuple[L2CompatRule, ...] = field(default_factory=tuple)


def load_catalog(path: Path) -> L2Catalog:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or "knobs" not in raw:
        raise ValueError(f"{path} is not a valid L2 catalog (missing 'knobs' mapping)")
    knobs = {name: _parse_knob(name, spec) for name, spec in raw["knobs"].items()}
    constraints = tuple(_parse_constraint(c) for c in raw.get("constraints") or [])
    return L2Catalog(knobs=knobs, constraints=constraints)


def _parse_knob(name: str, spec: dict[str, Any]) -> L2KnobSpec:
    knob_type = spec.get("type")
    if knob_type not in _KNOB_TYPES:
        raise ValueError(f"knob {name!r} has unknown type {knob_type!r}")
    values = spec.get("values")
    return L2KnobSpec(
        name=name,
        type=knob_type,
        default=spec.get("default"),
        low=spec.get("low"),
        high=spec.get("high"),
        values=tuple(values) if values is not None else None,
        basilica_deploy_vllm_kwarg=spec.get("basilica_deploy_vllm_kwarg"),
        coupled_with=tuple(spec.get("coupled_with") or ()),
        axis=spec.get("axis"),
        description=spec.get("description", ""),
    )


def _parse_constraint(spec: dict[str, Any]) -> L2CompatRule:
    return L2CompatRule(
        rule=spec["rule"],
        description=spec.get("description", ""),
        when_knob=spec["when_knob"],
        when_values=tuple(spec["when_values"]),
        requires_knob=spec["requires_knob"],
        requires_values=tuple(spec["requires_values"]),
    )


def to_surrogate_surface(catalog: L2Catalog) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, knob in catalog.knobs.items():
        if knob.type == "int":
            out[name] = {"type": "int", "low": knob.low, "high": knob.high}
        elif knob.type == "float":
            out[name] = {"type": "float", "low": knob.low, "high": knob.high}
        elif knob.type == "categorical":
            if knob.values is None:
                raise ValueError(f"categorical knob {name!r} missing 'values'")
            out[name] = {"type": "categorical", "values": list(knob.values)}
        elif knob.type == "bool":
            out[name] = {"type": "categorical", "values": [True, False]}
    return out


def defaults(catalog: L2Catalog) -> dict[str, Any]:
    return {name: knob.default for name, knob in catalog.knobs.items()}


def config_to_deploy_kwargs(
    config: dict[str, Any], catalog: L2Catalog
) -> dict[str, Any]:
    """Translate a trial config into ``deploy_vllm`` kwargs.

    - ``gpu_type`` becomes ``gpu_models=[value]``.
    - ``gpu_count`` is also materialized as ``tensor_parallel_size`` for
      single-node TP (the common case at iteration-zero L2).
    - bools and numerics pass through as-is.
    """
    out: dict[str, Any] = {}
    for name, value in config.items():
        knob = catalog.knobs.get(name)
        if knob is None or not knob.basilica_deploy_vllm_kwarg:
            continue
        kwarg = knob.basilica_deploy_vllm_kwarg
        if kwarg == "gpu_models":
            out["gpu_models"] = [str(value)]
        else:
            out[kwarg] = value
    gpu_count = config.get("gpu_count")
    if gpu_count is not None:
        out.setdefault("tensor_parallel_size", int(gpu_count))
    return out
