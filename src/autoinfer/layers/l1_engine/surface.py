"""L1 engine search-surface loader and vLLM arg materializer.

``knobs.yaml`` is the canonical catalog; this module translates it into
three shapes:

- ``KnobSpec`` dataclasses for structured access.
- A dict consumed by ``OptunaSurrogate`` describing the search space.
- A concrete CLI + env pair that starts ``vllm serve`` with the values
  from a surrogate-proposed config.

Pure logic; no subprocess or network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_KNOB_TYPES = {"int", "float", "categorical", "bool"}


@dataclass(frozen=True)
class KnobSpec:
    name: str
    type: str
    default: Any
    low: float | int | None = None
    high: float | int | None = None
    step: float | int | None = None
    values: tuple[Any, ...] | None = None
    vllm_cli: str | None = None
    vllm_cli_enable: str | None = None
    vllm_cli_disable: str | None = None
    vllm_env: str | None = None
    coupled_with: tuple[str, ...] = ()
    axis: str | None = None
    description: str = ""


@dataclass(frozen=True)
class CompatRule:
    rule: str
    description: str
    when_knob: str
    when_values: tuple[Any, ...]
    requires_knob: str
    requires_values: tuple[Any, ...]


@dataclass(frozen=True)
class KnobCatalog:
    knobs: dict[str, KnobSpec]
    constraints: tuple[CompatRule, ...] = field(default_factory=tuple)


def load_catalog(path: Path) -> KnobCatalog:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or "knobs" not in raw:
        raise ValueError(f"{path} is not a valid knob catalog (missing 'knobs' mapping)")
    knobs = {name: _parse_knob(name, spec) for name, spec in raw["knobs"].items()}
    constraints = tuple(_parse_constraint(c) for c in raw.get("constraints") or [])
    return KnobCatalog(knobs=knobs, constraints=constraints)


def _parse_knob(name: str, spec: dict[str, Any]) -> KnobSpec:
    if not isinstance(spec, dict):
        raise ValueError(f"knob {name!r} spec is not a mapping")
    knob_type = spec.get("type")
    if knob_type not in _KNOB_TYPES:
        raise ValueError(f"knob {name!r} has unknown type {knob_type!r}")
    values = spec.get("values")
    return KnobSpec(
        name=name,
        type=knob_type,
        default=spec.get("default"),
        low=spec.get("low"),
        high=spec.get("high"),
        step=spec.get("step"),
        values=tuple(values) if values is not None else None,
        vllm_cli=spec.get("vllm_cli"),
        vllm_cli_enable=spec.get("vllm_cli_enable"),
        vllm_cli_disable=spec.get("vllm_cli_disable"),
        vllm_env=spec.get("vllm_env"),
        coupled_with=tuple(spec.get("coupled_with") or ()),
        axis=spec.get("axis"),
        description=spec.get("description", ""),
    )


def _parse_constraint(spec: dict[str, Any]) -> CompatRule:
    return CompatRule(
        rule=spec["rule"],
        description=spec.get("description", ""),
        when_knob=spec["when_knob"],
        when_values=tuple(spec["when_values"]),
        requires_knob=spec["requires_knob"],
        requires_values=tuple(spec["requires_values"]),
    )


def to_surrogate_surface(catalog: KnobCatalog) -> dict[str, dict[str, Any]]:
    """Convert catalog to the dict shape ``OptunaSurrogate`` consumes."""
    out: dict[str, dict[str, Any]] = {}
    for name, knob in catalog.knobs.items():
        if knob.type == "int":
            entry: dict[str, Any] = {"type": "int", "low": knob.low, "high": knob.high}
            if knob.step is not None:
                entry["step"] = knob.step
            out[name] = entry
        elif knob.type == "float":
            out[name] = {"type": "float", "low": knob.low, "high": knob.high}
        elif knob.type == "categorical":
            if knob.values is None:
                raise ValueError(f"categorical knob {name!r} missing 'values'")
            out[name] = {"type": "categorical", "values": list(knob.values)}
        elif knob.type == "bool":
            out[name] = {"type": "categorical", "values": [True, False]}
        else:
            raise ValueError(f"unreachable knob type: {knob.type}")
    return out


def defaults(catalog: KnobCatalog) -> dict[str, Any]:
    return {name: knob.default for name, knob in catalog.knobs.items()}


def derive_knob_classes(catalog: KnobCatalog) -> dict[str, dict[str, str]]:
    """Build per-knob value->class taxonomies from compatibility rules.

    Each rule's ``when_values`` for a given ``when_knob`` is treated as
    one structural class labelled with the rule name. Two values in the
    same class compare as distance 0 in ``FeasibilityModel``, so a
    single failure of any one variant generalises to the rest.

    Why: campaign 01 (2026-04-26) showed the L1 classifier with uniform
    per-string distance can't extract "fp8 region is infeasible on
    sm_80" from observed failures of fp8/fp8_e4m3/fp8_e5m2, because each
    pair was distance 1. Collapsing them via the catalog-declared rule
    lets a single fp8 failure generalise. T-26.

    Non-string ``when_values`` (e.g. bools in
    ``chunked_prefill_batched_tokens_bound``) are skipped — bool/numeric
    distances already have natural semantics.
    """
    classes: dict[str, dict[str, str]] = {}
    for rule in catalog.constraints:
        bucket = classes.setdefault(rule.when_knob, {})
        for v in rule.when_values:
            if isinstance(v, str):
                bucket[v] = rule.rule
    return {k: v for k, v in classes.items() if v}


def derive_knob_weights(
    catalog: KnobCatalog, *, high_weight: float = 10.0
) -> dict[str, float]:
    """Build per-knob distance weights from compatibility rules.

    Knobs that appear as ``when_knob`` or ``requires_knob`` in any
    compat rule are treated as deterministic feasibility predictors and
    get ``high_weight``. Every other knob is omitted from the dict;
    ``_config_distance`` uses 1.0 as the default for missing entries.
    T-26b.

    Why: campaign 02 (2026-04-27) showed T-26's class collapse on
    ``kv_cache_dtype`` was insufficient — ``_config_distance`` averaged
    over all 12 L1 knobs, so the FP8 cluster signal got diluted by 11
    unrelated knobs varying across surrogate proposals. Weighting
    catalog-rule knobs ~10x means a candidate matching a known-failed
    region on those knobs lands close to the FAIL neighbours regardless
    of how other knobs differ.

    Default ``high_weight=10.0``: with one upweighted knob at distance
    0 (matching FAIL region) vs 11 default-weighted knobs at average
    distance 0.5, the weighted average is ``(10*0 + 11*0.5)/(10+11) ≈
    0.26``, comfortably below the typical
    ``feasibility_threshold=0.4``. Tunable per deployment.
    """
    knobs_in_rules = {rule.when_knob for rule in catalog.constraints}
    knobs_in_rules |= {rule.requires_knob for rule in catalog.constraints}
    return {name: high_weight for name in catalog.knobs if name in knobs_in_rules}


def violates_constraints(config: dict[str, Any], catalog: KnobCatalog) -> list[str]:
    """Return the names of constraints ``config`` violates."""
    out: list[str] = []
    for rule in catalog.constraints:
        when_val = config.get(rule.when_knob)
        if when_val not in rule.when_values:
            continue
        req_val = config.get(rule.requires_knob)
        if req_val not in rule.requires_values:
            out.append(rule.rule)
    return out


def build_vllm_serve_args(
    model: str, port: int, config: dict[str, Any], catalog: KnobCatalog
) -> tuple[list[str], dict[str, str]]:
    """Assemble ``vllm serve`` CLI + env from a config dict."""
    args: list[str] = ["vllm", "serve", model, "--port", str(port)]
    env: dict[str, str] = {}
    for name, value in config.items():
        knob = catalog.knobs.get(name)
        if knob is None:
            continue
        if knob.vllm_env:
            env[knob.vllm_env] = str(value)
            continue
        if knob.type == "bool":
            if value and knob.vllm_cli_enable:
                args.append(knob.vllm_cli_enable)
            elif (not value) and knob.vllm_cli_disable:
                args.append(knob.vllm_cli_disable)
            continue
        if knob.vllm_cli is None:
            continue
        if value is None or value == "none":
            continue
        args.extend([knob.vllm_cli, str(value)])
    return args, env
