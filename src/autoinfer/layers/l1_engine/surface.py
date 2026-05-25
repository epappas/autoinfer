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

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from autoinfer.harness.failure import FailureKind

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


_KIND_DRIVERS: dict[FailureKind, frozenset[str]] = {
    FailureKind.OOM: frozenset(
        {
            "kv_cache_dtype",
            "gpu_memory_utilization",
            "max_num_seqs",
            "max_num_batched_tokens",
            "block_size",
            "long_prefill_token_threshold",
        }
    ),
    FailureKind.QUALITY_KL: frozenset(
        {
            "quantization",
            "dtype",
            "kv_cache_dtype",
        }
    ),
    FailureKind.QUALITY_INVARIANCE: frozenset(
        {
            "max_num_seqs",
            "block_size",
            "enable_chunked_prefill",
        }
    ),
}
"""Static per-FailureKind driver knobs for T-26c.

Each entry lists the knobs whose values drive the failure region for
that kind. ``derive_kind_weights`` upweights these knobs ~10x inside
the per-kind sub-classifier so it learns the right region.

STARTUP is treated separately: STARTUP failures are
catalog-rule-violation events (e.g. fp8 on sm_80), so its driver knobs
are derived from the catalog's compat-rule footprint, mirroring
``derive_knob_weights`` from T-26b.

Kinds not in this table (UNKNOWN, HANG, NCCL) fall back to
``self.knob_weights`` inside ``_predict_proba_for_kind`` — they share
the shared T-26b weights instead of getting per-kind sharpening, which
is the right default until enough HANG/NCCL data exists to justify
specialised drivers.
"""


def derive_kind_weights(
    catalog: KnobCatalog, *, high_weight: float = 10.0
) -> dict[FailureKind, dict[str, float]]:
    """Build per-FailureKind knob weights from a static driver taxonomy. T-26c.

    For each kind in ``_KIND_DRIVERS``, upweight any catalog knob in
    that kind's driver set to ``high_weight``. STARTUP is derived from
    the catalog's compat-rule footprint (``when_knob`` ∪
    ``requires_knob``), so a new catalog rule automatically extends
    STARTUP's reach without code changes.

    Kinds whose driver set has no overlap with the catalog are omitted
    — ``FeasibilityModel._predict_proba_for_kind`` falls back to
    ``self.knob_weights`` for those kinds, which is the right T-26b
    behaviour.

    Future work (T-26d) can extend this by data-mining per-kind priors
    from accumulated trial history once the catalog is mature.
    """
    out: dict[FailureKind, dict[str, float]] = {}
    catalog_knob_names = set(catalog.knobs.keys())
    for kind, drivers in _KIND_DRIVERS.items():
        weights = {
            name: high_weight for name in drivers if name in catalog_knob_names
        }
        if weights:
            out[kind] = weights
    rule_knobs = {rule.when_knob for rule in catalog.constraints}
    rule_knobs |= {rule.requires_knob for rule in catalog.constraints}
    startup_weights = {
        name: high_weight for name in catalog.knobs if name in rule_knobs
    }
    if startup_weights:
        out[FailureKind.STARTUP] = startup_weights
    return out


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


_GMU_CAP_ENV = "AUTOINFER_L1_GMU_MAX"


def _maybe_cap_gpu_memory_utilization(name: str, value: Any) -> Any:
    """Clamp candidate ``gpu_memory_utilization`` to the env-var cap.

    When ``AUTOINFER_L1_GMU_MAX`` is set in the environment, candidate
    vLLM's gpu_memory_utilization is clamped to ``min(value, cap)``.
    Used in 1-GPU mode where the reference replica shares the GPU with
    each candidate — the campaign_runner sets the cap so the candidate
    leaves room for the reference's HBM reservation.

    vLLM interprets ``--gpu-memory-utilization`` as a fraction of the
    GPU's TOTAL memory (not remainder), so the surrogate's default
    sweep [0.80, 0.95] OOMs when the reference is also resident.
    """
    if name != "gpu_memory_utilization":
        return value
    cap_str = os.environ.get(_GMU_CAP_ENV)
    if not cap_str:
        return value
    try:
        cap = float(cap_str)
    except ValueError:
        return value
    if not isinstance(value, (int, float)):
        return value
    return min(float(value), cap)


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
        # Clamp candidate gpu_memory_utilization in 1-GPU shared-mode.
        value = _maybe_cap_gpu_memory_utilization(name, value)
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
