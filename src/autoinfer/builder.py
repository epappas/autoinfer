"""Compose a ``ContinuousRunner`` from a validated ``RunConfig``.

Builds one or more layer specs (L1 / L2 / L3) from ``cfg.layers``,
assembles them under a single ``LayerScheduler`` + shared harness
substrate (ledger, event log). Single-layer configs continue to work;
when two or more layers are enabled, the joint runner interleaves per
the scheduler's stale-priority + round-robin policy (see
``controller.stale``).
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from autoinfer.config import RunConfig, WarmstartConfig
from autoinfer.controller import ContinuousRunner, LayerScheduler, LayerSpec
from autoinfer.harness.ledger import Ledger
from autoinfer.layers.l1_engine import KnobCatalog, L1EngineAdapter, defaults, load_catalog
from autoinfer.policy import (
    AnthropicProposalLLM,
    DeterministicProposalLLM,
    OpenAICompatibleProposalLLM,
    Operator,
    OptunaSurrogate,
    ProposalLLM,
)
from autoinfer.policy.feasibility import FeasibilityModel
from autoinfer.policy.surrogate import ConstrainedOptunaSurrogate, Surrogate
from autoinfer.telemetry import EventLog, capture_hw_context, write_hw_context


def build_runner(
    cfg: RunConfig,
    max_trials_override: int | None = None,
    per_layer_overrides: dict[str, int] | None = None,
) -> tuple[ContinuousRunner, Ledger]:
    """Build a runner that owns every layer enabled in ``cfg.layers``.

    Precedence for a layer's ``max_trials``:
      1. ``per_layer_overrides[layer_name]`` when present
      2. ``max_trials_override`` (uniform for every layer)
      3. ``cfg.layers.<layer>.max_trials`` from the config

    Per-layer overrides let smoke tests cap L1 and L2 independently
    (L1 trials take ~2 min on a local GPU; L2 trials take ~20 min on
    Basilica — a uniform "--max-trials 2" still spends ~45 min).
    """
    per_layer_overrides = per_layer_overrides or {}
    specs: dict[str, LayerSpec] = {}
    layer_events: list[dict[str, Any]] = []

    def _layer_override(name: str) -> int | None:
        if name in per_layer_overrides:
            return per_layer_overrides[name]
        return max_trials_override

    if cfg.layers.l1_engine is not None:
        name, spec, ev = _build_l1_spec(cfg, _layer_override("l1_engine"))
        specs[name] = spec
        layer_events.append(ev)
    if cfg.layers.l2_topology is not None:
        name, spec, ev = _build_l2_spec(cfg, _layer_override("l2_topology"))
        specs[name] = spec
        layer_events.append(ev)
    if cfg.layers.l3_kernel is not None:
        name, spec, ev = _build_l3_spec(cfg, _layer_override("l3_kernel"))
        specs[name] = spec
        layer_events.append(ev)

    if not specs:
        raise ValueError("builder requires at least one enabled layer")

    operator = _build_operator(cfg, specs)

    ledger_dir = Path(cfg.harness.ledger.output_dir)
    ledger = Ledger(output_dir=ledger_dir, pareto_axes=cfg.harness.ledger.pareto_axes)
    run_id = uuid.uuid4().hex[:12]
    events = EventLog(ledger_dir / "events.jsonl", run_id=run_id)
    hw_ctx = capture_hw_context()
    write_hw_context(ledger_dir / "hw_context.json", hw_ctx)
    events.emit(
        "config_loaded",
        name=cfg.name,
        layers=list(specs.keys()),
        surrogate_kind=cfg.policy.surrogate.kind,
        warmstart_provider=cfg.policy.warmstart.provider,
        warmstart_model=cfg.policy.warmstart.llm_model,
        operator_cadence=cfg.policy.operator.cadence if cfg.policy.operator else None,
        gpus=[g.get("name") for g in (hw_ctx.get("gpus") or [])],
        vllm_version=hw_ctx.get("vllm_version"),
        autoinfer_version=hw_ctx.get("autoinfer_version"),
        per_layer=layer_events,
    )

    runner = ContinuousRunner(
        scheduler=LayerScheduler(specs),
        ledger=ledger,
        objective_axis="tokens_per_sec",
        maximize=True,
        operator=operator,
        events=events,
    )
    return runner, ledger


def _build_l1_spec(
    cfg: RunConfig, max_trials_override: int | None
) -> tuple[str, LayerSpec, dict[str, Any]]:
    l1_cfg = cfg.layers.l1_engine
    assert l1_cfg is not None
    catalog = load_catalog(l1_cfg.knobs_path)
    prompts = _resolve_gate_prompts(cfg)
    effective_max_kl = _calibrate_max_kl(cfg, l1_cfg.model, prompts, label="l1")

    adapter = L1EngineAdapter(
        model=l1_cfg.model,
        catalog=catalog,
        trace_path=cfg.harness.driver.trace_path,
        reference_uri=cfg.harness.gate.replica_uri,
        quality_prompts=prompts,
        max_kl=effective_max_kl,
        result_dir=cfg.harness.ledger.output_dir,
        batch_sizes=cfg.harness.gate.batch_sizes,
        candidate_port=l1_cfg.candidate_port,
        startup_timeout_s=l1_cfg.startup_timeout_s,
        dataset_name=cfg.harness.driver.dataset_name,
        num_prompts=cfg.harness.driver.num_prompts,
    )
    surrogate = _build_surrogate(
        cfg, surface=adapter.surface(), objective_axis="tokens_per_sec", maximize=True,
    )
    warmstart = _build_warmstart(cfg.policy.warmstart, catalog)

    max_trials = max_trials_override if max_trials_override is not None else l1_cfg.max_trials
    spec = LayerSpec(
        adapter=adapter,
        surrogate=surrogate,
        warmstart=warmstart,
        max_trials=max_trials,
        warmstart_n=cfg.policy.warmstart.n_configs,
        warmstart_prior=cfg.policy.warmstart.hardware_notes or "",
        reserve_cap=l1_cfg.reserve_cap,
    )
    event = {
        "layer": "l1_engine",
        "model": l1_cfg.model,
        "max_trials": max_trials,
        "reserve_cap": l1_cfg.reserve_cap,
        "max_kl_configured": cfg.harness.gate.max_kl,
        "max_kl_effective": effective_max_kl,
        "self_kl_calibrated": cfg.harness.gate.calibrate_self_kl,
    }
    return "l1_engine", spec, event


def _build_l2_spec(
    cfg: RunConfig, max_trials_override: int | None
) -> tuple[str, LayerSpec, dict[str, Any]]:
    import basilica

    from autoinfer.layers.l2_topology import L2TopologyAdapter
    from autoinfer.layers.l2_topology import load_catalog as load_l2_catalog
    from autoinfer.layers.l2_topology.surface import defaults as l2_defaults

    l2_cfg = cfg.layers.l2_topology
    assert l2_cfg is not None
    catalog = load_l2_catalog(l2_cfg.knobs_path)
    prompts = _resolve_gate_prompts(cfg)
    effective_max_kl = _calibrate_max_kl(cfg, l2_cfg.model, prompts, label="l2")

    client = basilica.BasilicaClient()
    adapter = L2TopologyAdapter(
        model=l2_cfg.model,
        catalog=catalog,
        trace_path=cfg.harness.driver.trace_path,
        reference_uri=cfg.harness.gate.replica_uri,
        quality_prompts=prompts,
        max_kl=effective_max_kl,
        result_dir=cfg.harness.ledger.output_dir,
        basilica_client=client,
        batch_sizes=cfg.harness.gate.batch_sizes,
        dataset_name=cfg.harness.driver.dataset_name,
        num_prompts=cfg.harness.driver.num_prompts,
        memory=l2_cfg.memory,
        ttl_seconds=l2_cfg.ttl_seconds,
        deploy_timeout_s=l2_cfg.deploy_timeout_s,
    )
    surrogate = _build_surrogate(
        cfg, surface=adapter.surface(), objective_axis="tokens_per_sec", maximize=True,
    )

    defaults_cfg = l2_defaults(catalog)
    seeds: list[dict[str, Any]] = [defaults_cfg]
    for val in ("L40", "L40S", "RTX 6000 ADA"):
        if "gpu_type" in catalog.knobs and val in (catalog.knobs["gpu_type"].values or ()):
            seeds.append({**defaults_cfg, "gpu_type": val})
    for c in (1, 4):
        if c in (catalog.knobs.get("gpu_count").values or ()):  # type: ignore[union-attr]
            seeds.append({**defaults_cfg, "gpu_count": c})
    warmstart = _build_warmstart_with_seeds(cfg.policy.warmstart, seeds)

    max_trials = max_trials_override if max_trials_override is not None else l2_cfg.max_trials
    spec = LayerSpec(
        adapter=adapter,
        surrogate=surrogate,
        warmstart=warmstart,
        max_trials=max_trials,
        warmstart_n=cfg.policy.warmstart.n_configs,
        warmstart_prior=cfg.policy.warmstart.hardware_notes or "",
        reserve_cap=l2_cfg.reserve_cap,
    )
    event = {
        "layer": "l2_topology",
        "model": l2_cfg.model,
        "max_trials": max_trials,
        "reserve_cap": l2_cfg.reserve_cap,
        "max_kl_configured": cfg.harness.gate.max_kl,
        "max_kl_effective": effective_max_kl,
        "self_kl_calibrated": cfg.harness.gate.calibrate_self_kl,
    }
    return "l2_topology", spec, event


def _build_l3_spec(
    cfg: RunConfig, max_trials_override: int | None
) -> tuple[str, LayerSpec, dict[str, Any]]:
    from autoinfer.layers.l3_kernel import L3KernelAdapter, reference_seed_configs
    from autoinfer.layers.l3_kernel import load_catalog as load_l3_catalog
    from autoinfer.layers.l3_kernel.proposer import KernelProposer

    l3_cfg = cfg.layers.l3_kernel
    assert l3_cfg is not None
    catalog = load_l3_catalog(l3_cfg.knobs_path)

    adapter = L3KernelAdapter(
        catalog=catalog,
        atol=l3_cfg.atol,
        rtol=l3_cfg.rtol,
        perf_repeats=l3_cfg.perf_repeats,
        warmup_runs=l3_cfg.warmup_runs,
    )
    surrogate = _build_surrogate(
        cfg, surface=adapter.surface(), objective_axis="tokens_per_sec", maximize=True,
    )
    seeds = reference_seed_configs()
    warmstart: ProposalLLM
    if cfg.policy.warmstart.provider == "deterministic":
        warmstart = _build_warmstart_with_seeds(cfg.policy.warmstart, seeds)
    else:
        # LLM-driven kernel proposer: wraps the underlying chat-completion
        # client (which exposes .complete(prompt)) so L3 candidates are
        # actual source-code proposals, not surrogate-knob picks.
        raw_llm = _build_warmstart_with_seeds(cfg.policy.warmstart, seeds)
        warmstart = KernelProposer(llm=raw_llm)  # type: ignore[arg-type]

    max_trials = max_trials_override if max_trials_override is not None else l3_cfg.max_trials
    spec = LayerSpec(
        adapter=adapter,
        surrogate=surrogate,
        warmstart=warmstart,
        max_trials=max_trials,
        warmstart_n=min(cfg.policy.warmstart.n_configs, len(seeds)),
        warmstart_prior=cfg.policy.warmstart.hardware_notes or "",
        reserve_cap=l3_cfg.reserve_cap,
    )
    event = {
        "layer": "l3_kernel",
        "max_trials": max_trials,
        "reserve_cap": l3_cfg.reserve_cap,
        "atol": l3_cfg.atol,
        "rtol": l3_cfg.rtol,
    }
    return "l3_kernel", spec, event


def _build_operator(cfg: RunConfig, specs: Iterable[str]) -> Operator | None:
    """Build a shared operator that proposes per-layer using the LLM.

    The wrapped LLM is reused for every layer the runner iterates —
    per-call the scheduler passes that layer's surface. For joint
    multi-layer runs the caller must use ``provider='anthropic'`` or
    ``'openai_compatible'``; a deterministic seed list can't cover two
    surfaces at once.
    """
    del specs
    if cfg.policy.operator is None:
        return None
    op_wcfg = WarmstartConfig(
        provider=cfg.policy.warmstart.provider,
        llm_model=cfg.policy.operator.llm_model,
        base_url=cfg.policy.warmstart.base_url,
        api_key_env=cfg.policy.warmstart.api_key_env,
        seed_configs=cfg.policy.warmstart.seed_configs,
        hardware_notes=cfg.policy.warmstart.hardware_notes,
    )
    op_llm = _build_warmstart_with_seeds(op_wcfg, [])
    return Operator(llm=op_llm, cadence=cfg.policy.operator.cadence)


def _build_surrogate(
    cfg: RunConfig,
    surface: dict[str, Any],
    objective_axis: str,
    maximize: bool,
) -> Surrogate:
    """Build the perf surrogate, optionally wrapped in feasibility constraint.

    When ``cfg.policy.surrogate.feasibility_threshold > 0``, returns a
    ``ConstrainedOptunaSurrogate`` that learns a feasibility classifier
    from typed failures and rejects candidates whose nearest-neighbor
    history is too failure-dense (see policy/feasibility.py for the
    motivating data).
    """
    s_cfg = cfg.policy.surrogate
    inner = OptunaSurrogate(
        kind=s_cfg.kind,
        seed=s_cfg.seed,
        surface=surface,
        objective_axis=objective_axis,
        maximize=maximize,
    )
    if s_cfg.feasibility_threshold <= 0.0:
        return inner
    return ConstrainedOptunaSurrogate(
        inner=inner,
        feasibility=FeasibilityModel(
            k=s_cfg.feasibility_k,
            min_observations=s_cfg.feasibility_min_observations,
        ),
        threshold=s_cfg.feasibility_threshold,
        max_resamples=s_cfg.feasibility_max_resamples,
    )


def _resolve_gate_prompts(cfg: RunConfig) -> list[str]:
    prompts = _load_prompts(cfg.harness.gate.prompts_path)
    return prompts[: cfg.harness.gate.smoke_prompts]


def _calibrate_max_kl(
    cfg: RunConfig, model: str, prompts: list[str], label: str
) -> float:
    """Run self-KL calibration; calibration only RAISES the ceiling.

    Self-KL measures reference-vs-reference noise, which under greedy
    decode is much smaller (~0.05) than the candidate-vs-reference KL
    of clean configs (~1-5 nats). Using ``5*self_p95`` directly as the
    gate produces ceilings way below real candidate noise — exactly
    what killed the smoke v2's L2 H100 trial (kl=4.13 rejected by
    effective=0.3965).

    The user's configured ``max_kl`` represents their "definitely-real-
    drift" floor. Calibration should only loosen this when the
    reference is unusually noisy (raising the ceiling), never tighten
    below user intent. Take the max of (calibration result, configured
    max_kl) — calibration becomes a one-way valve.
    """
    configured = cfg.harness.gate.max_kl
    if not cfg.harness.gate.calibrate_self_kl:
        return configured
    from autoinfer.harness.gate import calibrate_self_kl

    try:
        stats = calibrate_self_kl(
            endpoint=cfg.harness.gate.replica_uri, model=model, prompts=prompts,
        )
        calibrated = stats["p95"] * cfg.harness.gate.calibration_multiplier
        return max(calibrated, configured)
    except Exception as e:  # noqa: BLE001
        print(f"[builder-{label}] self-kl calibration failed: {e}; using config max_kl", flush=True)
        return configured


def _build_warmstart_with_seeds(
    wcfg: WarmstartConfig, fallback_seeds: list[dict[str, Any]]
) -> ProposalLLM:
    if wcfg.provider == "deterministic":
        seeds = wcfg.seed_configs or fallback_seeds
        return DeterministicProposalLLM(seeds)
    if wcfg.provider == "anthropic":
        return AnthropicProposalLLM(
            model=wcfg.llm_model, api_key=_env_key(wcfg.api_key_env)
        )
    if wcfg.provider == "openai_compatible":
        if not wcfg.base_url:
            raise ValueError("provider='openai_compatible' requires base_url")
        return OpenAICompatibleProposalLLM(
            base_url=wcfg.base_url, model=wcfg.llm_model,
            api_key=_env_key(wcfg.api_key_env),
        )
    raise ValueError(f"unknown warmstart provider: {wcfg.provider!r}")


def _build_warmstart(wcfg: WarmstartConfig, catalog: KnobCatalog) -> ProposalLLM:
    if wcfg.provider == "deterministic" and not wcfg.seed_configs:
        base = defaults(catalog)
        fallback = [
            base,
            {**base, "attention_backend": "FLASHINFER"},
            {**base, "kv_cache_dtype": "fp8", "attention_backend": "FLASHINFER"},
            {**base, "enable_prefix_caching": True},
            {**base, "max_num_batched_tokens": 8192},
            {**base, "max_num_seqs": 256},
            {**base, "gpu_memory_utilization": 0.85},
            {**base, "block_size": 32},
        ]
        return DeterministicProposalLLM(fallback)
    return _build_warmstart_with_seeds(wcfg, [])


def _env_key(env_var: str | None) -> str | None:
    return os.environ.get(env_var) if env_var else None


def _load_prompts(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"prompts file not found: {path}")
    if path.suffix.lower() == ".jsonl":
        out: list[str] = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row: dict[str, Any] = json.loads(line)
                if "prompt" in row:
                    out.append(str(row["prompt"]))
                elif "text" in row:
                    out.append(str(row["text"]))
        return out
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
