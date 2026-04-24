"""Compose a ``ContinuousRunner`` from a validated ``RunConfig``.

Loads the L1 knobs catalog, constructs the adapter, wires the hybrid
policy stack (warmstart + surrogate + operator), and returns a
ready-to-run controller plus its ledger. Iteration zero is single-layer
(L1 only); multi-layer composition is session four.
"""

from __future__ import annotations

import json
import os
import uuid
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
from autoinfer.telemetry import EventLog, capture_hw_context, write_hw_context


def build_runner(
    cfg: RunConfig, max_trials_override: int | None = None
) -> tuple[ContinuousRunner, Ledger]:
    # L2 takes precedence over L1 when both are configured; iteration-zero
    # runs exactly one layer at a time (multi-layer composition is future).
    if cfg.layers.l2_topology is not None:
        return _build_l2_runner(cfg, max_trials_override)
    if cfg.layers.l1_engine is None:
        raise ValueError("builder requires cfg.layers.l1_engine or cfg.layers.l2_topology")
    l1_cfg = cfg.layers.l1_engine
    catalog = load_catalog(l1_cfg.knobs_path)

    prompts = _load_prompts(cfg.harness.gate.prompts_path)
    # cap to smoke_prompts so per-trial gate stays bounded; the gate
    # does 2 sequential HTTP calls per prompt (candidate + reference)
    # and each can take several seconds.
    prompts = prompts[: cfg.harness.gate.smoke_prompts]

    effective_max_kl = cfg.harness.gate.max_kl
    if cfg.harness.gate.calibrate_self_kl:
        from autoinfer.harness.gate import calibrate_self_kl

        try:
            stats = calibrate_self_kl(
                endpoint=cfg.harness.gate.replica_uri,
                model=l1_cfg.model,
                prompts=prompts,
            )
            effective_max_kl = stats["p95"] * cfg.harness.gate.calibration_multiplier
            # keep a lower floor so we still catch gross drift even if self
            # noise is extremely low
            effective_max_kl = max(effective_max_kl, 0.1)
        except Exception as e:  # noqa: BLE001
            print(f"[builder] self-kl calibration failed: {e}; using config max_kl", flush=True)

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

    surrogate = OptunaSurrogate(
        kind=cfg.policy.surrogate.kind,
        seed=cfg.policy.surrogate.seed,
        surface=adapter.surface(),
        objective_axis="tokens_per_sec",
        maximize=True,
    )

    warmstart = _build_warmstart(cfg.policy.warmstart, catalog)

    operator: Operator | None = None
    if cfg.policy.operator is not None:
        op_llm = _build_warmstart(
            WarmstartConfig(
                provider=cfg.policy.warmstart.provider,
                llm_model=cfg.policy.operator.llm_model,
                base_url=cfg.policy.warmstart.base_url,
                api_key_env=cfg.policy.warmstart.api_key_env,
                seed_configs=cfg.policy.warmstart.seed_configs,
            ),
            catalog,
        )
        operator = Operator(llm=op_llm, cadence=cfg.policy.operator.cadence)

    max_trials = (
        max_trials_override if max_trials_override is not None else l1_cfg.max_trials
    )
    spec = LayerSpec(
        adapter=adapter,
        surrogate=surrogate,
        warmstart=warmstart,
        max_trials=max_trials,
        warmstart_n=cfg.policy.warmstart.n_configs,
        warmstart_prior=cfg.policy.warmstart.hardware_notes or "",
    )
    ledger_dir = Path(cfg.harness.ledger.output_dir)
    ledger = Ledger(
        output_dir=ledger_dir,
        pareto_axes=cfg.harness.ledger.pareto_axes,
    )

    # Rich telemetry — event log + hw context snapshot at run start
    run_id = uuid.uuid4().hex[:12]
    events = EventLog(ledger_dir / "events.jsonl", run_id=run_id)
    hw_ctx = capture_hw_context()
    write_hw_context(ledger_dir / "hw_context.json", hw_ctx)
    events.emit(
        "config_loaded",
        name=cfg.name,
        model=l1_cfg.model,
        max_trials=max_trials,
        warmstart_provider=cfg.policy.warmstart.provider,
        warmstart_model=cfg.policy.warmstart.llm_model,
        surrogate_kind=cfg.policy.surrogate.kind,
        operator_cadence=cfg.policy.operator.cadence if cfg.policy.operator else None,
        gpus=[g.get("name") for g in (hw_ctx.get("gpus") or [])],
        vllm_version=hw_ctx.get("vllm_version"),
        autoinfer_version=hw_ctx.get("autoinfer_version"),
        max_kl_configured=cfg.harness.gate.max_kl,
        max_kl_effective=effective_max_kl,
        self_kl_calibrated=cfg.harness.gate.calibrate_self_kl,
    )

    runner = ContinuousRunner(
        scheduler=LayerScheduler({"l1_engine": spec}),
        ledger=ledger,
        objective_axis="tokens_per_sec",
        maximize=True,
        operator=operator,
        events=events,
    )
    return runner, ledger


def _build_l2_runner(
    cfg: RunConfig, max_trials_override: int | None
) -> tuple[ContinuousRunner, Ledger]:
    """L2 topology runner: per-trial Basilica deployments over varied GPU classes."""
    import basilica

    from autoinfer.layers.l2_topology import L2TopologyAdapter
    from autoinfer.layers.l2_topology import load_catalog as load_l2_catalog
    from autoinfer.layers.l2_topology.surface import (
        defaults as l2_defaults,
    )

    l2_cfg = cfg.layers.l2_topology
    assert l2_cfg is not None  # build_runner's precondition
    catalog = load_l2_catalog(l2_cfg.knobs_path)

    prompts = _load_prompts(cfg.harness.gate.prompts_path)
    prompts = prompts[: cfg.harness.gate.smoke_prompts]

    effective_max_kl = cfg.harness.gate.max_kl
    if cfg.harness.gate.calibrate_self_kl:
        from autoinfer.harness.gate import calibrate_self_kl

        try:
            stats = calibrate_self_kl(
                endpoint=cfg.harness.gate.replica_uri,
                model=l2_cfg.model,
                prompts=prompts,
            )
            effective_max_kl = max(
                stats["p95"] * cfg.harness.gate.calibration_multiplier, 0.1
            )
        except Exception as e:  # noqa: BLE001
            print(f"[builder-l2] self-kl calibration failed: {e}; using config max_kl", flush=True)

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

    surrogate = OptunaSurrogate(
        kind=cfg.policy.surrogate.kind,
        seed=cfg.policy.surrogate.seed,
        surface=adapter.surface(),
        objective_axis="tokens_per_sec",
        maximize=True,
    )

    # diverse deterministic seeds for L2 — reuse defaults + variants
    defaults_cfg = l2_defaults(catalog)
    seeds: list[dict[str, Any]] = [defaults_cfg]
    for val in ("L40", "L40S", "RTX 6000 ADA"):
        if "gpu_type" in catalog.knobs and val in (catalog.knobs["gpu_type"].values or ()):
            seeds.append({**defaults_cfg, "gpu_type": val})
    for c in (1, 4):
        if c in (catalog.knobs.get("gpu_count").values or ()):  # type: ignore[union-attr]
            seeds.append({**defaults_cfg, "gpu_count": c})

    warmstart = _build_warmstart_with_seeds(
        cfg.policy.warmstart, seeds
    )

    operator: Operator | None = None
    if cfg.policy.operator is not None:
        op_llm = _build_warmstart_with_seeds(
            WarmstartConfig(
                provider=cfg.policy.warmstart.provider,
                llm_model=cfg.policy.operator.llm_model,
                base_url=cfg.policy.warmstart.base_url,
                api_key_env=cfg.policy.warmstart.api_key_env,
                seed_configs=cfg.policy.warmstart.seed_configs,
                hardware_notes=cfg.policy.warmstart.hardware_notes,
            ),
            seeds,
        )
        operator = Operator(llm=op_llm, cadence=cfg.policy.operator.cadence)

    max_trials = max_trials_override if max_trials_override is not None else l2_cfg.max_trials
    spec = LayerSpec(
        adapter=adapter,
        surrogate=surrogate,
        warmstart=warmstart,
        max_trials=max_trials,
        warmstart_n=cfg.policy.warmstart.n_configs,
        warmstart_prior=cfg.policy.warmstart.hardware_notes or "",
    )

    ledger_dir = Path(cfg.harness.ledger.output_dir)
    ledger = Ledger(output_dir=ledger_dir, pareto_axes=cfg.harness.ledger.pareto_axes)
    run_id = uuid.uuid4().hex[:12]
    events = EventLog(ledger_dir / "events.jsonl", run_id=run_id)
    hw_ctx = capture_hw_context()
    write_hw_context(ledger_dir / "hw_context.json", hw_ctx)
    events.emit(
        "config_loaded",
        name=cfg.name,
        layer="l2_topology",
        model=l2_cfg.model,
        max_trials=max_trials,
        warmstart_provider=cfg.policy.warmstart.provider,
        warmstart_model=cfg.policy.warmstart.llm_model,
        surrogate_kind=cfg.policy.surrogate.kind,
        operator_cadence=cfg.policy.operator.cadence if cfg.policy.operator else None,
        gpus=[g.get("name") for g in (hw_ctx.get("gpus") or [])],
        max_kl_configured=cfg.harness.gate.max_kl,
        max_kl_effective=effective_max_kl,
        self_kl_calibrated=cfg.harness.gate.calibrate_self_kl,
    )
    runner = ContinuousRunner(
        scheduler=LayerScheduler({"l2_topology": spec}),
        ledger=ledger,
        objective_axis="tokens_per_sec",
        maximize=True,
        operator=operator,
        events=events,
    )
    return runner, ledger


def _build_warmstart_with_seeds(
    wcfg: WarmstartConfig, fallback_seeds: list[dict[str, Any]]
) -> ProposalLLM:
    """Same as _build_warmstart but with a caller-supplied fallback seed list.

    L2 defaults to a single gpu_type; varying it across warmstart trials
    only happens when the caller seeds the deterministic proposer.
    """
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
    if wcfg.provider == "deterministic":
        if wcfg.seed_configs:
            seeds = wcfg.seed_configs
        else:
            # diversify from the defaults so warmstart trials differ
            base = defaults(catalog)
            seeds = [
                base,
                {**base, "attention_backend": "FLASHINFER"},
                {**base, "kv_cache_dtype": "fp8", "attention_backend": "FLASHINFER"},
                {**base, "enable_prefix_caching": True},
                {**base, "max_num_batched_tokens": 8192},
                {**base, "max_num_seqs": 256},
                {**base, "gpu_memory_utilization": 0.85},
                {**base, "block_size": 32},
            ]
        return DeterministicProposalLLM(seeds)
    if wcfg.provider == "anthropic":
        return AnthropicProposalLLM(
            model=wcfg.llm_model,
            api_key=_env_key(wcfg.api_key_env),
        )
    if wcfg.provider == "openai_compatible":
        if not wcfg.base_url:
            raise ValueError("provider='openai_compatible' requires base_url")
        return OpenAICompatibleProposalLLM(
            base_url=wcfg.base_url,
            model=wcfg.llm_model,
            api_key=_env_key(wcfg.api_key_env),
        )
    raise ValueError(f"unknown warmstart provider: {wcfg.provider!r}")


def _env_key(env_var: str | None) -> str | None:
    return os.environ.get(env_var) if env_var else None


def _load_prompts(path: Path) -> list[str]:
    """Load prompts from JSONL (``{"prompt": "..."}`` per line) or plain text."""
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
