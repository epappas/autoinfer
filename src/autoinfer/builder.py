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
    if cfg.layers.l1_engine is None:
        raise ValueError("iteration-zero builder requires cfg.layers.l1_engine")
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
