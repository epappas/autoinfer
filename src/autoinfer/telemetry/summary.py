"""Per-run aggregate summaries — article-grade data.

Writes three companion files alongside ``ledger/trials/``:

- ``results.tsv``   — one row per trial, every knob + outcome.
- ``run_summary.json`` — the aggregate an article can link to.
- ``hw_context.json``  — hardware + versions captured at run start.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from autoinfer.harness.ledger import Entry


def capture_hw_context() -> dict[str, Any]:
    """Snapshot GPU + software context on the host where the run starts."""
    ctx: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_sha": _git_sha(),
        "env": {
            k: os.environ.get(k, "")
            for k in ("CUDA_VISIBLE_DEVICES", "VLLM_ATTENTION_BACKEND")
            if k in os.environ
        },
    }
    ctx["gpus"] = _nvidia_smi_json()
    ctx["vllm_version"] = _pip_show("vllm")
    ctx["torch_version"] = _pip_show("torch")
    ctx["autoinfer_version"] = _pip_show("autoinfer")
    return ctx


def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True, timeout=3,
        )
        return out.strip()
    except Exception:
        return None


def _pip_show(pkg: str) -> str | None:
    """Return the installed version of ``pkg`` (None if absent).

    Uses ``importlib.metadata`` so it sees what THIS Python interpreter
    can actually import — robust across pip/uv/poetry-managed envs.
    Falls back to ``pip show`` (the original implementation) if metadata
    lookup raises something unexpected.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version(pkg)
        except PackageNotFoundError:
            return None
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["pip", "show", pkg], stderr=subprocess.DEVNULL, text=True, timeout=5,
        )
        for line in out.splitlines():
            if line.startswith("Version:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        return None
    return None


def _nvidia_smi_json() -> list[dict[str, str]] | None:
    fmt = "csv,noheader,nounits"
    query = "index,name,driver_version,memory.total,memory.used,compute_cap"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", f"--format={fmt}"],
            stderr=subprocess.DEVNULL, text=True, timeout=5,
        )
    except Exception:
        return None
    rows: list[dict[str, str]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 6:
            continue
        rows.append({
            "index": parts[0], "name": parts[1], "driver": parts[2],
            "memory_total_mib": parts[3], "memory_used_mib": parts[4],
            "compute_cap": parts[5],
        })
    return rows


def write_hw_context(path: Path, ctx: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ctx, indent=2, sort_keys=True, default=str))


_TSV_HEADER = [
    "trial_id", "layer", "phase", "outcome",
    "tokens_per_sec", "ttft_p99_ms", "tpot_p99_ms", "peak_hbm_gb",
    "kl_divergence", "failure_kind",
    "attention_backend", "kv_cache_dtype", "enable_prefix_caching",
    "enable_chunked_prefill", "block_size", "max_num_batched_tokens",
    "max_num_seqs", "gpu_memory_utilization", "quantization", "dtype",
]


def _phase_from_tid(tid: str) -> str:
    # e.g. "l1_engine_w0003" -> "w"; "l1_engine_s0011" -> "s"; "l1_engine_o0006" -> "o"
    parts = tid.rsplit("_", 1)
    return parts[-1][0] if len(parts) == 2 and parts[-1] else "?"


def _phase_label(phase: str) -> str:
    return {"w": "warmstart", "s": "surrogate", "o": "operator"}.get(phase, phase)


def write_results_tsv(path: Path, entries: list[Entry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(_TSV_HEADER)
        for e in entries:
            m = e.measurement
            fk = e.failure.kind.value if e.failure else ""
            outcome = (
                "kept" if e.kept
                else ("failure:" + fk if e.failure else "stale")
            )
            row = [
                e.trial_id, e.layer, _phase_label(_phase_from_tid(e.trial_id)), outcome,
                (m.tokens_per_sec if m else ""),
                (m.ttft_p99_ms if m else ""),
                (m.tpot_p99_ms if m else ""),
                (m.peak_hbm_gb if m else ""),
                (m.kl_divergence if m else ""),
                fk,
                e.config.get("attention_backend", ""),
                e.config.get("kv_cache_dtype", ""),
                e.config.get("enable_prefix_caching", ""),
                e.config.get("enable_chunked_prefill", ""),
                e.config.get("block_size", ""),
                e.config.get("max_num_batched_tokens", ""),
                e.config.get("max_num_seqs", ""),
                e.config.get("gpu_memory_utilization", ""),
                e.config.get("quantization", ""),
                e.config.get("dtype", ""),
            ]
            w.writerow(row)


def _entry_summary(e: Entry) -> dict[str, Any]:
    """Compact single-entry view for inclusion in run_summary.json."""
    if e.measurement is None:
        return {"trial_id": e.trial_id, "layer": e.layer, "config": e.config}
    return {
        "trial_id": e.trial_id,
        "layer": e.layer,
        "config": e.config,
        "tokens_per_sec": e.measurement.tokens_per_sec,
        "tpot_p99_ms": e.measurement.tpot_p99_ms,
        "ttft_p99_ms": e.measurement.ttft_p99_ms,
        "kl_divergence": e.measurement.kl_divergence,
        "peak_hbm_gb": e.measurement.peak_hbm_gb,
    }


def build_run_summary(
    run_id: str,
    entries: list[Entry],
    pareto: list[Entry],
    hw_context: dict[str, Any],
    elapsed_s: float,
) -> dict[str, Any]:
    kept = [e for e in entries if e.kept]
    failed = [e for e in entries if e.failure is not None]
    by_kind: dict[str, int] = {}
    for e in failed:
        if e.failure is None:
            continue
        k = e.failure.kind.value
        by_kind[k] = by_kind.get(k, 0) + 1
    phase_counts: dict[str, int] = {}
    for e in entries:
        p = _phase_label(_phase_from_tid(e.trial_id))
        phase_counts[p] = phase_counts.get(p, 0) + 1
    top = None
    if kept:
        best = max(kept, key=lambda e: e.measurement.tokens_per_sec if e.measurement else 0)
        if best.measurement:
            top = _entry_summary(best)

    # T-22: per-layer best in run_summary so downstream tools (plots,
    # articles, the analyzer) can read per-layer winners without
    # re-loading every trial JSON. Joint Pareto stays as-is — that's
    # the unit-comparable frontier.
    by_layer_best: dict[str, dict[str, Any]] = {}
    layer_kept: dict[str, list[Entry]] = {}
    for e in kept:
        layer_kept.setdefault(e.layer, []).append(e)
    for layer, ents in layer_kept.items():
        best = max(ents, key=lambda e: e.measurement.tokens_per_sec if e.measurement else 0)
        if best.measurement:
            by_layer_best[layer] = _entry_summary(best)

    pareto_serialised = [_entry_summary(e) for e in pareto]

    n_kept_by_layer = {layer: len(ents) for layer, ents in layer_kept.items()}
    n_failed_by_layer: dict[str, int] = {}
    for e in failed:
        n_failed_by_layer[e.layer] = n_failed_by_layer.get(e.layer, 0) + 1

    return {
        "run_id": run_id,
        "elapsed_s": elapsed_s,
        "n_trials": len(entries),
        "n_kept": len(kept),
        "n_failed": len(failed),
        "n_kept_by_layer": n_kept_by_layer,
        "n_failed_by_layer": n_failed_by_layer,
        "failures_by_kind": by_kind,
        "phase_counts": phase_counts,
        "pareto_size": len(pareto),
        "pareto_frontier": pareto_serialised,
        "best_by_layer": by_layer_best,
        "top_by_tokens_per_sec": top,
        "hw_context": hw_context,
    }


def write_run_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
