#!/usr/bin/env python3
"""Analyze a captured iteration-zero run and emit article-grade summaries.

Reads a directory of per-trial JSONs (and optionally events.jsonl /
results.tsv / run_summary.json / hw_context.json) and prints:

- One-line summary per trial (kept vs failure kind, key knobs, metrics).
- Pareto frontier with all axes.
- Failure-by-kind histogram.
- Phase breakdown (warmstart / surrogate / operator).
- Best-by-tokens_per_sec and tradeoffs.

Joint-aware: when trials span multiple layers (l1_engine + l2_topology
+ l3_kernel), reports per-layer best, joint Pareto frontier, and the
single-layer-vs-joint comparison that matters for the thesis claim.

Usage:
    uv run python scripts/analyze_run.py <run-dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

L1_KNOBS = (
    "attention_backend",
    "kv_cache_dtype",
    "dtype",
    "enable_prefix_caching",
    "max_num_batched_tokens",
    "max_num_seqs",
    "block_size",
    "gpu_memory_utilization",
)
L2_KNOBS = ("gpu_type", "gpu_count", "dtype", "gpu_memory_utilization", "enforce_eager")
L3_KNOBS = ("target_op", "shape_regime", "dtype")


def load_trials(run_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in sorted(run_dir.rglob("*.json")):
        if (
            p.name.endswith("_bench.json")
            or p.name in ("run_summary.json", "hw_context.json")
        ):
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if "trial_id" in d and "config" in d:
            out.append(d)
    return out


def classify(d: dict[str, Any]) -> str:
    if d.get("stale"):
        return "stale"
    if d.get("measurement"):
        return "kept"
    if d.get("failure"):
        return "fail:" + d["failure"]["kind"]
    return "?"


def phase_of(tid: str) -> str:
    parts = tid.rsplit("_", 1)
    if len(parts) != 2 or not parts[-1]:
        return "?"
    p = parts[-1][0]
    return {"w": "warmstart", "s": "surrogate", "o": "operator"}.get(p, p)


def is_dominated(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """True if b dominates a on (max tok/s, min p99 TPOT, min HBM)."""
    am, bm = a["measurement"], b["measurement"]
    return (
        bm["tokens_per_sec"] >= am["tokens_per_sec"]
        and bm["tpot_p99_ms"] <= am["tpot_p99_ms"]
        and bm["peak_hbm_gb"] <= am["peak_hbm_gb"]
        and (
            bm["tokens_per_sec"] > am["tokens_per_sec"]
            or bm["tpot_p99_ms"] < am["tpot_p99_ms"]
            or bm["peak_hbm_gb"] < am["peak_hbm_gb"]
        )
    )


def pareto_front(kept: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [a for a in kept if not any(is_dominated(a, b) for b in kept if b is not a)]


def _layer_knobs(layer: str) -> tuple[str, ...]:
    return {"l1_engine": L1_KNOBS, "l2_topology": L2_KNOBS, "l3_kernel": L3_KNOBS}.get(layer, ())


def fmt_trial_row(d: dict[str, Any]) -> str:
    c = d["config"]
    layer = d.get("layer", "?")
    knobs = _layer_knobs(layer)
    cfg_summary = " ".join(f"{k}={c.get(k, '')}" for k in knobs if k in c)
    base = f"{d['trial_id']:26s} L={layer[:3]:3s} {phase_of(d['trial_id']):9s} {cfg_summary}"
    if d.get("stale"):
        m = d.get("measurement") or {}
        return (
            f"STALE {base} -> tok/s={m.get('tokens_per_sec', 0):7.1f}"
            f" tpot99={m.get('tpot_p99_ms', 0):6.1f}ms (invalidated)"
        )
    if d.get("measurement"):
        m = d["measurement"]
        return (
            f"KEPT  {base}"
            f" -> tok/s={m['tokens_per_sec']:7.1f}"
            f" ttft99={m['ttft_p99_ms']:8.1f}ms"
            f" tpot99={m['tpot_p99_ms']:6.1f}ms"
            f" kl={m['kl_divergence']:5.2f}"
            f" hbm={m['peak_hbm_gb']:5.1f}GB"
        )
    if d.get("failure"):
        return f"FAIL  {base} -> {d['failure']['kind']}"
    return f"????  {base}"


def _print_layer_breakdown(trials: list[dict[str, Any]]) -> None:
    by_layer: dict[str, list[dict[str, Any]]] = {}
    for t in trials:
        by_layer.setdefault(t.get("layer", "unknown"), []).append(t)
    print("PER-LAYER BREAKDOWN:")
    for layer, ts in sorted(by_layer.items()):
        kept = [t for t in ts if t.get("measurement") and not t.get("stale")]
        stale = [t for t in ts if t.get("stale")]
        fails = [t for t in ts if t.get("failure")]
        print(
            f"  {layer:14s} total={len(ts):3d}"
            f" kept={len(kept):3d} stale={len(stale):3d} fail={len(fails):3d}"
        )
        if kept:
            top = max(kept, key=lambda x: x["measurement"]["tokens_per_sec"])
            m = top["measurement"]
            print(
                f"    best {top['trial_id']}:"
                f" tok/s={m['tokens_per_sec']:.1f}"
                f" tpot99={m['tpot_p99_ms']:.1f}ms"
            )
    print()


def _print_cross_layer(front: list[dict[str, Any]]) -> None:
    """Surface the joint-vs-single thesis claim."""
    layers_on_front = {t.get("layer") for t in front}
    if len(layers_on_front) < 2:
        return
    print("CROSS-LAYER PARETO COMPOSITION:")
    print(f"  layers contributing to frontier: {sorted(layers_on_front)}")
    for t in sorted(front, key=lambda x: -x["measurement"]["tokens_per_sec"]):
        m = t["measurement"]
        print(
            f"  {t.get('layer', '?'):14s} {t['trial_id']:26s}"
            f" tok/s={m['tokens_per_sec']:7.1f}"
            f" tpot99={m['tpot_p99_ms']:6.1f}ms"
        )
    print()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    args = p.parse_args()
    if not args.run_dir.exists():
        print(f"missing: {args.run_dir}", file=sys.stderr)
        return 2

    trials = load_trials(args.run_dir)
    print(f"=== {args.run_dir} ===")
    print(f"loaded {len(trials)} trials\n")

    by_phase: dict[str, int] = {}
    by_outcome: dict[str, int] = {}
    for t in trials:
        by_phase[phase_of(t["trial_id"])] = by_phase.get(phase_of(t["trial_id"]), 0) + 1
        by_outcome[classify(t)] = by_outcome.get(classify(t), 0) + 1

    print("PHASE BREAKDOWN:")
    for k, v in sorted(by_phase.items()):
        print(f"  {k:12s} {v}")
    print()
    print("OUTCOME BREAKDOWN:")
    for k, v in sorted(by_outcome.items()):
        print(f"  {k:20s} {v}")
    print()

    _print_layer_breakdown(trials)

    kept = [t for t in trials if t.get("measurement") and not t.get("stale")]
    if kept:
        front = pareto_front(kept)
        print(f"PARETO FRONTIER ({len(front)} of {len(kept)} kept, joint across layers):")
        for t in sorted(front, key=lambda x: -x["measurement"]["tokens_per_sec"]):
            print("  " + fmt_trial_row(t))
        print()

        _print_cross_layer(front)

        top = max(kept, key=lambda t: t["measurement"]["tokens_per_sec"])
        print("TOP BY TOKENS_PER_SEC:")
        print("  " + fmt_trial_row(top))
        print()

    print("ALL TRIALS (sorted by tok/s, kept first then stale, then fails):")
    sortable_kept = sorted(
        [t for t in trials if t.get("measurement")],
        key=lambda x: -x["measurement"]["tokens_per_sec"],
    )
    for t in sortable_kept:
        print("  " + fmt_trial_row(t))
    for t in trials:
        if t.get("failure"):
            print("  " + fmt_trial_row(t))
    return 0


if __name__ == "__main__":
    sys.exit(main())
