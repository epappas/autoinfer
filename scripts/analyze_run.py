#!/usr/bin/env python3
"""Analyze a captured iteration-zero run and emit article-grade summaries.

Reads a directory of per-trial JSONs (and optionally events.jsonl /
results.tsv / run_summary.json / hw_context.json) and prints:

- One-line summary per trial (kept vs failure kind, key knobs, metrics).
- Pareto frontier with all axes.
- Failure-by-kind histogram.
- Phase breakdown (warmstart / surrogate / operator).
- Best-by-tokens_per_sec and tradeoffs.

Usage:
    uv run python scripts/analyze_run.py <run-dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


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


def fmt_trial_row(d: dict[str, Any]) -> str:
    c = d["config"]
    base = (
        f"{d['trial_id']:24s} {phase_of(d['trial_id']):10s}"
        f" attn={c.get('attention_backend',''):10s}"
        f" kv={c.get('kv_cache_dtype',''):8s}"
        f" dtype={str(c.get('dtype','')):9s}"
        f" prefix={str(c.get('enable_prefix_caching','')):5s}"
        f" bt={c.get('max_num_batched_tokens',''):>5}"
        f" ms={c.get('max_num_seqs',''):>4}"
        f" bs={c.get('block_size',''):>3}"
        f" gmu={c.get('gpu_memory_utilization',''):>5}"
    )
    if d.get("measurement"):
        m = d["measurement"]
        return (
            f"KEPT {base}"
            f" -> tok/s={m['tokens_per_sec']:7.1f}"
            f" ttft99={m['ttft_p99_ms']:8.1f}ms"
            f" tpot99={m['tpot_p99_ms']:6.1f}ms"
            f" kl={m['kl_divergence']:5.2f}"
            f" hbm={m['peak_hbm_gb']:5.1f}GB"
        )
    if d.get("failure"):
        return f"FAIL {base} -> {d['failure']['kind']}"
    return f"???? {base}"


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

    kept = [t for t in trials if t.get("measurement")]
    if kept:
        front = pareto_front(kept)
        print(f"PARETO FRONTIER ({len(front)} of {len(kept)} kept):")
        for t in sorted(front, key=lambda x: -x["measurement"]["tokens_per_sec"]):
            print("  " + fmt_trial_row(t))
        print()

        top = max(kept, key=lambda t: t["measurement"]["tokens_per_sec"])
        print("TOP BY TOKENS_PER_SEC:")
        print("  " + fmt_trial_row(top))
        print()

    print("ALL KEPT:")
    for t in sorted(kept, key=lambda x: -x["measurement"]["tokens_per_sec"]):
        print("  " + fmt_trial_row(t))
    return 0


if __name__ == "__main__":
    sys.exit(main())
