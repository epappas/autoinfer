#!/usr/bin/env python3
"""Fetch ShareGPT, sample prompts for the workload trace and quality gate.

Runs inside the Basilica deployment. Writes two files:

- ``<out-dir>/trace/sharegpt_sample.jsonl`` -- trace for ``vllm bench serve``
- ``<out-dir>/quality/500_prompts.jsonl``   -- held-out prompts for the gate

Deterministic: same seed yields the same split every time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from urllib.request import urlopen

DEFAULT_SOURCE_URL = (
    "https://huggingface.co/datasets/anon8231489123/"
    "ShareGPT_Vicuna_unfiltered/resolve/main/"
    "ShareGPT_V3_unfiltered_cleaned_split.json"
)


def fetch_source(url: str, cache_path: Path) -> None:
    if cache_path.exists():
        print(f"cached: {cache_path}")
        return
    print(f"fetching: {url}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as resp, cache_path.open("wb") as f:
        bytes_written = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            bytes_written += len(chunk)
    print(f"wrote {bytes_written / 1e6:.1f} MB -> {cache_path}")


def load_conversations(cache_path: Path) -> list[dict]:
    with cache_path.open() as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError("ShareGPT source is not a JSON array at the top level")
    return raw


def extract_prompt(conv: dict) -> str | None:
    turns = conv.get("conversations") or conv.get("turns") or []
    for turn in turns:
        text = turn.get("value") if isinstance(turn, dict) else None
        if isinstance(text, str) and text.strip():
            return text.strip()
    return None


def write_jsonl(path: Path, prompts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for p in prompts:
            f.write(json.dumps({"prompt": p}) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    print(f"wrote {len(prompts)} prompts -> {path}  sha256:{digest}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("."))
    p.add_argument("--trace-size", type=int, default=500)
    p.add_argument("--quality-size", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    p.add_argument(
        "--cache-path",
        type=Path,
        default=Path("./trace/sharegpt_full.json"),
        help="Where to cache the downloaded ShareGPT JSON.",
    )
    p.add_argument("--min-prompt-chars", type=int, default=20)
    args = p.parse_args()

    fetch_source(args.source_url, args.cache_path)
    convs = load_conversations(args.cache_path)
    pool: list[str] = []
    for c in convs:
        prompt = extract_prompt(c)
        if prompt is not None and len(prompt) >= args.min_prompt_chars:
            pool.append(prompt)

    need = args.trace_size + args.quality_size
    if len(pool) < need:
        print(
            f"WARNING: only {len(pool)} usable prompts for requested {need}",
            file=sys.stderr,
        )

    rng = random.Random(args.seed)
    rng.shuffle(pool)
    trace_prompts = pool[: args.trace_size]
    quality_prompts = pool[args.trace_size : args.trace_size + args.quality_size]

    write_jsonl(args.out_dir / "trace/sharegpt_sample.jsonl", trace_prompts)
    write_jsonl(args.out_dir / "quality/500_prompts.jsonl", quality_prompts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
