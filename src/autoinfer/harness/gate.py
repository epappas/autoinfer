"""Quality gate: live reference replica + logit KL + batch-invariance check.

Implements P8 (live reference replica, not cached values) and P9 (quality
failures are typed). KL math is pure; the HTTP layer is real (no mocks)
and requires a running OpenAI-compatible endpoint for integration tests.
"""

from __future__ import annotations

import concurrent.futures
import math
from dataclasses import dataclass, field
from typing import Any

import httpx

_LOG_FLOOR = math.log(1e-10)


@dataclass(frozen=True)
class GateResult:
    mean_kl: float
    max_kl: float
    per_prompt_kl: tuple[float, ...]
    batch_invariant: bool
    batch_invariance_outputs: tuple[str, ...] = field(default_factory=tuple)

    def passes(self, max_kl: float) -> bool:
        return self.mean_kl <= max_kl and self.batch_invariant


def topk_kl_divergence(
    ref: list[dict[str, float]],
    cand: list[dict[str, float]],
) -> float:
    """Approximate KL(ref || cand) from aligned top-K logprob sequences.

    Each element is ``{token: logprob}``. Misaligned lengths return
    ``float('inf')``. Missing candidate tokens floor at ``1e-10``.
    """
    if len(ref) != len(cand):
        return float("inf")
    if not ref:
        return 0.0
    total = 0.0
    for ref_pos, cand_pos in zip(ref, cand, strict=True):
        for tok, ref_lp in ref_pos.items():
            cand_lp = cand_pos.get(tok, _LOG_FLOOR)
            total += math.exp(ref_lp) * (ref_lp - cand_lp)
    return total / len(ref)


def fetch_logprobs(
    endpoint: str,
    model: str,
    prompt: str,
    top_k: int = 5,
    max_tokens: int = 32,
    timeout_s: float = 60.0,
) -> list[dict[str, float]]:
    """Call OpenAI-compatible /v1/completions; return per-token top-K logprobs."""
    with httpx.Client(timeout=timeout_s) as client:
        resp = client.post(
            f"{endpoint.rstrip('/')}/v1/completions",
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "logprobs": top_k,
                "temperature": 0.0,
            },
        )
        resp.raise_for_status()
        payload = resp.json()
    return _extract_top_logprobs(payload)


def _extract_top_logprobs(payload: dict[str, Any]) -> list[dict[str, float]]:
    choices = payload.get("choices") or []
    if not choices:
        return []
    lp = choices[0].get("logprobs") or {}
    positions = lp.get("top_logprobs") or []
    return [dict(pos or {}) for pos in positions]


def fetch_completion(
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int = 32,
    timeout_s: float = 60.0,
) -> str:
    with httpx.Client(timeout=timeout_s) as client:
        resp = client.post(
            f"{endpoint.rstrip('/')}/v1/completions",
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
            },
        )
        resp.raise_for_status()
        choices = resp.json().get("choices") or []
    text: str = choices[0].get("text", "") if choices else ""
    return text


def batch_invariance_check(
    endpoint: str,
    model: str,
    prompt: str,
    batch_sizes: tuple[int, ...],
    max_tokens: int = 32,
) -> tuple[bool, list[str]]:
    """Send ``prompt`` concurrently at each batch size; all outputs must match."""
    reps: list[str] = [
        _concurrent_first(endpoint, model, prompt, bs, max_tokens) for bs in batch_sizes
    ]
    return (all(o == reps[0] for o in reps), reps)


def _concurrent_first(
    endpoint: str, model: str, prompt: str, concurrency: int, max_tokens: int
) -> str:
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [
            ex.submit(fetch_completion, endpoint, model, prompt, max_tokens)
            for _ in range(concurrency)
        ]
        return futures[0].result()


def run_gate(
    candidate_endpoint: str,
    reference_endpoint: str,
    model: str,
    prompts: list[str],
    batch_sizes: tuple[int, ...] = (1, 8, 64),
    top_k: int = 5,
    max_tokens: int = 32,
) -> GateResult:
    """End-to-end quality gate: KL across prompts + batch invariance sample."""
    if not prompts:
        raise ValueError("prompts must be non-empty")
    per_prompt: list[float] = []
    for p in prompts:
        ref = fetch_logprobs(reference_endpoint, model, p, top_k, max_tokens)
        cand = fetch_logprobs(candidate_endpoint, model, p, top_k, max_tokens)
        per_prompt.append(topk_kl_divergence(ref, cand))
    invariant, reps = batch_invariance_check(
        candidate_endpoint, model, prompts[0], batch_sizes, max_tokens
    )
    return GateResult(
        mean_kl=sum(per_prompt) / len(per_prompt),
        max_kl=max(per_prompt),
        per_prompt_kl=tuple(per_prompt),
        batch_invariant=invariant,
        batch_invariance_outputs=tuple(reps),
    )
