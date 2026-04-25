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


def topk_js_divergence(
    ref: list[dict[str, float]],
    cand: list[dict[str, float]],
) -> float:
    """Symmetric, bounded top-K Jensen-Shannon divergence.

    Unlike top-K KL with a hard floor (which inflates a single missing
    token to ~30 KL units), JS works on the **union** of both top-K
    sets, renormalises each side over that union, and averages two
    KLs against the mean distribution. Bounded in ``[0, log(2)]`` so
    one outlier prompt cannot blow the noise ceiling open.

    Returns the mean per-position JS divergence (in nats).
    """
    if len(ref) != len(cand):
        return float("inf")
    if not ref:
        return 0.0
    total = 0.0
    for ref_pos, cand_pos in zip(ref, cand, strict=True):
        union = set(ref_pos) | set(cand_pos)
        if not union:
            continue
        # Renormalise both sides over the union — top-K only gives us
        # exposed mass; assume missing tokens carry zero (NOT floor).
        p_raw = {t: math.exp(ref_pos.get(t, _LOG_FLOOR)) for t in union}
        q_raw = {t: math.exp(cand_pos.get(t, _LOG_FLOOR)) for t in union}
        p_sum = sum(p_raw.values()) or 1.0
        q_sum = sum(q_raw.values()) or 1.0
        p = {t: v / p_sum for t, v in p_raw.items()}
        q = {t: v / q_sum for t, v in q_raw.items()}
        js = 0.0
        for t in union:
            pt, qt = p[t], q[t]
            mt = 0.5 * (pt + qt)
            if pt > 0 and mt > 0:
                js += 0.5 * pt * math.log(pt / mt)
            if qt > 0 and mt > 0:
                js += 0.5 * qt * math.log(qt / mt)
        total += js
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


def _aggregate_self_kl(per: list[float]) -> dict[str, float]:
    """Aggregate sorted per-prompt self-KL into a calibration summary.

    Pure: no I/O. Caller supplies the sorted list. The exposed ``p95``
    is median-capped at 5x median to keep one outlier prompt from
    blowing the gate ceiling open (see ``calibrate_self_kl``).
    """
    if not per:
        raise ValueError("per must be non-empty")
    n = len(per)
    raw_p95 = per[int(n * 0.95)] if n >= 20 else per[-1]
    median = per[n // 2]
    p95 = min(raw_p95, 5.0 * median) if median > 0.0 else raw_p95
    return {
        "n": float(n),
        "mean": sum(per) / n,
        "median": median,
        "p95": p95,
        "raw_p95": raw_p95,
        "max": per[-1],
    }


def calibrate_self_kl(
    endpoint: str,
    model: str,
    prompts: list[str],
    top_k: int = 5,
    max_tokens: int = 32,
    concurrency: int = 4,
) -> dict[str, float]:
    """Gate the endpoint against itself to measure noise-floor KL.

    Two sequential calls to the SAME endpoint for each prompt; any KL is
    pure scheduling/batch-composition noise, not real drift. Returns a
    summary dict with ``mean, median, p95, max, n``.

    The exposed ``p95`` is the **median-capped** 95th percentile —
    ``min(raw_p95, 5 * median)``. With small samples (n<=20) raw p95
    degenerates to the max, and one outlier prompt (a single missing
    top-K token under floor=1e-10 inflates KL by ~30 units per
    mismatch) blows the noise ceiling 50x. Capping at 5x median keeps
    well-behaved distributions intact while bounding outlier damage.
    """
    if not prompts:
        raise ValueError("prompts must be non-empty")

    def _pair(prompt: str) -> float:
        a = fetch_logprobs(endpoint, model, prompt, top_k, max_tokens)
        b = fetch_logprobs(endpoint, model, prompt, top_k, max_tokens)
        return topk_kl_divergence(a, b)

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        per = sorted(ex.map(_pair, prompts))
    return _aggregate_self_kl(per)


def run_gate(
    candidate_endpoint: str,
    reference_endpoint: str,
    model: str,
    prompts: list[str],
    batch_sizes: tuple[int, ...] = (1, 8, 64),
    top_k: int = 5,
    max_tokens: int = 32,
    concurrency: int = 8,
) -> GateResult:
    """End-to-end quality gate: KL across prompts + batch invariance sample.

    Per-prompt KL is computed in parallel via a bounded thread pool so a
    500-prompt gate finishes in seconds rather than minutes. ``concurrency``
    caps how many concurrent (candidate, reference) pairs hit each endpoint
    — leave at 8 or lower so a single-GPU reference replica stays responsive.
    """
    if not prompts:
        raise ValueError("prompts must be non-empty")
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")

    def _kl_for(prompt: str) -> float:
        ref = fetch_logprobs(reference_endpoint, model, prompt, top_k, max_tokens)
        cand = fetch_logprobs(candidate_endpoint, model, prompt, top_k, max_tokens)
        return topk_kl_divergence(ref, cand)

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        per_prompt = list(ex.map(_kl_for, prompts))

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
