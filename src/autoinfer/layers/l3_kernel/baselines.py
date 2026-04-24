"""Reference PyTorch implementations of L3 target ops.

These three functions are autoinfer's **correctness oracle** for L3. An
LLM-proposed kernel candidate must reproduce their output within the
adapter's ``atol``/``rtol`` on the test matrix before it can be timed.

All functions are deliberately straight-line PyTorch, with no fused
ops, no autocast, no custom kernels — so the reference is easy to
audit and reproduces across CPU, CUDA, and any future backends.

Pure; no I/O.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

KernelFn = Callable[..., torch.Tensor]


def rmsnorm_ref(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Root-mean-square layer norm over the last axis.

    ``y = x / sqrt(mean(x^2) + eps) * w`` computed in float32 for
    numerical stability, then cast back to ``x.dtype``.
    """
    if x.ndim < 1:
        raise ValueError("rmsnorm expects x with at least 1 dim")
    if w.shape != x.shape[-1:]:
        raise ValueError(f"rmsnorm weight shape {tuple(w.shape)} != x last dim {x.shape[-1]}")
    orig_dtype = x.dtype
    xf = x.to(torch.float32)
    rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * rms * w.to(torch.float32)).to(orig_dtype)


def silu_mul_ref(a: torch.Tensor) -> torch.Tensor:
    """Gated-MLP SiLU-multiply: split last dim in half, ``silu(x) * y``."""
    if a.ndim < 1 or a.shape[-1] % 2 != 0:
        raise ValueError("silu_mul expects last dim divisible by 2")
    x, y = a.chunk(2, dim=-1)
    return torch.nn.functional.silu(x) * y


def rope_ref(
    q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Rotary-position embedding applied to ``q`` (or ``k``).

    ``q`` has shape ``[B, H, T, D]``; ``cos``/``sin`` have ``[T, D]``.
    The last dim is rotated pairwise (even/odd interleave).
    """
    if q.ndim != 4:
        raise ValueError(f"rope expects q with ndim 4, got {q.ndim}")
    if cos.shape != sin.shape:
        raise ValueError("rope cos/sin shape mismatch")
    d = q.shape[-1]
    if d % 2 != 0:
        raise ValueError("rope last dim must be even")
    if cos.shape[-1] != d:
        raise ValueError(f"rope cos last dim {cos.shape[-1]} != q last dim {d}")
    q_even = q[..., 0::2]
    q_odd = q[..., 1::2]
    c = cos[..., 0::2]
    s = sin[..., 0::2]
    rot_even = q_even * c - q_odd * s
    rot_odd = q_even * s + q_odd * c
    out = torch.empty_like(q)
    out[..., 0::2] = rot_even
    out[..., 1::2] = rot_odd
    return out


REFERENCES: dict[str, KernelFn] = {
    "rmsnorm": rmsnorm_ref,
    "silu_mul": silu_mul_ref,
    "rope": rope_ref,
}


__all__ = ["KernelFn", "REFERENCES", "rmsnorm_ref", "silu_mul_ref", "rope_ref"]
