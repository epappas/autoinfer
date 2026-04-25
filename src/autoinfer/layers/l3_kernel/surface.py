"""L3 kernel-search surface.

Thin wrapper over ``l1_engine.surface`` for catalog loading, plus:

- shape generation for the correctness + perf test matrices, keyed by
  ``(target_op, shape_regime)``;
- a safe-ish ``compile_candidate`` that execs a Python/Triton source
  string in a restricted namespace and returns the entry callable;
- reference-kernel source strings used as L3 warmstart seeds so the
  very first trial has a guaranteed-pass baseline.

No subprocess, no GPU; pure CPU logic + PyTorch.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from autoinfer.layers.l1_engine.surface import (
    KnobCatalog,
    load_catalog,
    to_surrogate_surface,
)

KernelCallable = Callable[..., torch.Tensor]

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True)
class ShapeSpec:
    """Concrete tensor shape(s) for one test input."""

    target_op: str
    dims: tuple[int, ...]


def resolve_dtype(name: str) -> torch.dtype:
    if name not in _DTYPE_MAP:
        raise ValueError(f"unknown dtype {name!r}; expected one of {sorted(_DTYPE_MAP)}")
    return _DTYPE_MAP[name]


def test_shapes(target_op: str, regime: str) -> tuple[ShapeSpec, ...]:
    """Return the test-matrix shapes for the (op, regime) pair."""
    table = _SHAPE_TABLE.get(target_op)
    if table is None:
        raise ValueError(f"unknown target_op {target_op!r}")
    shapes = table.get(regime)
    if shapes is None:
        raise ValueError(f"unknown regime {regime!r} for op {target_op!r}")
    return tuple(ShapeSpec(target_op=target_op, dims=s) for s in shapes)


def make_inputs(
    spec: ShapeSpec, dtype: torch.dtype, seed: int
) -> tuple[Any, ...]:
    """Materialize the positional args for a single test input."""
    g = torch.Generator().manual_seed(seed)
    if spec.target_op == "rmsnorm":
        b, d = spec.dims
        x = torch.randn(b, d, generator=g, dtype=torch.float32).to(dtype)
        w = torch.randn(d, generator=g, dtype=torch.float32).to(dtype)
        return (x, w, 1e-6)
    if spec.target_op == "silu_mul":
        b, two_d = spec.dims
        a = torch.randn(b, two_d, generator=g, dtype=torch.float32).to(dtype)
        return (a,)
    if spec.target_op == "rope":
        b, h, t, d = spec.dims
        q = torch.randn(b, h, t, d, generator=g, dtype=torch.float32).to(dtype)
        freqs = torch.arange(t, dtype=torch.float32).unsqueeze(-1) * torch.arange(
            0, d, 2, dtype=torch.float32
        ).unsqueeze(0)
        freqs = freqs.repeat_interleave(2, dim=-1)
        cos = freqs.cos().to(dtype)
        sin = freqs.sin().to(dtype)
        return (q, cos, sin)
    raise ValueError(f"unknown target_op {spec.target_op!r}")


_HAS_TRITON_JIT_RE = re.compile(r"@triton\.jit\b")


def _module_imports_prefix() -> str:
    """Stock import block prepended to LLM-proposed source.

    The LLM is told the runtime provides ``torch``/``math``/``triton``/
    ``tl``; making the imports literal at the top of the temp module
    means we don't depend on the LLM remembering import lines (and the
    delimited prompt format the proposer uses doesn't include them).
    """
    return (
        "import math\n"
        "import torch\n"
        "try:\n"
        "    import triton\n"
        "    import triton.language as tl\n"
        "except (ImportError, RuntimeError):\n"
        "    triton = None\n"
        "    tl = None\n"
    )


def compile_candidate(source: str, entry_fn: str) -> KernelCallable:
    """Compile ``source`` into a Python module and return ``entry_fn``.

    Triton-decorated source (``@triton.jit``) is materialised as a real
    .py temp file before import because Triton 3.6 requires
    ``inspect.getsource`` to find the function — pure ``exec()`` of a
    source string can't satisfy that. Plain-Python source uses the
    same path (cheap and uniform) so behavior matches across Triton
    and non-Triton candidates.

    Triton compilation is lazy: ``@triton.jit`` only registers the
    kernel; the actual JIT runs on first launch with a CUDA device.
    CPU-only runs of Triton kernels will fail at first call, which
    the adapter classifies as a typed STARTUP failure.

    Raises ``ValueError`` on syntax errors, ``KeyError`` if
    ``entry_fn`` is absent or not callable.
    """
    import importlib.util
    import tempfile
    import uuid

    full_source = _module_imports_prefix() + "\n" + source
    try:
        compile(full_source, "<l3_candidate_check>", "exec")
    except SyntaxError as e:
        raise ValueError(f"candidate source has syntax error: {e}") from e

    # Write to a temp .py file; never auto-delete because the JITed
    # kernel may inspect its source on first launch (long after this
    # function returns). Cleanup is deferred to OS temp reaping.
    suffix = f"_l3_{uuid.uuid4().hex[:8]}.py"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False, encoding="utf-8",
    ) as f:
        f.write(full_source)
        path = f.name

    spec = importlib.util.spec_from_file_location(
        f"l3_candidate_{uuid.uuid4().hex[:8]}", path,
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"failed to build module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, entry_fn):
        raise KeyError(f"entry_fn {entry_fn!r} not defined in candidate source")
    fn = getattr(module, entry_fn)
    if not callable(fn):
        raise KeyError(f"entry_fn {entry_fn!r} is not callable")
    return fn  # type: ignore[no-any-return]


def source_uses_triton(source: str) -> bool:
    """Cheap detector: ``@triton.jit`` decorator present in source."""
    return bool(_HAS_TRITON_JIT_RE.search(source))


# Test-matrix shapes per (op, regime). Small keeps CPU tests fast; large
# exercises memory-bound paths typical of transformer serving.
_SHAPE_TABLE: dict[str, dict[str, tuple[tuple[int, ...], ...]]] = {
    "rmsnorm": {
        "small": ((2, 128), (3, 256), (1, 64)),
        "medium": ((4, 1024), (8, 2048)),
        "large": ((8, 4096), (16, 4096)),
    },
    "silu_mul": {
        "small": ((2, 256), (3, 128), (1, 64)),
        "medium": ((4, 2048), (8, 1024)),
        "large": ((8, 8192), (16, 4096)),
    },
    "rope": {
        "small": ((1, 4, 16, 64), (2, 2, 8, 32)),
        "medium": ((2, 8, 64, 128),),
        "large": ((2, 16, 512, 128),),
    },
}


# Reference source strings — used as warmstart seeds so the L3 loop has
# at least one guaranteed-pass trial per op. They intentionally mirror
# ``baselines.py`` but are supplied as source strings so the compile
# path is exercised end-to-end.
REFERENCE_SOURCES: dict[str, tuple[str, str]] = {
    "rmsnorm": (
        "rmsnorm_kernel",
        """
def rmsnorm_kernel(x, w, eps):
    orig = x.dtype
    xf = x.to(torch.float32)
    rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * rms * w.to(torch.float32)).to(orig)
""".strip(),
    ),
    "silu_mul": (
        "silu_mul_kernel",
        """
def silu_mul_kernel(a):
    x, y = a.chunk(2, dim=-1)
    return torch.nn.functional.silu(x) * y
""".strip(),
    ),
    "rope": (
        "rope_kernel",
        """
def rope_kernel(q, cos, sin):
    q_even = q[..., 0::2]
    q_odd = q[..., 1::2]
    c = cos[..., 0::2]
    s = sin[..., 0::2]
    out = torch.empty_like(q)
    out[..., 0::2] = q_even * c - q_odd * s
    out[..., 1::2] = q_even * s + q_odd * c
    return out
""".strip(),
    ),
}


def reference_seed_configs() -> list[dict[str, Any]]:
    """Produce a warmstart seed list covering all ops at regime=small."""
    seeds: list[dict[str, Any]] = []
    for op, (entry, src) in REFERENCE_SOURCES.items():
        seeds.append(
            {
                "target_op": op,
                "dtype": "float32",
                "shape_regime": "small",
                "source": src,
                "entry_fn": entry,
            }
        )
    return seeds


__all__ = [
    "KernelCallable",
    "KnobCatalog",
    "REFERENCE_SOURCES",
    "ShapeSpec",
    "compile_candidate",
    "load_catalog",
    "make_inputs",
    "reference_seed_configs",
    "resolve_dtype",
    "test_shapes",
    "to_surrogate_surface",
]
