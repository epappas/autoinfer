"""Kernel injection into vLLM (the load-bearing piece of L3).

Replaces the implementation of a vLLM custom op (RMSNorm, SiluAndMul,
RotaryEmbedding) with an LLM-proposed kernel **at runtime**, so the
serving path executes the candidate kernel instead of the default
backend. End-to-end token throughput is then measurable in the same
units as L1 / L2 — that's the unit-comparable metric the
``pareto_eligible=False`` workaround was a stopgap for.

vLLM's dispatch:
- ``CustomOp.__init__`` binds ``self._forward_method = self.dispatch_forward(...)``
  ONCE per instance, picking ``forward_cuda`` / ``forward_native`` etc.
- ``CustomOp.forward(*args, **kwargs)`` then calls ``self._forward_method``.
- ``dispatch_forward`` reads the method off the instance (which inherits
  from the class), so a class-level monkeypatch BEFORE any instance is
  constructed propagates to every instance vLLM later builds.

Hence the injector strategy:
1. Build a wrapper Python entry script.
2. Wrapper imports the candidate kernel, patches the relevant vLLM
   class's ``forward_cuda`` (and ``forward_xpu`` which delegates to it),
   then execs vLLM's CLI.
3. The campaign runner launches the wrapper instead of ``vllm serve``;
   from there everything (model load, scheduler, attention) is normal
   vLLM, except the targeted op runs the candidate kernel.

Falls back to the original ``forward_cuda`` for variants the candidate
doesn't claim to handle (e.g. ``RMSNorm.forward_cuda(x, residual)`` —
the LLM proposer signature is ``rmsnorm_kernel(x, w, eps)``, no fused-
add residual variant). The fallback keeps the patched op correct on
all call paths the model uses.
"""

from __future__ import annotations

from dataclasses import dataclass

# Map from target_op (the L3 surface value) to a (vllm_module, vllm_class,
# adapter_function_name) tuple. The adapter function is module-level in
# this file; it shapes the LLM kernel's call signature into vLLM's.
_TARGET_BINDINGS: dict[str, tuple[str, str, str]] = {
    "rmsnorm": (
        "vllm.model_executor.layers.layernorm",
        "RMSNorm",
        "_rmsnorm_adapter",
    ),
    "silu_mul": (
        "vllm.model_executor.layers.activation",
        "SiluAndMul",
        "_silu_mul_adapter",
    ),
}
"""Supported L3 target ops and their vLLM patch points. RoPE is
omitted from the v1 injector — the vLLM RotaryEmbedding API has many
variants (positions, query/key fused, kv-cache aware) and conflating
them into a single kernel signature is non-trivial. A future revision
will add it once the pattern is proven on the simpler ops."""


SUPPORTED_TARGET_OPS = frozenset(_TARGET_BINDINGS.keys())


@dataclass(frozen=True)
class InjectionPlan:
    """Everything the wrapper script needs to inject the kernel.

    Carries the candidate's ``source`` and ``entry_fn`` (already
    correctness-checked by the L3 adapter) plus the target op binding.
    The wrapper script materialises this into Python that runs at
    ``vllm serve`` startup.
    """

    target_op: str
    entry_fn: str
    source: str

    def __post_init__(self) -> None:
        if self.target_op not in _TARGET_BINDINGS:
            raise ValueError(
                f"unsupported target_op {self.target_op!r}; "
                f"expected one of {sorted(SUPPORTED_TARGET_OPS)}"
            )
        if not self.entry_fn or not self.source:
            raise ValueError("entry_fn and source must be non-empty")


def render_wrapper_script(
    plan: InjectionPlan, vllm_argv: list[str]
) -> str:
    """Generate the Python source of the launcher that vLLM is started under.

    Returns a self-contained script string. The campaign runner writes
    it to a temp .py file (so ``@triton.jit``'s ``inspect.getsource``
    works — same constraint as ``compile_candidate``) and execs it.

    ``vllm_argv`` is the arg list the launcher will assign to
    ``sys.argv`` before invoking vLLM's CLI. Usually it's
    ``['vllm', 'serve', '<model>', '--port', '<n>', ...]`` — the L3
    adapter assembles it from the trial config the same way L1 does.
    """
    module, cls, adapter = _TARGET_BINDINGS[plan.target_op]
    argv_repr = repr(list(vllm_argv))
    return _WRAPPER_TEMPLATE.format(
        kernel_source=plan.source,
        entry_fn=plan.entry_fn,
        vllm_module=module,
        vllm_class=cls,
        adapter_fn=adapter,
        argv=argv_repr,
        adapters_section=_ADAPTERS_SOURCE,
    )


_ADAPTERS_SOURCE = '''
def _rmsnorm_adapter(orig_forward_cuda, kernel_fn):
    """Wrap an LLM-proposed rmsnorm kernel for vLLM's RMSNorm.forward_cuda.

    LLM signature: ``kernel_fn(x, w, eps) -> Tensor``.
    vLLM signature: ``forward_cuda(self, x, residual=None) -> Tensor | tuple``.

    The fused-add residual variant isn't covered by the LLM kernel so
    we fall back to the original vLLM impl when residual is set.
    """

    def patched(self, x, residual=None):
        if residual is None:
            return kernel_fn(x, self.weight, self.variance_epsilon)
        return orig_forward_cuda(self, x, residual)

    return patched


def _silu_mul_adapter(orig_forward_cuda, kernel_fn):
    """Wrap an LLM-proposed silu_mul kernel for vLLM's SiluAndMul.forward_cuda.

    LLM signature: ``kernel_fn(a) -> Tensor``.
    vLLM signature: ``forward_cuda(self, x) -> Tensor``.
    """

    def patched(self, x):
        return kernel_fn(x)

    return patched
'''


# The wrapper template runs INSIDE the campaign container. It must be
# self-contained (no imports from autoinfer) because the L3 trial
# subprocess starts before autoinfer's package code is on the path —
# only what's installed in the venv (vllm, torch, triton) is available.
_WRAPPER_TEMPLATE = '''# Auto-generated by autoinfer.layers.l3_kernel.injector
# DO NOT EDIT BY HAND — regenerated per trial.
import math
import sys

import torch

try:
    import triton
    import triton.language as tl
except (ImportError, RuntimeError):
    triton = None
    tl = None

# === LLM-proposed kernel source ===
{kernel_source}
# === end kernel ===

{adapters_section}

def _patch_vllm_op():
    import importlib

    mod = importlib.import_module("{vllm_module}")
    cls = getattr(mod, "{vllm_class}")
    orig_forward_cuda = cls.forward_cuda
    kernel_fn = {entry_fn}
    cls.forward_cuda = {adapter_fn}(orig_forward_cuda, kernel_fn)
    # forward_xpu delegates to forward_cuda by default; the patch flows
    # through automatically. forward_native is left alone so the gate's
    # CPU fallback still uses vLLM's reference behavior.
    print(
        "[autoinfer.l3.injector] patched {vllm_module}.{vllm_class}.forward_cuda "
        f"with kernel '{entry_fn}'",
        flush=True,
    )


_patch_vllm_op()

sys.argv = {argv}
from vllm.entrypoints.cli.main import main as _vllm_main

raise SystemExit(_vllm_main())
'''


__all__ = [
    "InjectionPlan",
    "SUPPORTED_TARGET_OPS",
    "render_wrapper_script",
]
