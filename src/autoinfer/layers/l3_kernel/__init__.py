"""L3 kernel-search layer: LLM-proposed kernels vs PyTorch references."""

from autoinfer.layers.l3_kernel.adapter import L3KernelAdapter
from autoinfer.layers.l3_kernel.baselines import (
    REFERENCES,
    KernelFn,
    rmsnorm_ref,
    rope_ref,
    silu_mul_ref,
)
from autoinfer.layers.l3_kernel.surface import (
    REFERENCE_SOURCES,
    KernelCallable,
    ShapeSpec,
    compile_candidate,
    load_catalog,
    make_inputs,
    reference_seed_configs,
    resolve_dtype,
    test_shapes,
    to_surrogate_surface,
)

__all__ = [
    "KernelCallable",
    "KernelFn",
    "L3KernelAdapter",
    "REFERENCES",
    "REFERENCE_SOURCES",
    "ShapeSpec",
    "compile_candidate",
    "load_catalog",
    "make_inputs",
    "reference_seed_configs",
    "resolve_dtype",
    "rmsnorm_ref",
    "rope_ref",
    "silu_mul_ref",
    "test_shapes",
    "to_surrogate_surface",
]
