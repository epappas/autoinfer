"""L3 kernel-search layer: LLM-proposed kernels vs PyTorch references."""

from autoinfer.layers.l3_kernel.adapter import L3KernelAdapter
from autoinfer.layers.l3_kernel.baselines import (
    REFERENCES,
    KernelFn,
    rmsnorm_ref,
    rope_ref,
    silu_mul_ref,
)
from autoinfer.layers.l3_kernel.injector import (
    SUPPORTED_TARGET_OPS,
    InjectionPlan,
    render_wrapper_script,
)
from autoinfer.layers.l3_kernel.proposer import (
    KernelProposer,
    build_kernel_prompt,
    parse_kernel_blocks,
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
    source_uses_triton,
    test_shapes,
    to_surrogate_surface,
)

__all__ = [
    "InjectionPlan",
    "KernelCallable",
    "KernelFn",
    "KernelProposer",
    "L3KernelAdapter",
    "REFERENCES",
    "REFERENCE_SOURCES",
    "SUPPORTED_TARGET_OPS",
    "ShapeSpec",
    "build_kernel_prompt",
    "compile_candidate",
    "load_catalog",
    "make_inputs",
    "parse_kernel_blocks",
    "reference_seed_configs",
    "render_wrapper_script",
    "resolve_dtype",
    "rmsnorm_ref",
    "rope_ref",
    "silu_mul_ref",
    "source_uses_triton",
    "test_shapes",
    "to_surrogate_surface",
]
