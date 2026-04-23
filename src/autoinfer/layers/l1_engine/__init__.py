"""L1 engine-config layer: vLLM EngineArgs + runtime-env search."""

from autoinfer.layers.l1_engine.adapter import (
    L1EngineAdapter,
    compose_measurement,
    query_gpu_memory_used_gb,
)
from autoinfer.layers.l1_engine.surface import (
    CompatRule,
    KnobCatalog,
    KnobSpec,
    build_vllm_serve_args,
    defaults,
    load_catalog,
    to_surrogate_surface,
    violates_constraints,
)

__all__ = [
    "CompatRule",
    "KnobCatalog",
    "KnobSpec",
    "L1EngineAdapter",
    "build_vllm_serve_args",
    "compose_measurement",
    "defaults",
    "load_catalog",
    "query_gpu_memory_used_gb",
    "to_surrogate_surface",
    "violates_constraints",
]
