"""L2 topology layer: per-trial Basilica deployments with varied GPU classes."""

from autoinfer.layers.l2_topology.adapter import L2TopologyAdapter
from autoinfer.layers.l2_topology.surface import (
    L2Catalog,
    L2CompatRule,
    L2KnobSpec,
    config_to_deploy_kwargs,
    defaults,
    load_catalog,
    to_surrogate_surface,
)

__all__ = [
    "L2Catalog",
    "L2CompatRule",
    "L2KnobSpec",
    "L2TopologyAdapter",
    "config_to_deploy_kwargs",
    "defaults",
    "load_catalog",
    "to_surrogate_surface",
]
