from __future__ import annotations

from pathlib import Path

import pytest

from autoinfer.layers.l2_topology.surface import (
    L2Catalog,
    L2KnobSpec,
    config_to_deploy_kwargs,
    defaults,
    load_catalog,
    to_surrogate_surface,
)

_REPO_CATALOG = Path(__file__).parent.parent / "src/autoinfer/layers/l2_topology/knobs.yaml"


def test_repo_catalog_loads() -> None:
    catalog = load_catalog(_REPO_CATALOG)
    assert "gpu_type" in catalog.knobs
    assert "gpu_count" in catalog.knobs


def test_defaults_cover_every_knob() -> None:
    catalog = load_catalog(_REPO_CATALOG)
    for name in catalog.knobs:
        assert name in defaults(catalog)


def test_surrogate_surface_shape() -> None:
    catalog = load_catalog(_REPO_CATALOG)
    surface = to_surrogate_surface(catalog)
    for spec in surface.values():
        assert spec["type"] in {"int", "float", "categorical"}


def test_gpu_type_maps_to_gpu_models_list() -> None:
    catalog = load_catalog(_REPO_CATALOG)
    kwargs = config_to_deploy_kwargs(
        {"gpu_type": "RTX A6000", "gpu_count": 2, "dtype": "bfloat16"},
        catalog,
    )
    assert kwargs["gpu_models"] == ["RTX A6000"]
    assert kwargs["gpu_count"] == 2
    # gpu_count also becomes tensor_parallel_size
    assert kwargs["tensor_parallel_size"] == 2
    assert kwargs["dtype"] == "bfloat16"


def test_gpu_count_sets_tensor_parallel_only_if_not_provided() -> None:
    catalog = load_catalog(_REPO_CATALOG)
    kwargs = config_to_deploy_kwargs({"gpu_count": 4}, catalog)
    assert kwargs["tensor_parallel_size"] == 4


def test_unknown_knob_silently_ignored() -> None:
    catalog = load_catalog(_REPO_CATALOG)
    kwargs = config_to_deploy_kwargs(
        {"gpu_type": "RTX A6000", "made_up_knob": 42},
        catalog,
    )
    assert "made_up_knob" not in kwargs


def test_bool_knob_passes_through() -> None:
    catalog = load_catalog(_REPO_CATALOG)
    kwargs = config_to_deploy_kwargs({"enforce_eager": True}, catalog)
    assert kwargs["enforce_eager"] is True


def test_catalog_rejects_empty_categorical() -> None:
    catalog = L2Catalog(
        knobs={
            "x": L2KnobSpec(name="x", type="categorical", default="a", values=None)
        }
    )
    with pytest.raises(ValueError):
        to_surrogate_surface(catalog)
