from __future__ import annotations

from pathlib import Path

import pytest
import torch

from autoinfer.layers.l3_kernel import (
    REFERENCE_SOURCES,
    compile_candidate,
    load_catalog,
    make_inputs,
    reference_seed_configs,
    resolve_dtype,
    to_surrogate_surface,
)
from autoinfer.layers.l3_kernel import (
    test_shapes as get_test_shapes,
)

_REPO_CATALOG = Path(__file__).parent.parent / "src/autoinfer/layers/l3_kernel/knobs.yaml"


def test_repo_catalog_loads() -> None:
    catalog = load_catalog(_REPO_CATALOG)
    for knob in ("target_op", "dtype", "shape_regime"):
        assert knob in catalog.knobs


def test_surrogate_surface_is_categorical() -> None:
    catalog = load_catalog(_REPO_CATALOG)
    surface = to_surrogate_surface(catalog)
    for name, spec in surface.items():
        assert spec["type"] == "categorical", name
        assert spec["values"]


def test_resolve_dtype_known() -> None:
    assert resolve_dtype("float32") is torch.float32
    assert resolve_dtype("bfloat16") is torch.bfloat16


def test_resolve_dtype_unknown_raises() -> None:
    with pytest.raises(ValueError):
        resolve_dtype("float8")


def test_test_shapes_known_ops() -> None:
    for op in ("rmsnorm", "silu_mul", "rope"):
        shapes = get_test_shapes(op, "small")
        assert len(shapes) >= 1
        assert all(s.target_op == op for s in shapes)


def test_test_shapes_unknown_raises() -> None:
    with pytest.raises(ValueError):
        get_test_shapes("softmax", "small")
    with pytest.raises(ValueError):
        get_test_shapes("rmsnorm", "huge")


def test_make_inputs_rmsnorm_shapes() -> None:
    shapes = get_test_shapes("rmsnorm", "small")
    inputs = make_inputs(shapes[0], torch.float32, seed=0)
    assert len(inputs) == 3
    x, w, eps = inputs
    assert x.shape == shapes[0].dims
    assert w.shape == (shapes[0].dims[-1],)
    assert isinstance(eps, float)


def test_make_inputs_silu_mul_shapes() -> None:
    shapes = get_test_shapes("silu_mul", "small")
    inputs = make_inputs(shapes[0], torch.float32, seed=0)
    assert len(inputs) == 1
    assert inputs[0].shape == shapes[0].dims


def test_make_inputs_rope_shapes() -> None:
    shapes = get_test_shapes("rope", "small")
    q, cos, sin = make_inputs(shapes[0], torch.float32, seed=0)
    assert q.shape == shapes[0].dims
    assert cos.shape == (shapes[0].dims[-2], shapes[0].dims[-1])
    assert sin.shape == cos.shape


def test_compile_candidate_uses_shared_temp_dir() -> None:
    """T-13: all compiled candidates share one process-level temp dir;
    new candidates don't accumulate distinct directories."""
    from autoinfer.layers.l3_kernel.surface import _l3_temp_root

    entry_a, src_a = REFERENCE_SOURCES["rmsnorm"]
    entry_b, src_b = REFERENCE_SOURCES["silu_mul"]
    compile_candidate(src_a, entry_a)
    root1 = _l3_temp_root()
    compile_candidate(src_b, entry_b)
    root2 = _l3_temp_root()
    assert root1 is root2, "compile_candidate should reuse a single temp dir"
    assert root1.exists()
    files = sorted(root1.glob("l3_candidate_*.py"))
    assert len(files) >= 2


def test_compile_candidate_returns_callable() -> None:
    entry, src = REFERENCE_SOURCES["rmsnorm"]
    fn = compile_candidate(src, entry)
    assert callable(fn)


def test_compile_candidate_syntax_error_raises() -> None:
    with pytest.raises(ValueError):
        compile_candidate("def bad( :", "bad")


def test_compile_candidate_missing_entry_fn() -> None:
    with pytest.raises(KeyError):
        compile_candidate("x = 1", "missing")


def test_compile_candidate_entry_not_callable() -> None:
    with pytest.raises(KeyError):
        compile_candidate("missing = 42", "missing")


def test_reference_seed_configs_cover_all_ops() -> None:
    seeds = reference_seed_configs()
    ops = {s["target_op"] for s in seeds}
    assert ops == {"rmsnorm", "silu_mul", "rope"}
    for s in seeds:
        assert "source" in s and "entry_fn" in s
