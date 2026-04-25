from __future__ import annotations

from pathlib import Path

import pytest

from autoinfer.layers.l1_engine.surface import (
    KnobCatalog,
    KnobSpec,
    build_vllm_serve_args,
    defaults,
    load_catalog,
    to_surrogate_surface,
    violates_constraints,
)

_REPO_CATALOG = Path(__file__).parent.parent / "src/autoinfer/layers/l1_engine/knobs.yaml"


def test_repo_catalog_loads() -> None:
    catalog = load_catalog(_REPO_CATALOG)
    assert isinstance(catalog, KnobCatalog)
    assert "max_num_batched_tokens" in catalog.knobs
    assert "attention_backend" in catalog.knobs


def test_repo_catalog_defaults_are_complete() -> None:
    catalog = load_catalog(_REPO_CATALOG)
    defs = defaults(catalog)
    for name in catalog.knobs:
        assert name in defs
        assert defs[name] is not None


def test_to_surrogate_surface_shape() -> None:
    catalog = load_catalog(_REPO_CATALOG)
    surface = to_surrogate_surface(catalog)
    for spec in surface.values():
        assert spec["type"] in {"int", "float", "categorical"}
        if spec["type"] in {"int", "float"}:
            assert "low" in spec and "high" in spec
        elif spec["type"] == "categorical":
            assert "values" in spec and len(spec["values"]) > 0


def test_bool_knob_maps_to_categorical_true_false(tmp_path: Path) -> None:
    path = tmp_path / "k.yaml"
    path.write_text(
        """
knobs:
  enable_x:
    type: bool
    default: false
    vllm_cli_enable: --enable-x
    vllm_cli_disable: --no-enable-x
""".strip()
    )
    catalog = load_catalog(path)
    surface = to_surrogate_surface(catalog)
    assert surface["enable_x"] == {"type": "categorical", "values": [True, False]}


def test_build_args_categorical_passes_value() -> None:
    catalog = load_catalog(_REPO_CATALOG)
    args, env = build_vllm_serve_args(
        "Qwen/Qwen3-8B",
        8000,
        {"max_num_batched_tokens": 4096, "max_num_seqs": 128},
        catalog,
    )
    assert args[:3] == ["vllm", "serve", "Qwen/Qwen3-8B"]
    assert "--max-num-batched-tokens" in args
    i = args.index("--max-num-batched-tokens")
    assert args[i + 1] == "4096"
    assert env == {}


def test_build_args_bool_emits_enable_flag() -> None:
    catalog = load_catalog(_REPO_CATALOG)
    args, _ = build_vllm_serve_args(
        "m", 8000, {"enable_chunked_prefill": True}, catalog
    )
    assert "--enable-chunked-prefill" in args
    assert "--no-enable-chunked-prefill" not in args


def test_build_args_bool_emits_disable_flag() -> None:
    catalog = load_catalog(_REPO_CATALOG)
    args, _ = build_vllm_serve_args(
        "m", 8000, {"enable_chunked_prefill": False}, catalog
    )
    assert "--no-enable-chunked-prefill" in args
    assert "--enable-chunked-prefill" not in args


def test_build_args_env_knob_sets_env_not_cli() -> None:
    catalog = load_catalog(_REPO_CATALOG)
    args, env = build_vllm_serve_args(
        "m", 8000, {"attention_backend": "FLASHINFER"}, catalog
    )
    assert env == {"VLLM_ATTENTION_BACKEND": "FLASHINFER"}
    assert "FLASHINFER" not in args


def test_build_args_none_quantization_skipped() -> None:
    catalog = load_catalog(_REPO_CATALOG)
    args, _ = build_vllm_serve_args(
        "m", 8000, {"quantization": "none"}, catalog
    )
    assert "--quantization" not in args


def test_build_args_unknown_keys_ignored() -> None:
    catalog = load_catalog(_REPO_CATALOG)
    args, _ = build_vllm_serve_args(
        "m", 8000, {"totally_not_a_knob": 42, "max_num_seqs": 64}, catalog
    )
    assert "--max-num-seqs" in args
    assert "totally_not_a_knob" not in " ".join(args)


def test_violates_constraints_fp8_requires_good_backend() -> None:
    catalog = load_catalog(_REPO_CATALOG)
    bad = {"kv_cache_dtype": "fp8", "attention_backend": "XFORMERS"}
    ok = {"kv_cache_dtype": "fp8", "attention_backend": "FLASHINFER"}
    assert "kv_fp8_requires_compatible_backend" in violates_constraints(bad, catalog)
    assert "kv_fp8_requires_compatible_backend" not in violates_constraints(ok, catalog)


def test_violates_constraints_no_violations_on_defaults() -> None:
    catalog = load_catalog(_REPO_CATALOG)
    defs = defaults(catalog)
    assert violates_constraints(defs, catalog) == []


def test_chunked_prefill_off_requires_full_max_model_len_batched_tokens() -> None:
    """Regression: real campaign wasted 2 trials when surrogate proposed
    enable_chunked_prefill=False with batched_tokens=8192, which vLLM
    rejected at startup because Qwen3-8B's max_model_len is 32768."""
    catalog = load_catalog(_REPO_CATALOG)
    too_small = {
        "enable_chunked_prefill": False,
        "max_num_batched_tokens": 8192,
    }
    too_small_2 = {
        "enable_chunked_prefill": False,
        "max_num_batched_tokens": 16384,
    }
    enough = {
        "enable_chunked_prefill": False,
        "max_num_batched_tokens": 32768,
    }
    chunked_on = {
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 8192,
    }
    rule = "chunked_prefill_batched_tokens_bound"
    assert rule in violates_constraints(too_small, catalog)
    assert rule in violates_constraints(too_small_2, catalog)
    assert rule not in violates_constraints(enough, catalog)
    assert rule not in violates_constraints(chunked_on, catalog)


def test_unknown_knob_type_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("knobs:\n  x:\n    type: weird\n    default: 1\n")
    with pytest.raises(ValueError):
        load_catalog(path)


def test_missing_knobs_section_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("other: 1\n")
    with pytest.raises(ValueError):
        load_catalog(path)


def test_categorical_without_values_rejected() -> None:
    catalog = KnobCatalog(
        knobs={
            "x": KnobSpec(
                name="x", type="categorical", default=None, values=None
            )
        }
    )
    with pytest.raises(ValueError):
        to_surrogate_surface(catalog)
