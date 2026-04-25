"""Tests for the kernel injector — pure script-generation logic.

The injector's runtime effect (patching vLLM in a subprocess) is
covered by container-side smoke tests, not unit tests. Here we verify
the script we emit is syntactically valid Python, contains the kernel
source, and wires the target op + adapter correctly.
"""

from __future__ import annotations

import pytest

from autoinfer.layers.l3_kernel.injector import (
    SUPPORTED_TARGET_OPS,
    InjectionPlan,
    render_wrapper_script,
)


def test_supported_ops_at_least_rmsnorm_and_silu() -> None:
    assert "rmsnorm" in SUPPORTED_TARGET_OPS
    assert "silu_mul" in SUPPORTED_TARGET_OPS


def test_injection_plan_rejects_unknown_op() -> None:
    with pytest.raises(ValueError):
        InjectionPlan(target_op="softmax", entry_fn="f", source="def f(x): return x")


def test_injection_plan_rejects_empty_source() -> None:
    with pytest.raises(ValueError):
        InjectionPlan(target_op="rmsnorm", entry_fn="f", source="")


def test_injection_plan_rejects_empty_entry_fn() -> None:
    with pytest.raises(ValueError):
        InjectionPlan(target_op="rmsnorm", entry_fn="", source="def f(x): return x")


def test_render_script_compiles_as_valid_python() -> None:
    """The generated wrapper must be syntactically valid Python so the
    subprocess can exec it without surprises at startup."""
    plan = InjectionPlan(
        target_op="rmsnorm",
        entry_fn="my_kernel",
        source="def my_kernel(x, w, eps):\n    return x * w",
    )
    script = render_wrapper_script(plan, ["vllm", "serve", "Qwen/Qwen3-8B"])
    # Will raise SyntaxError if invalid
    compile(script, "<wrapper>", "exec")


def test_render_script_includes_kernel_source() -> None:
    plan = InjectionPlan(
        target_op="rmsnorm",
        entry_fn="my_kernel",
        source="def my_kernel(x, w, eps):\n    return x * w",
    )
    script = render_wrapper_script(plan, ["vllm", "serve", "Qwen/Qwen3-8B"])
    assert "def my_kernel(x, w, eps):" in script
    assert "return x * w" in script


def test_render_script_targets_rmsnorm_module() -> None:
    plan = InjectionPlan(
        target_op="rmsnorm",
        entry_fn="my_kernel",
        source="def my_kernel(x, w, eps): return x",
    )
    script = render_wrapper_script(plan, ["vllm", "serve", "Q"])
    assert "vllm.model_executor.layers.layernorm" in script
    assert "RMSNorm" in script
    assert "_rmsnorm_adapter" in script


def test_render_script_targets_silu_mul_module() -> None:
    plan = InjectionPlan(
        target_op="silu_mul",
        entry_fn="my_kernel",
        source="def my_kernel(a): return a",
    )
    script = render_wrapper_script(plan, ["vllm", "serve", "Q"])
    assert "vllm.model_executor.layers.activation" in script
    assert "SiluAndMul" in script
    assert "_silu_mul_adapter" in script


def test_render_script_assigns_argv() -> None:
    plan = InjectionPlan(
        target_op="rmsnorm",
        entry_fn="f",
        source="def f(x, w, eps): return x",
    )
    argv = ["vllm", "serve", "Qwen/Qwen3-8B", "--port", "8000"]
    script = render_wrapper_script(plan, argv)
    # repr() of the list should appear verbatim in the script
    assert repr(argv) in script
    assert "sys.argv" in script


def test_render_script_dispatches_through_vllm_cli_entry() -> None:
    plan = InjectionPlan(
        target_op="rmsnorm",
        entry_fn="f",
        source="def f(x, w, eps): return x",
    )
    script = render_wrapper_script(plan, ["vllm", "serve", "Q"])
    assert "vllm.entrypoints.cli.main" in script


def test_adapter_signatures_are_compatible_with_vllm() -> None:
    """The wrapper's adapters must produce callables matching vLLM's
    forward_cuda signatures. We exec the adapter section in isolation
    and verify the patched function shape."""
    # We exercise the adapter section in isolation; the rendered script
    # itself is just a syntax check (covered by test_render_script_compiles_as_valid_python).
    InjectionPlan(
        target_op="rmsnorm",
        entry_fn="kfn",
        source="def kfn(x, w, eps): return x + w",
    )
    from autoinfer.layers.l3_kernel.injector import _ADAPTERS_SOURCE  # noqa: SLF001

    ns: dict[str, object] = {}
    exec("def kfn(x, w, eps): return x + w\n" + _ADAPTERS_SOURCE, ns)
    rmsnorm_adapter = ns["_rmsnorm_adapter"]
    assert callable(rmsnorm_adapter)
    # Build a fake "original forward_cuda" + "kernel"
    calls: list[str] = []

    def fake_orig(self, x, residual):
        calls.append("fallback")
        return ("fallback_out", residual)

    def fake_kernel(x, w, eps):
        calls.append(f"kernel:eps={eps}")
        return "kernel_out"

    patched = rmsnorm_adapter(fake_orig, fake_kernel)

    class FakeRMSNorm:
        weight = "w"
        variance_epsilon = 1e-6

    fake_self = FakeRMSNorm()
    # residual=None → calls our kernel
    out = patched(fake_self, "x")
    assert out == "kernel_out"
    assert calls == ["kernel:eps=1e-06"]
    # residual=tensor → falls back to original
    out2 = patched(fake_self, "x", residual="r")
    assert out2 == ("fallback_out", "r")
    assert calls[-1] == "fallback"


def test_silu_mul_adapter_passes_through_signature() -> None:
    from autoinfer.layers.l3_kernel.injector import _ADAPTERS_SOURCE  # noqa: SLF001

    ns: dict[str, object] = {}
    exec(_ADAPTERS_SOURCE, ns)
    silu_mul_adapter = ns["_silu_mul_adapter"]

    def fake_orig(self, x):
        raise AssertionError("orig should not be called for silu_mul")

    def fake_kernel(a):
        return f"silu({a})"

    patched = silu_mul_adapter(fake_orig, fake_kernel)
    out = patched(None, "input_tensor")
    assert out == "silu(input_tensor)"


def test_wrapper_guards_main_for_multiprocessing_spawn() -> None:
    """Regression: vLLM v1's engine-core subprocess uses
    multiprocessing.spawn, which RE-RUNS the parent's main script in
    each child. Without ``if __name__ == "__main__"`` around the
    serve invocation, the child also calls _vllm_main() and tries to
    spawn another engine, infinite-recursing. Smoke 2026-04-25 23:21
    surfaced this.

    The patch itself MUST run unconditionally in both parent and
    child — the subprocess loads vllm independently and needs the
    same forward_cuda override."""
    plan = InjectionPlan(
        target_op="rmsnorm",
        entry_fn="f",
        source="def f(x, w, eps): return x",
    )
    script = render_wrapper_script(plan, ["vllm", "serve", "Q"])
    assert "_patch_vllm_op()" in script
    assert 'if __name__ == "__main__":' in script
    # serve invocation is INSIDE the __main__ guard
    main_idx = script.index('if __name__ == "__main__":')
    # Look for the actual SystemExit-wrapped invocation, not the
    # string occurring in the explanatory comment above the guard.
    serve_idx = script.index("SystemExit(_vllm_main())")
    assert serve_idx > main_idx, "serve invocation must be inside __main__ guard"
    # patch is OUTSIDE the guard
    patch_idx = script.index("_patch_vllm_op()")
    assert patch_idx < main_idx, "patch must run unconditionally (outside __main__ guard)"


def test_render_script_no_unsubstituted_placeholders() -> None:
    """All ``{name}`` placeholders must be filled — leftover ``{...}``
    in the rendered script would crash at exec time."""
    plan = InjectionPlan(
        target_op="rmsnorm",
        entry_fn="f",
        source="def f(x, w, eps): return x",
    )
    script = render_wrapper_script(plan, ["vllm", "serve", "Q"])
    # The kernel source might legitimately contain "{" (e.g. set/dict
    # literals), but no curly-braced placeholders matching the template
    # variable names should remain.
    for name in ("kernel_source", "entry_fn", "vllm_module",
                 "vllm_class", "adapter_fn", "argv", "adapters_section"):
        assert "{" + name + "}" not in script
