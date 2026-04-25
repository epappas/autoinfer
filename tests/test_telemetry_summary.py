from __future__ import annotations

from autoinfer.telemetry.summary import _pip_show, capture_hw_context


def test_pip_show_returns_version_for_installed_package() -> None:
    """importlib.metadata path: a known-installed package returns its version."""
    v = _pip_show("autoinfer")
    assert v is not None
    assert v[0].isdigit()


def test_pip_show_returns_none_for_unknown_package() -> None:
    assert _pip_show("definitely-not-a-real-package-xyz12345") is None


def test_capture_hw_context_includes_versions_and_keys() -> None:
    ctx = capture_hw_context()
    assert "python" in ctx
    assert "platform" in ctx
    assert ctx.get("autoinfer_version") is not None
    # torch may or may not be installed in the dev env; presence of the
    # key matters more than its value
    assert "torch_version" in ctx
    assert "vllm_version" in ctx
