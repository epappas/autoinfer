"""ReferenceReplica determinism-wiring tests.

Only the pure ``build_cmd_env`` surface is covered here — the subprocess
spawn is GPU-gated and validated by the campaign runners. No mocks: the
helper builds the command list + env dict that ``subprocess.Popen``
consumes; assertions are direct equality on those values.
"""

from __future__ import annotations

from autoinfer.harness.replica import ReferenceReplica


def test_defaults_no_seed_no_env_override() -> None:
    """T-36 default: no ``--seed``, no ``VLLM_ENABLE_V1_MULTIPROCESSING``
    override. Backward-compat with pre-T-36 replicas."""
    r = ReferenceReplica(model="m", port=8001)
    cmd, env = r.build_cmd_env(base_env={})
    assert cmd[:3] == ["vllm", "serve", "m"]
    assert "--seed" not in cmd
    assert "VLLM_ENABLE_V1_MULTIPROCESSING" not in env
    assert r.seed is None
    assert r.multiprocessing_v1 is True


def test_seed_emits_cli_arg() -> None:
    r = ReferenceReplica(model="m", port=8001, seed=42)
    cmd, _ = r.build_cmd_env(base_env={})
    assert "--seed" in cmd
    assert cmd[cmd.index("--seed") + 1] == "42"
    assert r.seed == 42


def test_seed_zero_is_explicit_and_emitted() -> None:
    """``seed=0`` is the only "valid + special" case; verify it does
    emit (instead of being treated as falsy and dropped)."""
    r = ReferenceReplica(model="m", port=8001, seed=0)
    cmd, _ = r.build_cmd_env(base_env={})
    assert "--seed" in cmd
    assert cmd[cmd.index("--seed") + 1] == "0"


def test_multiprocessing_v1_false_exports_env_var() -> None:
    """T-36: VLLM_ENABLE_V1_MULTIPROCESSING=0 lands in the process env."""
    r = ReferenceReplica(model="m", port=8001, multiprocessing_v1=False)
    _, env = r.build_cmd_env(base_env={})
    assert env["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"
    assert r.multiprocessing_v1 is False


def test_multiprocessing_v1_true_does_not_export_env_var() -> None:
    r = ReferenceReplica(model="m", port=8001, multiprocessing_v1=True)
    _, env = r.build_cmd_env(base_env={"HOME": "/home/x"})
    assert "VLLM_ENABLE_V1_MULTIPROCESSING" not in env
    # base_env values pass through.
    assert env["HOME"] == "/home/x"


def test_build_cmd_env_does_not_mutate_base_env() -> None:
    base = {"HOME": "/home/x"}
    r = ReferenceReplica(model="m", port=8001, multiprocessing_v1=False)
    r.build_cmd_env(base_env=base)
    assert "VLLM_ENABLE_V1_MULTIPROCESSING" not in base


def test_extra_args_preserved_alongside_seed() -> None:
    r = ReferenceReplica(
        model="m", port=8001, seed=7, extra_args=["--enforce-eager"]
    )
    cmd, _ = r.build_cmd_env(base_env={})
    assert "--enforce-eager" in cmd
    assert "--seed" in cmd


def test_uri_returns_loopback_port() -> None:
    r = ReferenceReplica(model="m", port=9090)
    assert r.uri == "http://127.0.0.1:9090"
