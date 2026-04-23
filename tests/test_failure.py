from __future__ import annotations

from autoinfer.harness.failure import FailureKind, FailureRecord, classify_stderr


def test_classify_oom() -> None:
    assert classify_stderr("torch.cuda.OutOfMemoryError: CUDA out of memory") == FailureKind.OOM


def test_classify_nccl_takes_priority_over_timeout() -> None:
    assert classify_stderr("NCCL watchdog timeout") == FailureKind.NCCL


def test_classify_hang() -> None:
    assert classify_stderr("RuntimeError: deadline exceeded") == FailureKind.HANG


def test_classify_startup() -> None:
    assert classify_stderr("ImportError: cannot import name vllm") == FailureKind.STARTUP


def test_classify_unknown() -> None:
    assert classify_stderr("something completely unexpected") == FailureKind.UNKNOWN


def test_failure_record_to_dict_roundtrip() -> None:
    r = FailureRecord(
        kind=FailureKind.OOM,
        message="oom at trial 7",
        trial_id="t7",
        layer="l1_engine",
        metadata={"attention_backend": "flashinfer"},
    )
    d = r.to_dict()
    assert d["kind"] == "oom"
    assert d["trial_id"] == "t7"
    assert d["metadata"] == {"attention_backend": "flashinfer"}


def test_failure_record_is_frozen() -> None:
    import dataclasses

    r = FailureRecord(FailureKind.OOM, "m", "t", "l1_engine")
    try:
        r.message = "changed"  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("FailureRecord should be frozen")
