from __future__ import annotations

from pathlib import Path

from autoinfer.harness.driver import (
    _format_goodput_args,
    build_bench_command,
    parse_bench_output,
)


def test_parse_full_output() -> None:
    payload = {
        "output_throughput": 1234.5,
        "request_throughput": 10.2,
        "median_ttft_ms": 120.0,
        "p95_ttft_ms": 180.0,
        "p99_ttft_ms": 240.0,
        "median_tpot_ms": 30.0,
        "p95_tpot_ms": 45.0,
        "p99_tpot_ms": 60.0,
        "request_goodput": 9.0,
    }
    r = parse_bench_output(payload)
    assert r.tokens_per_sec == 1234.5
    assert r.request_throughput == 10.2
    assert r.ttft_ms == {"p50": 120.0, "p95": 180.0, "p99": 240.0}
    assert r.tpot_ms == {"p50": 30.0, "p95": 45.0, "p99": 60.0}
    assert r.goodput_req_per_sec == 9.0
    assert r.raw is payload


def test_parse_falls_back_to_total_token_throughput() -> None:
    payload = {"total_token_throughput": 500.0}
    r = parse_bench_output(payload)
    assert r.tokens_per_sec == 500.0


def test_parse_falls_back_to_mean_when_median_missing() -> None:
    payload = {"mean_ttft_ms": 100.0, "mean_tpot_ms": 25.0}
    r = parse_bench_output(payload)
    assert r.ttft_ms["p50"] == 100.0
    assert r.tpot_ms["p50"] == 25.0
    assert r.ttft_ms["p95"] == 0.0


def test_parse_missing_goodput_falls_back_to_throughput() -> None:
    payload = {"request_throughput": 7.0}
    r = parse_bench_output(payload)
    assert r.goodput_req_per_sec == 7.0


def test_build_bench_command_random_default() -> None:
    cmd = build_bench_command(
        endpoint="http://localhost:8000",
        trace_path=Path("/tmp/trace.jsonl"),
        model="Qwen/Qwen3-8B",
        result_dir=Path("/tmp/out"),
        result_name="bench.json",
    )
    assert cmd[:3] == ["vllm", "bench", "serve"]
    assert "--dataset-name" in cmd
    assert cmd[cmd.index("--dataset-name") + 1] == "random"
    assert "--random-input-len" in cmd
    assert "--random-output-len" in cmd
    # random mode does not pass --dataset-path
    assert "--dataset-path" not in cmd
    assert "--save-result" in cmd


def test_build_bench_command_custom_passes_dataset_path() -> None:
    cmd = build_bench_command(
        endpoint="http://localhost:8000",
        trace_path=Path("/tmp/trace.jsonl"),
        model="Qwen/Qwen3-8B",
        result_dir=Path("/tmp/out"),
        result_name="bench.json",
        dataset_name="custom",
    )
    assert "--dataset-path" in cmd
    assert "/tmp/trace.jsonl" in cmd
    # custom mode does not emit random-specific flags
    assert "--random-input-len" not in cmd


def test_build_bench_command_with_rate() -> None:
    cmd = build_bench_command(
        endpoint="http://x",
        trace_path=Path("t"),
        model="m",
        result_dir=Path("d"),
        result_name="r",
        num_prompts=500,
        request_rate=8.0,
    )
    assert "--num-prompts" in cmd
    i = cmd.index("--num-prompts")
    assert cmd[i + 1] == "500"
    assert "--request-rate" in cmd
    j = cmd.index("--request-rate")
    assert cmd[j + 1] == "8.0"


# ---------------------------------------------------------------------------
# T-34 — --goodput SLO + per-trial seed plumbing.
# ---------------------------------------------------------------------------


def test_format_goodput_args_full_set() -> None:
    out = _format_goodput_args({"TTFT": 800.0, "TPOT": 80.0, "E2E": 500.0})
    assert out == ["--goodput", "TTFT:800", "TPOT:80", "E2E:500"]


def test_format_goodput_args_subset_preserves_canonical_order() -> None:
    out = _format_goodput_args({"E2E": 500.0, "TTFT": 800.0})
    assert out == ["--goodput", "TTFT:800", "E2E:500"]


def test_format_goodput_args_empty_dict_returns_empty() -> None:
    assert _format_goodput_args({}) == []


def test_format_goodput_args_fractional_values_use_g_format() -> None:
    """Tokens like ``TTFT:799.5`` are valid vLLM syntax; the ``:g``
    formatter drops trailing zeros and keeps fractional accuracy."""
    out = _format_goodput_args({"TTFT": 799.5})
    assert out == ["--goodput", "TTFT:799.5"]


def test_build_bench_command_emits_goodput_when_slo_set() -> None:
    cmd = build_bench_command(
        endpoint="http://x",
        trace_path=Path("t"),
        model="m",
        result_dir=Path("d"),
        result_name="r",
        goodput_slo_ms={"TTFT": 800.0, "TPOT": 80.0, "E2E": 500.0},
    )
    assert "--goodput" in cmd
    i = cmd.index("--goodput")
    assert cmd[i + 1 : i + 4] == ["TTFT:800", "TPOT:80", "E2E:500"]


def test_build_bench_command_no_goodput_when_slo_unset() -> None:
    cmd = build_bench_command(
        endpoint="http://x",
        trace_path=Path("t"),
        model="m",
        result_dir=Path("d"),
        result_name="r",
    )
    assert "--goodput" not in cmd


def test_build_bench_command_seed_emits_cli() -> None:
    """T-35: per-trial determinism via --seed."""
    cmd = build_bench_command(
        endpoint="http://x",
        trace_path=Path("t"),
        model="m",
        result_dir=Path("d"),
        result_name="r",
        seed=42,
    )
    assert "--seed" in cmd
    assert cmd[cmd.index("--seed") + 1] == "42"


def test_build_bench_command_no_seed_when_unset() -> None:
    cmd = build_bench_command(
        endpoint="http://x",
        trace_path=Path("t"),
        model="m",
        result_dir=Path("d"),
        result_name="r",
    )
    assert "--seed" not in cmd


def test_parse_goodput_present_overrides_request_throughput() -> None:
    """When vLLM returns a non-null ``request_goodput`` (= SLO mode on),
    it takes precedence over request_throughput in DriverResult."""
    payload = {
        "request_goodput": 6.5,
        "request_throughput": 9.0,
    }
    r = parse_bench_output(payload)
    assert r.goodput_req_per_sec == 6.5
