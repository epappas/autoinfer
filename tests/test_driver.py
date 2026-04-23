from __future__ import annotations

from pathlib import Path

from autoinfer.harness.driver import build_bench_command, parse_bench_output


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
