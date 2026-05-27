from __future__ import annotations

from pathlib import Path

import pytest

from autoinfer.harness.driver import (
    DriverResult,
    _format_goodput_args,
    _with_chosen_rate,
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


def test_build_bench_command_emits_percentile_metrics_including_e2el() -> None:
    """T-40: vLLM bench's ``--save-result`` JSON only writes percentile
    fields for metrics listed in ``--percentile-metrics``. Without an
    explicit ``e2el`` in the list, the JSON has no ``p99_e2el_ms`` field
    and rate-search reads 0 → false negative → infinite iteration.
    Pin that all four metrics flow through every bench invocation."""
    cmd = build_bench_command(
        endpoint="http://x",
        trace_path=Path("t"),
        model="m",
        result_dir=Path("d"),
        result_name="r",
    )
    assert "--percentile-metrics" in cmd
    idx = cmd.index("--percentile-metrics")
    assert cmd[idx + 1] == "ttft,tpot,itl,e2el"


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
    """vLLM's check_goodput_args accepts lowercase metric names only
    (ttft, tpot, e2el). Dict input still uses upper-case keys for API
    backwards-compat; lowercase translation happens at emission. C04a
    attempt 6 (2026-05-26) confirmed: ``--goodput E2E:500`` rejected
    by ``vllm bench serve``."""
    out = _format_goodput_args({"TTFT": 800.0, "TPOT": 80.0, "E2E": 500.0})
    assert out == ["--goodput", "ttft:800", "tpot:80", "e2el:500"]


def test_format_goodput_args_subset_preserves_canonical_order() -> None:
    out = _format_goodput_args({"E2E": 500.0, "TTFT": 800.0})
    assert out == ["--goodput", "ttft:800", "e2el:500"]


def test_format_goodput_args_empty_dict_returns_empty() -> None:
    assert _format_goodput_args({}) == []


def test_format_goodput_args_fractional_values_use_g_format() -> None:
    """Tokens like ``ttft:799.5`` are valid vLLM syntax; the ``:g``
    formatter drops trailing zeros and keeps fractional accuracy."""
    out = _format_goodput_args({"TTFT": 799.5})
    assert out == ["--goodput", "ttft:799.5"]


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
    assert cmd[i + 1 : i + 4] == ["ttft:800", "tpot:80", "e2el:500"]


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


def test_parse_extracts_e2el_percentiles() -> None:
    """T-38: e2el percentiles needed for rate-search SLO check."""
    payload = {
        "p99_e2el_ms": 494.0,
        "median_e2el_ms": 320.0,
        "request_throughput": 21.0,
    }
    r = parse_bench_output(payload)
    assert r.e2el_ms["p99"] == 494.0
    assert r.e2el_ms["p50"] == 320.0


def test_with_chosen_rate_preserves_all_fields_and_sets_rate() -> None:
    """T-38: ``_with_chosen_rate`` produces a copy of DriverResult with
    chosen_request_rate set; all other fields preserved."""
    src = DriverResult(
        tokens_per_sec=100.0,
        request_throughput=21.39,
        ttft_ms={"p99": 126.0},
        tpot_ms={"p99": 22.0},
        e2el_ms={"p99": 494.0},
        goodput_req_per_sec=21.39,
        raw={"some": "field"},
    )
    out = _with_chosen_rate(src, 23.0)
    assert out.chosen_request_rate == 23.0
    assert out.tokens_per_sec == src.tokens_per_sec
    assert out.request_throughput == src.request_throughput
    assert out.ttft_ms == src.ttft_ms
    assert out.e2el_ms == src.e2el_ms
    assert out.goodput_req_per_sec == src.goodput_req_per_sec
    assert out.raw == src.raw


def test_with_chosen_rate_accepts_inf() -> None:
    """When initial inf-rate measurement meets SLO, chosen_request_rate
    = float('inf')."""
    src = DriverResult(
        tokens_per_sec=100.0,
        request_throughput=21.39,
        ttft_ms={},
        tpot_ms={},
        e2el_ms={"p99": 480.0},
        goodput_req_per_sec=21.39,
        raw={},
    )
    out = _with_chosen_rate(src, float("inf"))
    assert out.chosen_request_rate == float("inf")


# ---------------------------------------------------------------------------
# T-38 — rate-down search loop (mirroring auto_tune.sh's algorithm).
#
# The rate-search function delegates per-rate bench execution to
# ``run_driver``. To test the search LOGIC in isolation from the
# subprocess layer, we inject a stub ``run_driver`` via monkey-patch
# that returns preset DriverResults keyed on the request_rate kwarg.
# This is test-only isolation of the algorithm under test (the search
# loop), not faking of the function under test.
# ---------------------------------------------------------------------------


def _make_result(*, e2el_p99: float, throughput: float) -> DriverResult:
    """Helper to build a DriverResult with just the fields the
    rate-search algorithm reads."""
    return DriverResult(
        tokens_per_sec=throughput * 20,
        request_throughput=throughput,
        ttft_ms={},
        tpot_ms={},
        e2el_ms={"p99": e2el_p99},
        goodput_req_per_sec=throughput,
        raw={},
    )


def test_rate_search_returns_inf_when_initial_meets_slo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """If the initial --request-rate inf measurement already meets the
    SLO, return immediately with chosen_request_rate=inf."""
    from autoinfer.harness import driver as drv

    calls: list[float | None] = []

    def stub(*, request_rate: float | None, **kw: object) -> DriverResult:
        calls.append(request_rate)
        return _make_result(e2el_p99=400.0, throughput=47.0)

    monkeypatch.setattr(drv, "run_driver", stub)
    out = drv.run_driver_with_rate_search(
        endpoint="http://x",
        trace_path=tmp_path / "t",
        model="m",
        result_dir=tmp_path,
        result_name_prefix="c",
        max_e2e_slo_ms=500.0,
    )
    assert out.chosen_request_rate == float("inf")
    assert calls == [None]  # only the initial inf-rate measurement


def test_rate_search_iterates_down_until_slo_met(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Initial inf-rate gives throughput=47 and e2el_p99=20000 (over
    SLO). Loop starts at rate=48 and decrements; assume rates 48..24
    are over SLO and rate=23 meets it (P99=494). Return at rate=23."""
    from autoinfer.harness import driver as drv

    calls: list[float | None] = []

    def stub(*, request_rate: float | None, **kw: object) -> DriverResult:
        calls.append(request_rate)
        if request_rate is None:
            # Initial inf-rate — over SLO, observed throughput 47.
            return _make_result(e2el_p99=20000.0, throughput=47.0)
        if request_rate <= 23.0:
            # Rate 23 finds SLO compliance.
            return _make_result(e2el_p99=494.0, throughput=21.39)
        # Rates above 23 still over SLO.
        return _make_result(e2el_p99=20000.0, throughput=21.39)

    monkeypatch.setattr(drv, "run_driver", stub)
    out = drv.run_driver_with_rate_search(
        endpoint="http://x",
        trace_path=tmp_path / "t",
        model="m",
        result_dir=tmp_path,
        result_name_prefix="c",
        max_e2e_slo_ms=500.0,
    )
    assert out.chosen_request_rate == 23.0
    # Iteration: initial inf, then 48, 47, 46, ..., 24, 23 — total 27 calls.
    assert calls[0] is None
    assert calls[1] == 48.0
    assert calls[-1] == 23.0
    # Result reflects the rate=23 measurement.
    assert out.e2el_ms["p99"] == 494.0
    assert out.goodput_req_per_sec == 21.39


def test_rate_search_bails_at_min_rate_when_no_rate_meets_slo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Workload too slow for even rate=1 to meet SLO. Loop bails at
    min_request_rate=1; returns the last measurement with
    chosen_request_rate=1.0 and (typically) goodput=0."""
    from autoinfer.harness import driver as drv

    calls: list[float | None] = []

    def stub(*, request_rate: float | None, **kw: object) -> DriverResult:
        calls.append(request_rate)
        # No rate meets SLO; pretend throughput is 5 req/s so loop
        # starts at rate=6.
        if request_rate is None:
            return _make_result(e2el_p99=60000.0, throughput=5.0)
        return _make_result(e2el_p99=60000.0, throughput=5.0)

    monkeypatch.setattr(drv, "run_driver", stub)
    out = drv.run_driver_with_rate_search(
        endpoint="http://x",
        trace_path=tmp_path / "t",
        model="m",
        result_dir=tmp_path,
        result_name_prefix="c",
        max_e2e_slo_ms=500.0,
    )
    assert out.chosen_request_rate == 1.0
    # Iteration: initial inf, then 6, 5, 4, 3, 2, 1 — total 7 calls.
    assert calls == [None, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]


def test_rate_search_passes_seed_and_slo_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """seed + goodput_slo_ms must flow into every per-rate run_driver call."""
    from autoinfer.harness import driver as drv

    seen_seeds: list[int | None] = []
    seen_slos: list[dict[str, float] | None] = []

    def stub(
        *, request_rate: float | None,
        seed: int | None = None,
        goodput_slo_ms: dict[str, float] | None = None,
        **kw: object,
    ) -> DriverResult:
        seen_seeds.append(seed)
        seen_slos.append(goodput_slo_ms)
        return _make_result(e2el_p99=400.0, throughput=20.0)

    monkeypatch.setattr(drv, "run_driver", stub)
    drv.run_driver_with_rate_search(
        endpoint="http://x",
        trace_path=tmp_path / "t",
        model="m",
        result_dir=tmp_path,
        result_name_prefix="c",
        max_e2e_slo_ms=500.0,
        seed=42,
        goodput_slo_ms={"TTFT": 500.0, "TPOT": 50.0, "E2E": 500.0},
    )
    # initial inf-rate already met SLO so only one call
    assert seen_seeds == [42]
    assert seen_slos[0] == {"TTFT": 500.0, "TPOT": 50.0, "E2E": 500.0}
