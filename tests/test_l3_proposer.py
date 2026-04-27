"""Tests for the L3 kernel proposer.

Exercises the pure helpers (build_kernel_prompt, parse_kernel_blocks)
and the KernelProposer class with a stub LLM. No network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from autoinfer.layers.l3_kernel.proposer import (
    KernelProposer,
    PairedControlProposer,
    build_kernel_prompt,
    build_paired_kernel_prompt,
    build_single_cell_kernel_prompt,
    parse_kernel_blocks,
)


def _surface() -> dict[str, dict[str, Any]]:
    return {
        "target_op": {"type": "categorical", "values": ["rmsnorm", "silu_mul", "rope"]},
        "dtype": {"type": "categorical", "values": ["float32", "float16", "bfloat16"]},
        "shape_regime": {"type": "categorical", "values": ["small", "medium", "large"]},
    }


@dataclass
class _StubLLM:
    response: str
    calls: list[str] = field(default_factory=list)

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.response


def test_build_kernel_prompt_includes_surface_and_references() -> None:
    prompt = build_kernel_prompt(
        surface=_surface(), n=3, prior_notes="GPU=A100", history=[]
    )
    assert "rmsnorm" in prompt and "silu_mul" in prompt and "rope" in prompt
    # references included so LLM has a baseline to beat
    assert "rmsnorm_kernel" in prompt
    assert "silu_mul_kernel" in prompt
    assert "rope_kernel" in prompt
    assert "GPU=A100" in prompt
    assert "Propose exactly 3" in prompt


def test_parse_single_block() -> None:
    text = """\
TARGET_OP: rmsnorm
DTYPE: float16
SHAPE_REGIME: small
ENTRY_FN: my_kernel
SOURCE:
@@@
def my_kernel(x, w, eps):
    return x * w
@@@
"""
    blocks = parse_kernel_blocks(text)
    assert len(blocks) == 1
    b = blocks[0]
    assert b["target_op"] == "rmsnorm"
    assert b["dtype"] == "float16"
    assert b["shape_regime"] == "small"
    assert b["entry_fn"] == "my_kernel"
    assert "def my_kernel" in b["source"]


def test_parse_two_blocks_separated_by_blank_lines() -> None:
    text = """\
TARGET_OP: rmsnorm
DTYPE: float32
SHAPE_REGIME: small
ENTRY_FN: kernel_a
SOURCE:
@@@
def kernel_a(x, w, eps):
    return x
@@@

TARGET_OP: silu_mul
DTYPE: bfloat16
SHAPE_REGIME: medium
ENTRY_FN: kernel_b
SOURCE:
@@@
def kernel_b(a):
    return a
@@@
"""
    blocks = parse_kernel_blocks(text)
    assert len(blocks) == 2
    assert blocks[0]["target_op"] == "rmsnorm"
    assert blocks[1]["target_op"] == "silu_mul"
    assert "def kernel_a" in blocks[0]["source"]
    assert "def kernel_b" in blocks[1]["source"]


def test_parse_skips_block_missing_required_field() -> None:
    text = """\
TARGET_OP: rmsnorm
DTYPE: float32
ENTRY_FN: incomplete
SOURCE:
@@@
def incomplete(x): return x
@@@
"""
    blocks = parse_kernel_blocks(text)
    assert blocks == []  # missing SHAPE_REGIME


def test_parse_skips_block_with_unclosed_delimiter() -> None:
    text = """\
TARGET_OP: rmsnorm
DTYPE: float32
SHAPE_REGIME: small
ENTRY_FN: unclosed
SOURCE:
@@@
def unclosed(x): return x
"""
    assert parse_kernel_blocks(text) == []


def test_parse_handles_prose_around_blocks() -> None:
    text = """\
Sure, here are 1 kernel proposals:

TARGET_OP: rope
DTYPE: bfloat16
SHAPE_REGIME: large
ENTRY_FN: rope_v2
SOURCE:
@@@
def rope_v2(q, cos, sin):
    return q
@@@

Hope this helps!
"""
    blocks = parse_kernel_blocks(text)
    assert len(blocks) == 1
    assert blocks[0]["entry_fn"] == "rope_v2"


def test_proposer_returns_blocks() -> None:
    response = """\
TARGET_OP: rmsnorm
DTYPE: float16
SHAPE_REGIME: small
ENTRY_FN: kernel_v1
SOURCE:
@@@
def kernel_v1(x, w, eps):
    return x
@@@
"""
    prop = KernelProposer(llm=_StubLLM(response=response))
    blocks = prop.propose_configs(
        surface=_surface(), n=1, prior_notes="", history=[],
    )
    assert len(blocks) == 1
    assert blocks[0]["target_op"] == "rmsnorm"


def test_proposer_caps_at_n() -> None:
    response = """\
TARGET_OP: rmsnorm
DTYPE: float32
SHAPE_REGIME: small
ENTRY_FN: a
SOURCE:
@@@
def a(x, w, eps): return x
@@@

TARGET_OP: silu_mul
DTYPE: bfloat16
SHAPE_REGIME: small
ENTRY_FN: b
SOURCE:
@@@
def b(a): return a
@@@

TARGET_OP: rope
DTYPE: float16
SHAPE_REGIME: small
ENTRY_FN: c
SOURCE:
@@@
def c(q, cos, sin): return q
@@@
"""
    prop = KernelProposer(llm=_StubLLM(response=response))
    blocks = prop.propose_configs(
        surface=_surface(), n=2, prior_notes="", history=[],
    )
    assert len(blocks) == 2


def test_proposer_falls_back_to_reference_seeds_on_empty() -> None:
    """Resilience: if LLM returns garbage, use known-good reference kernels."""
    prop = KernelProposer(llm=_StubLLM(response="sorry, I don't know"))
    blocks = prop.propose_configs(
        surface=_surface(), n=2, prior_notes="", history=[],
    )
    assert len(blocks) == 2
    # all reference seeds carry source + entry_fn
    for b in blocks:
        assert "source" in b and "entry_fn" in b


def test_proposer_raises_when_fallback_disabled() -> None:
    prop = KernelProposer(
        llm=_StubLLM(response="garbage"), fallback_when_empty=False,
    )
    with pytest.raises(ValueError):
        prop.propose_configs(
            surface=_surface(), n=1, prior_notes="", history=[],
        )


def test_proposer_n_must_be_positive() -> None:
    prop = KernelProposer(llm=_StubLLM(response="ignored"))
    with pytest.raises(ValueError):
        prop.propose_configs(
            surface=_surface(), n=0, prior_notes="", history=[],
        )


def test_proposer_passes_history_into_prompt() -> None:
    stub = _StubLLM(response="garbage")
    prop = KernelProposer(llm=stub)
    history = [{"trial_id": "t1", "metrics": {"tokens_per_sec": 100.0}}]
    prop.propose_configs(
        surface=_surface(), n=1, prior_notes="GPU=H100", history=history,
    )
    prompt = stub.calls[0]
    assert "GPU=H100" in prompt
    assert "t1" in prompt


# T-27: paired-control proposer ----------------------------------------------


def _ref_seed(op: str, dtype: str, regime: str) -> dict[str, Any]:
    return {
        "target_op": op,
        "dtype": dtype,
        "shape_regime": regime,
        "source": f"def ref_{op}(*args, **kwargs): return args[0]\n",
        "entry_fn": f"ref_{op}",
    }


def _llm_block(op: str, dtype: str, regime: str, entry: str) -> str:
    return (
        f"TARGET_OP: {op}\n"
        f"DTYPE: {dtype}\n"
        f"SHAPE_REGIME: {regime}\n"
        f"ENTRY_FN: {entry}\n"
        f"SOURCE:\n@@@\n"
        f"def {entry}(*args, **kwargs):\n    return args[0]\n"
        f"@@@\n"
    )


def test_build_single_cell_kernel_prompt_includes_only_target_cell() -> None:
    """T-29: single-cell prompt only includes the requested cell's
    reference, not all 3, to keep the context tight and the LLM focused."""
    cell = {"target_op": "rmsnorm", "dtype": "bfloat16", "shape_regime": "medium"}
    prompt = build_single_cell_kernel_prompt(cell=cell, prior_notes="GPU=H100", history=[])
    assert "rmsnorm" in prompt
    assert "bfloat16" in prompt
    assert "medium" in prompt
    assert "GPU=H100" in prompt
    # Reference for the target op IS included as the baseline-to-beat
    assert "rmsnorm_kernel" in prompt
    # Other ops' references should NOT be included (avoid context bloat)
    assert "silu_mul_kernel" not in prompt
    assert "rope_kernel" not in prompt
    # Single-block instruction
    assert "ONE kernel" in prompt
    assert "TARGET_OP    = rmsnorm" in prompt


def test_build_single_cell_kernel_prompt_rejects_unknown_op() -> None:
    cell = {"target_op": "attention", "dtype": "bfloat16", "shape_regime": "medium"}
    with pytest.raises(ValueError):
        build_single_cell_kernel_prompt(cell=cell, prior_notes="", history=[])


def test_build_paired_kernel_prompt_lists_cells_in_order() -> None:
    cells = [
        {"target_op": "rmsnorm", "dtype": "bfloat16", "shape_regime": "medium"},
        {"target_op": "silu_mul", "dtype": "float16", "shape_regime": "small"},
    ]
    prompt = build_paired_kernel_prompt(
        surface={}, cells=cells, prior_notes="", history=[],
    )
    # Each cell must appear in the prompt body.
    assert "rmsnorm" in prompt and "bfloat16" in prompt and "medium" in prompt
    assert "silu_mul" in prompt and "float16" in prompt and "small" in prompt
    assert "PAIRED-CONTROL" in prompt
    # The prompt orders cells: rmsnorm appears before silu_mul.
    assert prompt.find("rmsnorm") < prompt.find("silu_mul")


@dataclass
class _CellAwareStubLLM:
    """T-29 stub: returns a per-cell response based on the prompt's
    target cell. Matches the post-T-29 sequential-call protocol where
    each LLM round-trip names exactly one cell.
    """
    by_op: dict[str, str] = field(default_factory=dict)
    """{target_op: response_text}"""
    calls: list[str] = field(default_factory=list)

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        for op in ("rmsnorm", "silu_mul", "rope"):
            # Single-cell prompt's cell line: "  TARGET_OP    = rmsnorm"
            if f"TARGET_OP    = {op}" in prompt:
                return self.by_op.get(op, "")
        return ""


def test_propose_for_cells_returns_one_per_cell_in_order() -> None:
    """T-29: each cell gets its own LLM call; results are returned in
    input order regardless of how the LLM orders its outputs."""
    cells = [
        {"target_op": "rmsnorm", "dtype": "bfloat16", "shape_regime": "medium"},
        {"target_op": "silu_mul", "dtype": "float16", "shape_regime": "small"},
    ]
    stub = _CellAwareStubLLM(by_op={
        "rmsnorm": _llm_block("rmsnorm", "bfloat16", "medium", "novel_rms"),
        "silu_mul": _llm_block("silu_mul", "float16", "small", "novel_silu"),
    })
    prop = KernelProposer(llm=stub)
    out = prop.propose_for_cells(cells=cells, prior_notes="", history=[])
    assert len(out) == 2
    assert out[0]["target_op"] == "rmsnorm"
    assert out[0]["dtype"] == "bfloat16"
    assert out[0]["shape_regime"] == "medium"
    assert out[0]["entry_fn"] == "novel_rms"
    assert out[1]["target_op"] == "silu_mul"
    assert out[1]["entry_fn"] == "novel_silu"
    # T-29 invariant: one LLM call per cell, not one giant batched call.
    assert len(stub.calls) == 2


def test_propose_for_cells_isolates_per_cell_failures() -> None:
    """T-29 core motivation: a bad LLM response for cell K must NOT
    propagate into cell K+1. Campaign 02's 6-block paired prompt broke
    silu_mul cells when the LLM degraded after the rmsnorm blocks; per-
    cell calls eliminate that coupling.
    """
    cells = [
        {"target_op": "rmsnorm", "dtype": "bfloat16", "shape_regime": "medium"},
        {"target_op": "silu_mul", "dtype": "float16", "shape_regime": "small"},
    ]
    # rmsnorm returns garbage; silu_mul returns a valid block. Pre-T-29
    # this would have left silu_mul broken too (the blocks shared a
    # response). Post-T-29, each cell's response is independent.
    stub = _CellAwareStubLLM(by_op={
        "rmsnorm": "sorry I don't know",
        "silu_mul": _llm_block("silu_mul", "float16", "small", "novel_silu"),
    })
    prop = KernelProposer(llm=stub)
    out = prop.propose_for_cells(cells=cells, prior_notes="", history=[])
    assert len(out) == 2
    # rmsnorm cell falls back to reference (no log marker assertion;
    # the fallback path is exercised — see other tests for that).
    assert out[0]["target_op"] == "rmsnorm"
    assert out[0]["dtype"] == "bfloat16"
    assert out[0].get("source") and out[0].get("entry_fn")
    # silu_mul cell got the LLM-novel block — independent of rmsnorm's failure.
    assert out[1]["entry_fn"] == "novel_silu"
    assert len(stub.calls) == 2


def test_propose_for_cells_overrides_llm_cell_drift() -> None:
    """If the LLM returns the right op but wrong dtype/regime, we still
    emit a config pinned to the requested cell, reusing the LLM's source.
    Without this guarantee, paired-control would silently lose its
    same-cell A/B property."""
    cells = [{"target_op": "rmsnorm", "dtype": "bfloat16", "shape_regime": "medium"}]
    drift = _llm_block("rmsnorm", "float32", "small", "novel_drift")
    prop = KernelProposer(llm=_StubLLM(response=drift))
    out = prop.propose_for_cells(cells=cells, prior_notes="", history=[])
    assert len(out) == 1
    assert out[0]["target_op"] == "rmsnorm"
    assert out[0]["dtype"] == "bfloat16"
    assert out[0]["shape_regime"] == "medium"
    assert out[0]["entry_fn"] == "novel_drift"
    assert "novel_drift" in out[0]["source"]


def test_propose_for_cells_falls_back_to_reference_when_op_missing() -> None:
    """If the LLM returns no block for an op, that cell falls back to
    the reference kernel for that op so the pair still runs (with the
    reference-twice degeneracy logged)."""
    cells = [{"target_op": "rmsnorm", "dtype": "bfloat16", "shape_regime": "medium"}]
    prop = KernelProposer(llm=_StubLLM(response="garbage no blocks"))
    out = prop.propose_for_cells(cells=cells, prior_notes="", history=[])
    assert len(out) == 1
    assert out[0]["target_op"] == "rmsnorm"
    assert out[0]["dtype"] == "bfloat16"
    assert out[0]["shape_regime"] == "medium"
    # Source must be present (reference fallback)
    assert out[0].get("source")
    assert out[0].get("entry_fn")


def test_paired_control_alternates_ref_then_novel() -> None:
    seeds = [
        _ref_seed("rmsnorm", "bfloat16", "medium"),
        _ref_seed("silu_mul", "float16", "small"),
    ]
    stub = _CellAwareStubLLM(by_op={
        "rmsnorm": _llm_block("rmsnorm", "bfloat16", "medium", "novel_rms"),
        "silu_mul": _llm_block("silu_mul", "float16", "small", "novel_silu"),
    })
    base = KernelProposer(llm=stub)
    pc = PairedControlProposer(base=base, reference_seeds=seeds)
    out = pc.propose_configs(surface={}, n=4, prior_notes="", history=[])
    assert len(out) == 4
    # Pair 0: ref @ rmsnorm/bf16/medium, then novel at the same cell
    assert out[0]["entry_fn"] == "ref_rmsnorm"
    assert out[1]["entry_fn"] == "novel_rms"
    assert out[0]["target_op"] == out[1]["target_op"] == "rmsnorm"
    assert out[0]["dtype"] == out[1]["dtype"] == "bfloat16"
    assert out[0]["shape_regime"] == out[1]["shape_regime"] == "medium"
    # Pair 1: ref @ silu_mul/fp16/small, then novel at the same cell
    assert out[2]["entry_fn"] == "ref_silu_mul"
    assert out[3]["entry_fn"] == "novel_silu"
    assert out[2]["target_op"] == out[3]["target_op"] == "silu_mul"


def test_paired_control_pairs_match_cells_under_drift() -> None:
    """Even if the LLM returns drifted dtype/regime, the paired pair
    still matches at the (op, dtype, regime) level — that's the whole
    point of T-27."""
    seeds = [_ref_seed("rmsnorm", "bfloat16", "medium")]
    response = _llm_block("rmsnorm", "float32", "small", "drift_rms")
    base = KernelProposer(llm=_StubLLM(response=response))
    pc = PairedControlProposer(base=base, reference_seeds=seeds)
    out = pc.propose_configs(surface={}, n=2, prior_notes="", history=[])
    assert (out[0]["target_op"], out[0]["dtype"], out[0]["shape_regime"]) == (
        "rmsnorm",
        "bfloat16",
        "medium",
    )
    assert (out[1]["target_op"], out[1]["dtype"], out[1]["shape_regime"]) == (
        "rmsnorm",
        "bfloat16",
        "medium",
    )
    assert out[1]["entry_fn"] == "drift_rms"  # LLM source is preserved


def test_paired_control_truncates_when_n_is_odd() -> None:
    """n=3 → ref0 / novel0 / ref1 (no novel for pair 1)."""
    seeds = [
        _ref_seed("rmsnorm", "bfloat16", "medium"),
        _ref_seed("silu_mul", "float16", "small"),
    ]
    response = (
        _llm_block("rmsnorm", "bfloat16", "medium", "novel_rms")
        + "\n"
        + _llm_block("silu_mul", "float16", "small", "novel_silu")
    )
    base = KernelProposer(llm=_StubLLM(response=response))
    pc = PairedControlProposer(base=base, reference_seeds=seeds)
    out = pc.propose_configs(surface={}, n=3, prior_notes="", history=[])
    assert len(out) == 3
    assert out[0]["entry_fn"] == "ref_rmsnorm"
    assert out[1]["entry_fn"] == "novel_rms"
    assert out[2]["entry_fn"] == "ref_silu_mul"


def test_paired_control_cycles_seeds_when_n_exceeds_2x_seeds() -> None:
    seeds = [_ref_seed("rmsnorm", "bfloat16", "medium")]  # only 1 cell
    response = _llm_block("rmsnorm", "bfloat16", "medium", "novel_rms")
    base = KernelProposer(llm=_StubLLM(response=response))
    pc = PairedControlProposer(base=base, reference_seeds=seeds)
    out = pc.propose_configs(surface={}, n=4, prior_notes="", history=[])
    assert len(out) == 4
    # Pairs alternate, both at the same cell (cycled)
    for i, cfg in enumerate(out):
        assert cfg["target_op"] == "rmsnorm"
        assert cfg["dtype"] == "bfloat16"
        assert cfg["shape_regime"] == "medium"
        assert cfg["entry_fn"] == ("ref_rmsnorm" if i % 2 == 0 else "novel_rms")


def test_paired_control_rejects_empty_seeds() -> None:
    base = KernelProposer(llm=_StubLLM(response=""))
    with pytest.raises(ValueError):
        PairedControlProposer(base=base, reference_seeds=[])


def test_paired_control_rejects_seed_missing_cell_keys() -> None:
    base = KernelProposer(llm=_StubLLM(response=""))
    bad = [{"target_op": "rmsnorm", "source": "def f(): pass", "entry_fn": "f"}]
    with pytest.raises(ValueError):
        PairedControlProposer(base=base, reference_seeds=bad)