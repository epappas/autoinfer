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
    build_kernel_prompt,
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