from __future__ import annotations

import pytest
import torch

from autoinfer.layers.l3_kernel.baselines import rmsnorm_ref, rope_ref, silu_mul_ref


def test_rmsnorm_identity_on_unit_weight_zero_input() -> None:
    x = torch.zeros(2, 8)
    w = torch.ones(8)
    out = rmsnorm_ref(x, w, 1e-6)
    assert out.shape == x.shape
    assert torch.allclose(out, torch.zeros_like(out))


def test_rmsnorm_normalizes_magnitude() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 16)
    w = torch.ones(16)
    out = rmsnorm_ref(x, w, 1e-6)
    # the rms of each row should be ~1
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_rmsnorm_preserves_dtype() -> None:
    x = torch.randn(2, 8, dtype=torch.float32).to(torch.float16)
    w = torch.ones(8, dtype=torch.float16)
    out = rmsnorm_ref(x, w, 1e-6)
    assert out.dtype == torch.float16


def test_rmsnorm_rejects_bad_weight_shape() -> None:
    with pytest.raises(ValueError):
        rmsnorm_ref(torch.zeros(2, 8), torch.zeros(4), 1e-6)


def test_silu_mul_matches_manual() -> None:
    torch.manual_seed(1)
    a = torch.randn(3, 8)
    x, y = a.chunk(2, dim=-1)
    expected = torch.nn.functional.silu(x) * y
    out = silu_mul_ref(a)
    assert torch.allclose(out, expected)


def test_silu_mul_rejects_odd_last_dim() -> None:
    with pytest.raises(ValueError):
        silu_mul_ref(torch.zeros(2, 7))


def test_rope_identity_when_cos_one_sin_zero() -> None:
    q = torch.randn(1, 2, 4, 8)
    cos = torch.ones(4, 8)
    sin = torch.zeros(4, 8)
    out = rope_ref(q, cos, sin)
    assert torch.allclose(out, q)


def test_rope_rotates_pairs_by_ninety_degrees() -> None:
    # cos=0, sin=1 swaps (q_even, q_odd) -> (-q_odd, q_even)
    q = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    cos = torch.zeros(1, 4)
    sin = torch.ones(1, 4)
    out = rope_ref(q, cos, sin)
    expected = torch.tensor([[[[-2.0, 1.0, -4.0, 3.0]]]])
    assert torch.allclose(out, expected)


def test_rope_rejects_odd_last_dim() -> None:
    with pytest.raises(ValueError):
        rope_ref(torch.zeros(1, 1, 1, 5), torch.zeros(1, 5), torch.zeros(1, 5))
