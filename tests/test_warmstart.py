from __future__ import annotations

import pytest

from autoinfer.policy.operator import Operator
from autoinfer.policy.warmstart import DeterministicProposalLLM, ProposalLLM


def _surface() -> dict[str, dict[str, object]]:
    return {
        "max_num_batched_tokens": {"type": "int", "low": 1024, "high": 16384},
        "kv_cache_dtype": {"type": "categorical", "values": ["auto", "fp8"]},
    }


def _configs() -> list[dict[str, object]]:
    return [
        {"max_num_batched_tokens": 2048, "kv_cache_dtype": "auto"},
        {"max_num_batched_tokens": 8192, "kv_cache_dtype": "fp8"},
    ]


def test_deterministic_conforms_to_protocol() -> None:
    llm = DeterministicProposalLLM(_configs())
    assert isinstance(llm, ProposalLLM)


def test_empty_configs_rejected() -> None:
    with pytest.raises(ValueError):
        DeterministicProposalLLM([])


def test_returns_n_configs_in_order() -> None:
    llm = DeterministicProposalLLM(_configs())
    out = llm.propose_configs(_surface(), n=2, prior_notes="", history=[])
    assert out == _configs()


def test_cycles_when_n_exceeds_list() -> None:
    llm = DeterministicProposalLLM(_configs())
    out = llm.propose_configs(_surface(), n=5, prior_notes="", history=[])
    assert len(out) == 5
    assert out[0] == _configs()[0]
    assert out[2] == _configs()[0]  # wraps
    assert out[4] == _configs()[0]


def test_idx_persists_across_calls() -> None:
    llm = DeterministicProposalLLM(_configs())
    first = llm.propose_configs(_surface(), n=1, prior_notes="", history=[])
    second = llm.propose_configs(_surface(), n=1, prior_notes="", history=[])
    assert first[0] == _configs()[0]
    assert second[0] == _configs()[1]


def test_rejects_config_with_unknown_key() -> None:
    bad = [{"made_up_knob": 123}]
    llm = DeterministicProposalLLM(bad)
    with pytest.raises(ValueError):
        llm.propose_configs(_surface(), n=1, prior_notes="", history=[])


def test_n_must_be_positive() -> None:
    llm = DeterministicProposalLLM(_configs())
    with pytest.raises(ValueError):
        llm.propose_configs(_surface(), n=0, prior_notes="", history=[])


def test_operator_proposes_on_stall() -> None:
    llm = DeterministicProposalLLM(_configs())
    op = Operator(llm=llm, cadence=10)
    assert op.should_propose(trials_since_last=0, stalled=True) is True


def test_operator_proposes_at_cadence() -> None:
    llm = DeterministicProposalLLM(_configs())
    op = Operator(llm=llm, cadence=3)
    assert op.should_propose(trials_since_last=2, stalled=False) is False
    assert op.should_propose(trials_since_last=3, stalled=False) is True


def test_operator_cadence_must_be_at_least_one() -> None:
    llm = DeterministicProposalLLM(_configs())
    with pytest.raises(ValueError):
        Operator(llm=llm, cadence=0)


def test_operator_propose_delegates() -> None:
    llm = DeterministicProposalLLM(_configs())
    op = Operator(llm=llm, cadence=10)
    out = op.propose(_surface(), n=1, prior_notes="", history=[])
    assert out == [_configs()[0]]
