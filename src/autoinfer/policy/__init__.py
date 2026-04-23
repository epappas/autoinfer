"""Hybrid LLM + classical-surrogate policy stack (P7)."""

from autoinfer.policy.fidelity import Rung, promote, successive_halving_rungs
from autoinfer.policy.llm_providers import (
    AnthropicProposalLLM,
    OpenAICompatibleProposalLLM,
    build_prompt,
    parse_configs,
)
from autoinfer.policy.operator import Operator
from autoinfer.policy.surrogate import OptunaSurrogate, Suggestion, Surrogate
from autoinfer.policy.warmstart import DeterministicProposalLLM, ProposalLLM

__all__ = [
    "AnthropicProposalLLM",
    "DeterministicProposalLLM",
    "OpenAICompatibleProposalLLM",
    "Operator",
    "OptunaSurrogate",
    "ProposalLLM",
    "Rung",
    "Suggestion",
    "Surrogate",
    "build_prompt",
    "parse_configs",
    "promote",
    "successive_halving_rungs",
]
