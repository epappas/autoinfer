"""LLM kernel proposer for L3.

Unlike the engine-knob proposer (`policy/llm_providers.py`) which returns
a JSON array of knob-value mappings, the kernel proposer must return
**source code** alongside the surrogate-searchable choice of
``(target_op, dtype, shape_regime)``. Source-code-in-JSON is fragile
(newlines, quotes need escaping); a delimited response format is what
the LLM actually produces reliably:

    TARGET_OP: rmsnorm
    DTYPE: float16
    SHAPE_REGIME: medium
    ENTRY_FN: rmsnorm_kernel_v1
    SOURCE:
    @@@
    import torch
    def rmsnorm_kernel_v1(x, w, eps):
        ...
    @@@

Conforms to ``ProposalLLM`` so the joint runner can plug it in
identically to the L1/L2 proposers. Builds on the underlying HTTP
client (`OpenAICompatibleProposalLLM` or `AnthropicProposalLLM`) by
intercepting the prompt + parse stages — no separate HTTP layer.

The proposer never inspects the search surface for knob structure; it
treats the surface as a hint to the LLM about valid op/dtype/regime
choices. Adapter-only keys (``source``, ``entry_fn``) flow through
because the warmstart/LLM validators were relaxed in 06afd69.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from autoinfer.layers.l3_kernel.baselines import REFERENCES
from autoinfer.layers.l3_kernel.surface import REFERENCE_SOURCES


@runtime_checkable
class _RawLLM(Protocol):
    """Subset of httpx-style chat-completion interface we need."""

    def complete(self, prompt: str) -> str: ...


_BLOCK_DELIM = "@@@"
_FIELD_RE = re.compile(r"^\s*([A-Z_]+)\s*:\s*(.*)$")


def build_kernel_prompt(
    surface: dict[str, dict[str, Any]],
    n: int,
    prior_notes: str,
    history: list[dict[str, Any]],
) -> str:
    """Compose the kernel-proposer prompt. Pure."""
    surface_json = json.dumps(surface, indent=2, default=str, sort_keys=True)
    tail = history[-15:] if history else []
    history_json = json.dumps(tail, indent=2, default=str, sort_keys=True)
    notes = prior_notes.strip() or "(none)"
    ops = ", ".join(sorted(REFERENCES.keys()))
    refs_section = "\n\n".join(
        f"### Reference {op} (entry={entry}):\n```python\n{src}\n```"
        for op, (entry, src) in REFERENCE_SOURCES.items()
    )
    return (
        "You are proposing kernel implementations for L3 of an "
        "inference-engine search. The candidate replaces a transformer "
        "op (currently one of: " + ops + "). Your kernel must produce "
        "output within atol=1e-3, rtol=1e-3 of the PyTorch reference on "
        "a small test matrix; only kernels that pass correctness are "
        "timed for throughput. The judging environment provides ``torch`` "
        "and ``math`` in the namespace; ``triton`` and ``triton.language`` "
        "are also importable when the runtime has a GPU. CPU-only runs "
        "use plain PyTorch source.\n\n"
        f"Surrogate surface:\n```json\n{surface_json}\n```\n\n"
        f"Hardware and runtime notes:\n{notes}\n\n"
        f"Recent trial history (last {len(tail)}):\n"
        f"```json\n{history_json}\n```\n\n"
        f"Reference implementations (your baseline to beat or match):\n\n"
        f"{refs_section}\n\n"
        f"Propose exactly {n} kernel candidate(s). For each, return a "
        "block of the following exact form (the @@@ delimiters bracket "
        "the source so newlines and quotes are unambiguous):\n\n"
        "TARGET_OP: <one of the surface target_op values>\n"
        "DTYPE: <one of the surface dtype values>\n"
        "SHAPE_REGIME: <one of the surface shape_regime values>\n"
        "ENTRY_FN: <name of the callable in your source>\n"
        "SOURCE:\n"
        f"{_BLOCK_DELIM}\n"
        "import torch\n\n"
        "def my_kernel(...):\n"
        "    ...\n"
        f"{_BLOCK_DELIM}\n\n"
        "Separate multiple candidate blocks with a blank line. Return "
        f"NOTHING but the {n} block(s) — no preamble, no commentary."
    )


def parse_kernel_blocks(text: str) -> list[dict[str, Any]]:
    """Extract candidate dicts from the LLM's delimited response.

    Robust to surrounding prose; locates each ``SOURCE:`` line and the
    next pair of ``@@@`` delimiters. Header fields above the SOURCE
    line are scanned for ``TARGET_OP``, ``DTYPE``, ``SHAPE_REGIME``,
    and ``ENTRY_FN``; missing required fields skip the block.
    """
    out: list[dict[str, Any]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip().upper().startswith("SOURCE"):
            header_lines = []
            j = i - 1
            # walk back collecting header KEY: VALUE pairs (until blank)
            while j >= 0 and lines[j].strip():
                header_lines.append(lines[j])
                j -= 1
            header_lines.reverse()
            fields: dict[str, str] = {}
            for hl in header_lines:
                m = _FIELD_RE.match(hl)
                if m:
                    fields[m.group(1).upper()] = m.group(2).strip()
            # find next @@@ ... @@@ pair
            k = i + 1
            while k < len(lines) and lines[k].strip() != _BLOCK_DELIM:
                k += 1
            if k >= len(lines):
                i += 1
                continue
            start = k + 1
            end = start
            while end < len(lines) and lines[end].strip() != _BLOCK_DELIM:
                end += 1
            if end >= len(lines):
                i += 1
                continue
            source = "\n".join(lines[start:end])
            block = {
                "target_op": fields.get("TARGET_OP", "").lower() or None,
                "dtype": fields.get("DTYPE", "") or None,
                "shape_regime": fields.get("SHAPE_REGIME", "") or None,
                "entry_fn": fields.get("ENTRY_FN", "") or None,
                "source": source,
            }
            if all(block[k] for k in ("target_op", "dtype", "shape_regime", "entry_fn", "source")):
                out.append(block)
            i = end + 1
        else:
            i += 1
    return out


@dataclass
class KernelProposer:
    """Generate L3 kernel candidates via an LLM source-code prompt.

    Wraps a raw chat-completion provider so the underlying HTTP layer
    is shared with `OpenAICompatibleProposalLLM` (you pass in a
    callable that takes a prompt and returns a response string).
    """

    llm: _RawLLM
    fallback_when_empty: bool = True
    """When the LLM returns no parseable blocks, fall back to the
    reference seed configs instead of raising. Keeps the warmstart
    path resilient against transient LLM failures."""

    def propose_configs(
        self,
        surface: dict[str, dict[str, Any]],
        n: int,
        prior_notes: str,
        history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if n <= 0:
            raise ValueError("n must be positive")
        prompt = build_kernel_prompt(surface, n, prior_notes, history)
        text = self.llm.complete(prompt)
        blocks = parse_kernel_blocks(text)
        if not blocks and self.fallback_when_empty:
            from autoinfer.layers.l3_kernel.surface import reference_seed_configs

            return reference_seed_configs()[:n]
        if not blocks:
            raise ValueError(
                f"LLM returned no parseable kernel blocks; first 300 chars: {text[:300]!r}"
            )
        return blocks[:n]


__all__ = [
    "KernelProposer",
    "build_kernel_prompt",
    "parse_kernel_blocks",
]
