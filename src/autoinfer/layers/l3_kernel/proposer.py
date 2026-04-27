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


def _cell_key(cfg: dict[str, Any]) -> tuple[str, str, str]:
    """Canonical cell tuple for a config — used to pair refs with novels."""
    return (
        str(cfg.get("target_op", "")),
        str(cfg.get("dtype", "")),
        str(cfg.get("shape_regime", "")),
    )


def build_paired_kernel_prompt(
    surface: dict[str, dict[str, Any]],
    cells: list[dict[str, Any]],
    prior_notes: str,
    history: list[dict[str, Any]],
) -> str:
    """T-27: prompt the LLM for one kernel per pinned cell.

    The proposer asks for ``len(cells)`` candidates, each constrained to
    a specific ``(target_op, dtype, shape_regime)`` cell. The reference
    kernel for each cell is included so the LLM has the exact target it
    must beat or match at correctness — and so the campaign can do a
    same-cell A/B against that reference.
    """
    surface_json = json.dumps(surface, indent=2, default=str, sort_keys=True)
    tail = history[-15:] if history else []
    history_json = json.dumps(tail, indent=2, default=str, sort_keys=True)
    notes = prior_notes.strip() or "(none)"
    cells_json = json.dumps(
        [
            {
                "target_op": c["target_op"],
                "dtype": c["dtype"],
                "shape_regime": c["shape_regime"],
            }
            for c in cells
        ],
        indent=2,
    )
    refs_section = "\n\n".join(
        f"### Reference {op} (entry={entry}):\n```python\n{src}\n```"
        for op, (entry, src) in REFERENCE_SOURCES.items()
    )
    return (
        "You are proposing kernel implementations for L3 of an "
        "inference-engine search, in PAIRED-CONTROL mode: every "
        "candidate you propose is run back-to-back against the "
        "reference kernel at the SAME (target_op, dtype, shape_regime) "
        "cell so we can measure your novelty against a same-cell "
        "control.\n\n"
        "Your kernel must produce output within atol=1e-3, rtol=1e-3 "
        "of the PyTorch reference; only kernels that pass correctness "
        "are timed. The judging environment provides ``torch`` and "
        "``math``; ``triton`` and ``triton.language`` are importable "
        "when a GPU is present.\n\n"
        f"Surrogate surface:\n```json\n{surface_json}\n```\n\n"
        f"Hardware and runtime notes:\n{notes}\n\n"
        f"Recent trial history (last {len(tail)}):\n"
        f"```json\n{history_json}\n```\n\n"
        f"Reference implementations:\n\n{refs_section}\n\n"
        f"Propose EXACTLY one candidate per cell, in the order given:\n"
        f"```json\n{cells_json}\n```\n\n"
        "Each block MUST set TARGET_OP / DTYPE / SHAPE_REGIME to the "
        "values of the corresponding cell. Same exact-form delimited "
        "block as before:\n\n"
        "TARGET_OP: <cell.target_op>\n"
        "DTYPE: <cell.dtype>\n"
        "SHAPE_REGIME: <cell.shape_regime>\n"
        "ENTRY_FN: <name of the callable in your source>\n"
        "SOURCE:\n"
        f"{_BLOCK_DELIM}\n"
        "import torch\n\n"
        "def my_kernel(...):\n"
        "    ...\n"
        f"{_BLOCK_DELIM}\n\n"
        "Separate blocks with a blank line. Return NOTHING but the "
        f"{len(cells)} block(s) — no preamble, no commentary."
    )


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
            # Telemetry: the runner / campaign-runner captures stdout
            # so this marker shows up in basilica logs and the per-trial
            # vllm.out file alongside the [autoinfer.l3.injector] line.
            # Without it, "LLM never generated a novel kernel" looks
            # identical to "LLM generated and matched reference" in
            # post-run analysis.
            print(
                f"[autoinfer.l3.proposer] fallback to reference seeds "
                f"(LLM returned no parseable blocks; first 200 chars: "
                f"{text[:200]!r})",
                flush=True,
            )
            from autoinfer.layers.l3_kernel.surface import reference_seed_configs

            return reference_seed_configs()[:n]
        if not blocks:
            raise ValueError(
                f"LLM returned no parseable kernel blocks; first 300 chars: {text[:300]!r}"
            )
        return blocks[:n]

    def propose_for_cells(
        self,
        cells: list[dict[str, Any]],
        prior_notes: str,
        history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """T-27. Ask the LLM for one kernel per pinned cell.

        Returns ``len(cells)`` configs in input order. For each cell, if
        the LLM produced a parseable block whose (target_op, dtype,
        shape_regime) matches the requested cell, that block is used —
        with the cell triple force-set on the returned config so any
        LLM-side drift is normalised. Otherwise we look for any
        parseable block whose target_op matches the cell's target_op
        and reuse that source (re-tagged with the requested
        dtype/shape_regime). If no block at all matches, that cell
        falls back to the reference kernel for its target_op (so the
        pair degenerates to two reference trials — still valid as a
        same-cell reproducibility data point, never silently emits a
        wrong-cell candidate).
        """
        if not cells:
            return []
        from autoinfer.layers.l3_kernel.surface import REFERENCE_SOURCES

        prompt = build_paired_kernel_prompt(
            surface={}, cells=cells, prior_notes=prior_notes, history=history
        )
        text = self.llm.complete(prompt)
        blocks = parse_kernel_blocks(text)

        # Index parsed blocks by exact cell, and (op-only) for fallback
        by_cell: dict[tuple[str, str, str], dict[str, Any]] = {}
        by_op: dict[str, dict[str, Any]] = {}
        for b in blocks:
            ck = _cell_key(b)
            by_cell.setdefault(ck, b)
            op = str(b.get("target_op", ""))
            if op:
                by_op.setdefault(op, b)

        out: list[dict[str, Any]] = []
        for cell in cells:
            ck = _cell_key(cell)
            chosen: dict[str, Any] | None = by_cell.get(ck)
            if chosen is None:
                chosen = by_op.get(str(cell["target_op"]))
            if chosen is None:
                op = str(cell["target_op"])
                ref = REFERENCE_SOURCES.get(op)
                if ref is None:
                    raise ValueError(
                        f"no reference for target_op={op!r}; cannot fall back"
                    )
                entry, src = ref
                print(
                    f"[autoinfer.l3.proposer] paired-control fallback to "
                    f"reference for cell={ck} (LLM produced no usable block)",
                    flush=True,
                )
                chosen = {"entry_fn": entry, "source": src}
            # Force the cell triple onto the returned config so paired
            # ref/novel trials run at the SAME (op, dtype, regime).
            normalised = dict(chosen)
            normalised["target_op"] = cell["target_op"]
            normalised["dtype"] = cell["dtype"]
            normalised["shape_regime"] = cell["shape_regime"]
            out.append(normalised)
        return out


@dataclass
class PairedControlProposer:
    """T-27. Wraps a base ``KernelProposer`` plus a list of reference
    seeds, and emits interleaved (reference, llm-novel) pairs at
    identical ``(target_op, dtype, shape_regime)`` cells so each LLM-
    novel kernel has a same-cell reference control to A/B against.

    Why: campaign 01 had 2 LLM-novel L3 trials at different cells than
    its 6 reference trials, so no honest novel-vs-reference comparison
    was possible (rmsnorm/large/bf16 had only the novel; silu_mul/
    medium/fp16 also only the novel). Paired control fixes the
    measurement design at the warmstart layer.

    Behaviour: ``propose_configs(surface, n, ...)`` returns at most
    ``n`` configs, alternating reference[i], novel[i], reference[i+1],
    novel[i+1], …, where novel[i] is an LLM-proposed kernel pinned to
    the same cell as reference[i]. If ``n`` is odd the last entry is
    a reference seed with no paired novel — the caller can right-size
    ``n`` (max_trials) to be even to avoid that.

    The LLM-novel half is generated via ``KernelProposer.propose_for_cells``;
    if the LLM returns no parseable block for some cell (transient
    failure), that cell falls back to its reference (and the pair
    degenerates to two identical-cell reference trials, which is
    still a useful baseline reproducibility check).
    """

    base: KernelProposer
    """Base LLM proposer used to generate the novel half of each pair."""

    reference_seeds: list[dict[str, Any]]
    """One reference config per cell to A/B at. Order is the schedule
    order; the first pair runs ``reference_seeds[0]`` then a novel
    pinned to that cell."""

    def __post_init__(self) -> None:
        if not self.reference_seeds:
            raise ValueError("reference_seeds must be non-empty")
        for s in self.reference_seeds:
            if not all(k in s for k in ("target_op", "dtype", "shape_regime")):
                raise ValueError(
                    "every reference seed must carry "
                    "target_op/dtype/shape_regime; got " + repr(s)
                )

    def propose_configs(
        self,
        surface: dict[str, dict[str, Any]],
        n: int,
        prior_notes: str,
        history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if n <= 0:
            raise ValueError("n must be positive")
        n_pairs = (n + 1) // 2
        cells_needed = [
            self.reference_seeds[i % len(self.reference_seeds)]
            for i in range(n_pairs)
        ]
        novels = self.base.propose_for_cells(
            cells=cells_needed,
            prior_notes=prior_notes,
            history=history,
        )
        out: list[dict[str, Any]] = []
        for i in range(n_pairs):
            out.append(dict(cells_needed[i]))
            if len(out) >= n:
                break
            out.append(dict(novels[i]))
            if len(out) >= n:
                break
        return out[:n]


__all__ = [
    "KernelProposer",
    "PairedControlProposer",
    "build_kernel_prompt",
    "build_paired_kernel_prompt",
    "parse_kernel_blocks",
]
