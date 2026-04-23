"""Real ``ProposalLLM`` implementations.

Two classes cover the common cases:

- ``OpenAICompatibleProposalLLM`` — OpenAI, vLLM OpenAI-spec server,
  OpenRouter, or any other endpoint implementing ``/v1/chat/completions``.
  Uses ``httpx`` directly; no vendor SDK dependency.
- ``AnthropicProposalLLM`` — Claude via the official ``anthropic`` SDK.
  Gated by the ``autoinfer[llm]`` extra.

Both conform to ``autoinfer.policy.warmstart.ProposalLLM`` and produce
configs validated against the supplied surface (unknown keys raise).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")


def build_prompt(
    surface: dict[str, dict[str, Any]],
    n: int,
    prior_notes: str,
    history: list[dict[str, Any]],
) -> str:
    """Compose the proposer prompt. Pure; used by all provider classes."""
    surface_json = json.dumps(surface, indent=2, default=str, sort_keys=True)
    tail = history[-20:] if history else []
    history_json = json.dumps(tail, indent=2, default=str, sort_keys=True)
    notes = prior_notes.strip() or "(none)"
    return (
        "You are proposing configurations for an LLM inference-engine "
        "search loop. Pick values that explore the search surface "
        "informatively given the trial history.\n\n"
        f"Search surface (knob -> type and range or values):\n"
        f"```json\n{surface_json}\n```\n\n"
        f"Prior notes: {notes}\n\n"
        f"Recent trial history (last 20 trials):\n```json\n{history_json}\n```\n\n"
        f"Propose exactly {n} configurations. Each configuration is a JSON "
        "object mapping knob name to a value drawn from the surface. "
        "Return ONLY a JSON array of objects with no surrounding prose. "
        "Example format:\n"
        '```json\n[{"knob_a": 42, "knob_b": "foo"}]\n```\n'
    )


def parse_configs(
    text: str, surface: dict[str, dict[str, Any]], n: int
) -> list[dict[str, Any]]:
    """Extract and validate configs from an LLM response. Pure."""
    match = _JSON_ARRAY_RE.search(text)
    if not match:
        raise ValueError(f"no JSON array in LLM response: {text[:200]!r}")
    configs = json.loads(match.group(0))
    if not isinstance(configs, list):
        raise ValueError("LLM response JSON is not a list")
    out: list[dict[str, Any]] = []
    for cfg in configs:
        if not isinstance(cfg, dict):
            continue
        for key in cfg:
            if key not in surface:
                raise ValueError(f"LLM proposed unknown knob {key!r}")
        out.append(cfg)
    if not out:
        raise ValueError("no valid configs in LLM response")
    return out[:n]


@dataclass
class OpenAICompatibleProposalLLM:
    """OpenAI / vLLM / OpenRouter / any /v1/chat/completions endpoint."""

    base_url: str
    model: str
    api_key: str | None = None
    timeout_s: float = 120.0
    max_tokens: int = 4096
    temperature: float = 0.3
    transport: httpx.BaseTransport | None = None

    def propose_configs(
        self,
        surface: dict[str, dict[str, Any]],
        n: int,
        prior_notes: str,
        history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        prompt = build_prompt(surface, n, prior_notes, history)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        kwargs: dict[str, Any] = {"timeout": self.timeout_s}
        if self.transport is not None:
            kwargs["transport"] = self.transport
        with httpx.Client(**kwargs) as client:
            resp = client.post(
                f"{self.base_url.rstrip('/')}/v1/chat/completions",
                headers=headers,
                json=body,
            )
            resp.raise_for_status()
            payload = resp.json()
        text = payload["choices"][0]["message"]["content"]
        return parse_configs(text, surface, n)


@dataclass
class AnthropicProposalLLM:
    """Anthropic Claude via the official SDK; requires ``autoinfer[llm]``."""

    model: str
    api_key: str | None = None
    max_tokens: int = 4096
    _client: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            import anthropic  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "AnthropicProposalLLM requires the 'llm' extra: "
                "`uv sync --extra dev --extra llm`"
            ) from e
        self._client = (
            anthropic.Anthropic(api_key=self.api_key)
            if self.api_key
            else anthropic.Anthropic()
        )

    def propose_configs(
        self,
        surface: dict[str, dict[str, Any]],
        n: int,
        prior_notes: str,
        history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        prompt = build_prompt(surface, n, prior_notes, history)
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text for block in resp.content if getattr(block, "text", None)
        )
        return parse_configs(text, surface, n)
