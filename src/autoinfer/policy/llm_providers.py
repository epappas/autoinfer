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
    """Compose the proposer prompt. Pure; used by all provider classes.

    The caller is expected to fold hardware-compat information into
    ``prior_notes`` (GPU class, supported precisions, known-broken knob
    combinations on this hardware). Without it the proposer will happily
    explore infeasible regions — observed empirically in the 20-trial
    Sonnet 4 campaign where FP8 on Ampere was proposed repeatedly.
    """
    surface_json = json.dumps(surface, indent=2, default=str, sort_keys=True)
    tail = history[-30:] if history else []
    history_json = json.dumps(tail, indent=2, default=str, sort_keys=True)
    notes = prior_notes.strip() or "(none)"
    return (
        "You are proposing vLLM engine-configuration variants for an LLM "
        "inference-engine search. Every proposal that crashes or fails a "
        "quality gate is wasted budget; the trial history shows which "
        "regions already failed and why. Exploit what works, explore "
        "genuinely new regions, avoid configurations the notes or history "
        "indicate are infeasible on this hardware.\n\n"
        f"Search surface (knob -> type and range or values):\n"
        f"```json\n{surface_json}\n```\n\n"
        f"Hardware and compatibility notes:\n{notes}\n\n"
        f"Recent trial history (last {len(tail)} trials):\n"
        f"```json\n{history_json}\n```\n\n"
        f"Propose exactly {n} configurations. Each is a JSON object "
        "mapping knob name to a value drawn from the surface. Return "
        "ONLY a JSON array of objects with no surrounding prose. "
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
        # Surface is the surrogate-searchable subset; adapters may accept
        # pass-through keys beyond it (e.g. L3 ``source``). Trust the
        # adapter's own validation rather than reject extras here.
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

    def complete(self, prompt: str) -> str:
        """Single-prompt chat-completion call. Returns the model's text."""
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
        # Follow OpenAI SDK convention: base_url includes the API version
        # (e.g. https://api.openai.com/v1, https://openrouter.ai/api/v1,
        # http://vllm-host:port/v1) and we append only /chat/completions.
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        with httpx.Client(**kwargs) as client:
            resp = client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            payload = resp.json()
        text: str = payload["choices"][0]["message"]["content"]
        return text

    def propose_configs(
        self,
        surface: dict[str, dict[str, Any]],
        n: int,
        prior_notes: str,
        history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        prompt = build_prompt(surface, n, prior_notes, history)
        text = self.complete(prompt)
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
            import anthropic  # type: ignore[import-not-found, unused-ignore]
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

    def complete(self, prompt: str) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text: str = "".join(
            block.text for block in resp.content if getattr(block, "text", None)
        )
        return text

    def propose_configs(
        self,
        surface: dict[str, dict[str, Any]],
        n: int,
        prior_notes: str,
        history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        prompt = build_prompt(surface, n, prior_notes, history)
        text = self.complete(prompt)
        return parse_configs(text, surface, n)
