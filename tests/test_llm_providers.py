from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from autoinfer.policy.llm_providers import (
    OpenAICompatibleProposalLLM,
    build_prompt,
    parse_configs,
)
from autoinfer.policy.warmstart import ProposalLLM


def _surface() -> dict[str, dict[str, Any]]:
    return {
        "max_num_batched_tokens": {"type": "int", "low": 1024, "high": 16384},
        "kv_cache_dtype": {"type": "categorical", "values": ["auto", "fp8"]},
    }


def test_build_prompt_contains_surface_and_history() -> None:
    prompt = build_prompt(
        surface=_surface(),
        n=3,
        prior_notes="explore low KV dtype",
        history=[{"trial_id": "t1", "config": {"max_num_batched_tokens": 2048}}],
    )
    assert "max_num_batched_tokens" in prompt
    assert "explore low KV dtype" in prompt
    assert '"trial_id": "t1"' in prompt
    assert "Propose exactly 3" in prompt


def test_parse_configs_accepts_fenced_response() -> None:
    text = 'prose prose ```json\n[{"max_num_batched_tokens": 4096}]\n``` more prose'
    out = parse_configs(text, _surface(), n=5)
    assert out == [{"max_num_batched_tokens": 4096}]


def test_parse_configs_rejects_unknown_knob() -> None:
    text = '[{"mystery_knob": 42}]'
    with pytest.raises(ValueError):
        parse_configs(text, _surface(), n=1)


def test_parse_configs_no_json_array_raises() -> None:
    with pytest.raises(ValueError):
        parse_configs("no array here", _surface(), n=1)


def test_parse_configs_truncates_to_n() -> None:
    text = '[{"kv_cache_dtype": "auto"}, {"kv_cache_dtype": "fp8"}, {"kv_cache_dtype": "auto"}]'
    out = parse_configs(text, _surface(), n=2)
    assert len(out) == 2


def test_openai_compatible_protocol_conformance() -> None:
    llm = OpenAICompatibleProposalLLM(
        base_url="http://example",
        model="test",
        api_key="k",
    )
    assert isinstance(llm, ProposalLLM)


def test_openai_compatible_sends_correct_request() -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(
            {
                "method": request.method,
                "url": str(request.url),
                "auth_header": request.headers.get("authorization"),
                "body": json.loads(request.content),
            }
        )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '[{"kv_cache_dtype": "auto"}]',
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    llm = OpenAICompatibleProposalLLM(
        base_url="https://api.example.com",
        model="test-model",
        api_key="sk-test",
        transport=transport,
    )
    out = llm.propose_configs(
        surface=_surface(),
        n=1,
        prior_notes="",
        history=[],
    )
    assert out == [{"kv_cache_dtype": "auto"}]
    assert len(captured) == 1
    req = captured[0]
    assert req["method"] == "POST"
    assert "v1/chat/completions" in req["url"]
    assert req["auth_header"] == "Bearer sk-test"
    assert req["body"]["model"] == "test-model"
    assert req["body"]["messages"][0]["role"] == "user"
    assert "max_num_batched_tokens" in req["body"]["messages"][0]["content"]


def test_openai_compatible_no_api_key_omits_auth_header() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '[{"kv_cache_dtype":"auto"}]'}}]},
        )

    transport = httpx.MockTransport(handler)
    llm = OpenAICompatibleProposalLLM(
        base_url="http://localhost:9000",
        model="local-model",
        api_key=None,
        transport=transport,
    )
    llm.propose_configs(surface=_surface(), n=1, prior_notes="", history=[])
    assert captured[0].headers.get("authorization") is None


def test_openai_compatible_http_error_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    transport = httpx.MockTransport(handler)
    llm = OpenAICompatibleProposalLLM(
        base_url="http://x",
        model="m",
        api_key=None,
        transport=transport,
    )
    with pytest.raises(httpx.HTTPStatusError):
        llm.propose_configs(surface=_surface(), n=1, prior_notes="", history=[])


def test_openai_compatible_bad_response_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "I refuse to output JSON"}}]},
        )

    transport = httpx.MockTransport(handler)
    llm = OpenAICompatibleProposalLLM(
        base_url="http://x",
        model="m",
        api_key=None,
        transport=transport,
    )
    with pytest.raises(ValueError):
        llm.propose_configs(surface=_surface(), n=1, prior_notes="", history=[])
