import asyncio
import json
import os
from collections.abc import Iterator

import httpx2
import openai
import pixeltable as pxt
import pixeltable.functions.openai as pxt_openai
import pytest

from pixelagent.openai import Agent

requires_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)


@pytest.fixture
def openai_requests(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[dict]]:
    """Capture actual SDK request bodies without making network calls."""
    requests = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        requests.append(body)
        message = {"role": "assistant", "content": "The price is 5."}
        if body.get("tools"):
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_stock_price",
                    "type": "function",
                    "function": {
                        "name": "stock_price_dict",
                        "arguments": json.dumps({"ticker": "NVDA"}),
                    },
                }],
            }
        return httpx2.Response(200, json={
            "id": "test_completion",
            "object": "chat.completion",
            "created": 0,
            "model": body["model"],
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        })

    client = openai.AsyncOpenAI(
        api_key="test-placeholder",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(respond)),
        max_retries=0,
    )
    monkeypatch.setattr(pxt_openai, "_openai_client", lambda: client)
    yield requests
    asyncio.run(client.close())


@pytest.mark.parametrize("model", ["o3", "gpt-4o-mini"])
def test_openai_default_token_cap(openai_requests: list[dict], model: str) -> None:
    agent = Agent(name="openai_token_cap", system_prompt="Be helpful.", model=model, max_tokens=128, reset=True)
    assert agent.chat("Hello") == "The price is 5."
    assert openai_requests[0]["max_completion_tokens"] == 128
    assert "max_tokens" not in openai_requests[0]


@pytest.mark.parametrize("token_option", ["max_tokens", "max_completion_tokens"])
def test_openai_explicit_token_cap(openai_requests: list[dict], token_option: str) -> None:
    options = {token_option: 64}
    agent = Agent(
        name="openai_explicit_token_cap", system_prompt="Be helpful.",
        max_tokens=128, chat_kwargs=options, reset=True,
    )
    agent.chat("Hello")
    agent.chat("Hello again", model_kwargs={token_option: 32})
    assert [body[token_option] for body in openai_requests] == [64, 32]
    other_option = "max_completion_tokens" if token_option == "max_tokens" else "max_tokens"
    assert all(other_option not in body for body in openai_requests)
    assert options == {token_option: 64}


def test_openai_tool_token_cap(openai_requests: list[dict], mock_stock_price_dict: pxt.func.Function) -> None:
    agent = Agent(
        name="openai_tool_token_cap", system_prompt="Be helpful.", model="o3",
        tools=pxt.tools(mock_stock_price_dict), max_tokens=128, reset=True,
    )
    assert agent.tool_call("What is NVDA's price?") == "The price is 5."
    assert len(openai_requests) == 2
    assert all(body["max_completion_tokens"] == 128 for body in openai_requests)
    assert all("max_tokens" not in body for body in openai_requests)
    assert "tools" in openai_requests[0]
    assert "tools" not in openai_requests[1]
    assert "regularMarketPrice" in openai_requests[1]["messages"][0]["content"]


@pytest.fixture
def openai_agent(mock_stock_price_dict):
    """Fixture to create an OpenAI agent with tools"""
    return Agent(
        name="openai_financial_assistant",
        model="gpt-4o-mini",
        system_prompt="You are a financial analyst assistant.",
        tools=pxt.tools(mock_stock_price_dict),
        reset=True,
    )


@pytest.mark.live
@requires_key
@pytest.mark.openai
@pytest.mark.chat
def test_openai_basic_chat(openai_agent):
    """Test basic chat functionality of the OpenAI agent"""
    res1 = openai_agent.chat("when i say high you say low")
    assert res1 is not None, "Agent should return a response"

    res2 = openai_agent.chat("high")
    assert "low" in res2.lower(), "Agent should respond with 'low' when prompted with 'high'"


@pytest.mark.live
@requires_key
@pytest.mark.openai
@pytest.mark.tool_calling
def test_openai_tool_calling(openai_agent):
    """Test tool calling functionality of the OpenAI agent"""
    result = openai_agent.tool_call("What's the current price of NVDA?")
    assert result is not None, "Tool call should return a response"
    assert "5" in result, "Tool call should include the mock stock price value"
