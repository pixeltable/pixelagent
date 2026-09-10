import os

import pixeltable as pxt
import pytest

from pixelagent.gemini import Agent

requires_key = pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set"
)


@pxt.udf
def weather(city: str) -> str:
    """Returns the weather in a given city."""
    return f"The weather in {city} is sunny."


@pytest.mark.live
@requires_key
@pytest.mark.gemini
@pytest.mark.chat
def test_gemini_agent_chat():
    """Test basic chat functionality with Gemini agent."""
    agent = Agent(
        name="gemini_chat_assistant",
        system_prompt="You're a helpful assistant.",
        reset=True,
    )

    response = agent.chat("Hi, how are you?")
    assert isinstance(response, str)
    assert len(response) > 0

    response2 = agent.chat("What was my last question?")
    assert isinstance(response2, str)
    assert len(response2) > 0


@pytest.mark.live
@requires_key
@pytest.mark.gemini
@pytest.mark.tool_calling
def test_gemini_agent_tools():
    """Test tool calling functionality with Gemini agent."""
    agent = Agent(
        name="gemini_tools_assistant",
        system_prompt="You're my assistant.",
        tools=pxt.tools(weather),
        reset=True,
    )

    response = agent.tool_call("Get weather in San Francisco")
    assert isinstance(response, str)
    assert "San Francisco" in response or "sunny" in response
