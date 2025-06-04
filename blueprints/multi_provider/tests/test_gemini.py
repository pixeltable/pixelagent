import os
import pytest

import pixeltable as pxt
from ..gemini import Agent


@pxt.udf
def weather(city: str) -> str:
    """
    Returns the weather in a given city.
    """
    return f"The weather in {city} is sunny."


@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set"
)
def test_gemini_agent_chat():
    """Test basic chat functionality with Gemini agent."""
    agent = Agent(
        name="test_gemini_agent",
        system_prompt="You're a helpful assistant.",
        reset=True,
    )
    
    # Test basic chat
    response = agent.chat("Hi, how are you?")
    assert isinstance(response, str)
    assert len(response) > 0
    
    # Test memory functionality
    response2 = agent.chat("What was my last question?")
    assert isinstance(response2, str)
    assert len(response2) > 0


@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set"
)
def test_gemini_agent_tools():
    """Test tool calling functionality with Gemini agent."""
    agent = Agent(
        name="test_gemini_tools",
        system_prompt="You're my assistant.",
        tools=pxt.tools(weather),
        reset=True,
    )
    
    # Test tool call
    response = agent.tool_call("Get weather in San Francisco")
    assert isinstance(response, str)
    assert "San Francisco" in response or "sunny" in response


@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set"
)
def test_gemini_agent_unlimited_memory():
    """Test unlimited memory functionality."""
    agent = Agent(
        name="test_gemini_unlimited",
        system_prompt="You're my assistant.",
        n_latest_messages=None,  # Unlimited memory
        reset=True,
    )
    
    # Test basic functionality with unlimited memory
    response = agent.chat("Remember this number: 42")
    assert isinstance(response, str)
    assert len(response) > 0
