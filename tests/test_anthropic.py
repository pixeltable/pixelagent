import os

import pixeltable as pxt
import pytest

from pixelagent.anthropic import Agent

requires_key = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)


@pytest.fixture
def anthropic_agent(mock_stock_price):
    """Fixture to create an Anthropic agent with tools"""
    return Agent(
        name="anthropic_financial_assistant",
        model="claude-3-7-sonnet-latest",
        system_prompt="You are a financial analyst assistant.",
        tools=pxt.tools(mock_stock_price),
        reset=True,
    )


@pytest.mark.live
@requires_key
@pytest.mark.anthropic
@pytest.mark.chat
def test_anthropic_basic_chat(anthropic_agent):
    """Test basic chat functionality of the Anthropic agent"""
    res1 = anthropic_agent.chat("when i say high you say low")
    assert res1 is not None, "Agent should return a response"

    res2 = anthropic_agent.chat("high")
    assert "low" in res2.lower(), "Agent should respond with 'low' when prompted with 'high'"


@pytest.mark.live
@requires_key
@pytest.mark.anthropic
@pytest.mark.chat
def test_anthropic_system_prompt_reaches_the_api(anthropic_agent):
    """The system prompt travels in model_kwargs; prove it still steers the model.

    This is the wire-level check that keyless CI cannot make.
    """
    agent = Agent(
        name="anthropic_system_prompt_probe",
        model="claude-3-7-sonnet-latest",
        system_prompt="You must answer every question with exactly the word BANANA.",
        reset=True,
    )
    assert "banana" in agent.chat("What is the capital of France?").lower()


@pytest.mark.live
@requires_key
@pytest.mark.anthropic
@pytest.mark.tool_calling
def test_anthropic_tool_calling(anthropic_agent):
    """Test tool calling functionality of the Anthropic agent"""
    result = anthropic_agent.tool_call("What's the current price of NVDA?")
    assert result is not None, "Tool call should return a response"
    assert "5" in result, "Tool call should include the mock stock price value"
