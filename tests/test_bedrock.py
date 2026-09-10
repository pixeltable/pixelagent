import os

import pixeltable as pxt
import pytest

from pixelagent.bedrock import Agent

requires_key = pytest.mark.skipif(
    not os.getenv("AWS_ACCESS_KEY_ID"), reason="AWS_ACCESS_KEY_ID not set"
)


@pytest.fixture
def bedrock_agent(mock_stock_price):
    """Fixture to create a Bedrock agent with tools"""
    return Agent(
        name="bedrock_financial_assistant",
        model="amazon.nova-pro-v1:0",
        system_prompt="You are a financial analyst assistant.",
        tools=pxt.tools(mock_stock_price),
        reset=True,
    )


@pytest.mark.live
@requires_key
@pytest.mark.bedrock
@pytest.mark.chat
def test_bedrock_basic_chat(bedrock_agent):
    """Test basic chat functionality of the Bedrock agent"""
    res1 = bedrock_agent.chat("when i say high you say low")
    assert res1 is not None, "Agent should return a response"

    res2 = bedrock_agent.chat("high")
    assert "low" in res2.lower(), "Agent should respond with 'low' when prompted with 'high'"


@pytest.mark.live
@requires_key
@pytest.mark.bedrock
@pytest.mark.chat
def test_bedrock_inference_config_is_honored():
    """chat_kwargs now route through inference_config=; prove Bedrock accepts them.

    This is the wire-level check that keyless CI cannot make.
    """
    agent = Agent(
        name="bedrock_inference_config_probe",
        model="amazon.nova-pro-v1:0",
        system_prompt="You are a helpful assistant.",
        chat_kwargs={"temperature": 0.0, "maxTokens": 64},
        reset=True,
    )
    assert agent.chat("Say hello.") is not None


@pytest.mark.live
@requires_key
@pytest.mark.bedrock
@pytest.mark.tool_calling
def test_bedrock_tool_calling(bedrock_agent):
    """Test tool calling functionality of the Bedrock agent"""
    result = bedrock_agent.tool_call("What's the current price of NVDA?")
    assert result is not None, "Tool call should return a response"
    assert "5" in result, "Tool call should include the mock stock price value"
