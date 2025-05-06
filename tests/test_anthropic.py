import pytest
import pixeltable as pxt
from pixelagent.anthropic import Agent


@pytest.fixture
def anthropic_agent(mock_stock_price):
    """Fixture to create an Anthropic agent with tools"""
    agent = Agent(
        name="financial_assistant",
        model="claude-3-7-sonnet-latest",
        system_prompt="You are a financial analyst assistant.",
        tools=pxt.tools(mock_stock_price)
    )
    return agent


@pytest.mark.anthropic
@pytest.mark.chat
def test_anthropic_basic_chat(anthropic_agent):
    """Test basic chat functionality of the Anthropic agent"""
    # Initial conversation
    res1 = anthropic_agent.chat("when i say high you say low")
    assert res1 is not None, "Agent should return a response"
    
    # Follow-up message
    res2 = anthropic_agent.chat("high")
    assert "low" in res2.lower(), "Agent should respond with 'low' when prompted with 'high'"


@pytest.mark.anthropic
@pytest.mark.tool_calling
def test_anthropic_tool_calling(anthropic_agent):
    """Test tool calling functionality of the Anthropic agent"""
    result = anthropic_agent.tool_call("What's the current price of NVDA?")
    assert result is not None, "Tool call should return a response"
    assert '5' in result, "Tool call should include the mock stock price value"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
