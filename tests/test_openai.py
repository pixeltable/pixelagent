import pytest
import pixeltable as pxt
from pixelagent.openai import Agent


@pytest.fixture
def openai_agent(mock_stock_price_dict):
    """Fixture to create an OpenAI agent with tools"""
    agent = Agent(
        name="financial_assistant",
        model="gpt-4o-mini",
        system_prompt="You are a financial analyst assistant.",
        tools=pxt.tools(mock_stock_price_dict)
    )
    return agent


@pytest.mark.openai
@pytest.mark.chat
def test_openai_basic_chat(openai_agent):
    """Test basic chat functionality of the OpenAI agent"""
    # Initial conversation
    res1 = openai_agent.chat("when i say high you say low")
    assert res1 is not None, "Agent should return a response"
    
    # Follow-up message
    res2 = openai_agent.chat("high")
    assert "low" in res2.lower(), "Agent should respond with 'low' when prompted with 'high'"


@pytest.mark.openai
@pytest.mark.tool_calling
def test_openai_tool_calling(openai_agent):
    """Test tool calling functionality of the OpenAI agent"""
    result = openai_agent.tool_call("What's the current price of NVDA?")
    assert result is not None, "Tool call should return a response"
    assert '5' in result, "Tool call should include the mock stock price value"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
