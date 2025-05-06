import pytest
import pixeltable as pxt
from pixelagent.bedrock import Agent


@pytest.fixture
def bedrock_agent(mock_stock_price):
    """Fixture to create a Bedrock agent with tools"""
    agent = Agent(
        name="financial_assistant",
        model="amazon.nova-pro-v1:0",
        system_prompt="You are a financial analyst assistant.",
        tools=pxt.tools(mock_stock_price)
    )
    return agent


@pytest.mark.bedrock
@pytest.mark.chat
def test_bedrock_basic_chat(bedrock_agent):
    """Test basic chat functionality of the Bedrock agent"""
    # Initial conversation
    res1 = bedrock_agent.chat("when i say high you say low")
    assert res1 is not None, "Agent should return a response"
    
    # Follow-up message
    res2 = bedrock_agent.chat("high")
    assert "low" in res2.lower(), "Agent should respond with 'low' when prompted with 'high'"


@pytest.mark.bedrock
@pytest.mark.tool_calling
def test_bedrock_tool_calling(bedrock_agent):
    """Test tool calling functionality of the Bedrock agent"""
    result = bedrock_agent.tool_call("What's the current price of NVDA?")
    assert result is not None, "Tool call should return a response"
    assert '5' in result, "Tool call should include the mock stock price value"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
