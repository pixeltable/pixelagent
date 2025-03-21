import pixeltable as pxt

from pixelagent.anthropic import Agent

import yfinance as yf

# Define a simple tool with a description
@pxt.udf
def stock_price(ticker: str) -> dict:
    """Retrieve the current stock price for a given ticker symbol."""
    stock = yf.Ticker(ticker)
    return stock.info

# Create the tools object with the UDF
tools = pxt.tools(stock_price)

# Create an agent with tools
agent = Agent(
    agent_name="financial_analyst",
    system_prompt="You are a CFA working at a top-tier investment banks.",
    tools=tools,
    reset=True,
)

# Test chat and tool_call functionality
print("--------------")
print(agent.chat("Hi, how are you?"))
print("--------------")
print(agent.tool_call("Get NVIDIA and Apple stock price"))
print("--------------")
print(agent.chat("What was my last question?"))
