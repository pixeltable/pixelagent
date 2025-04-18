import random

import pixeltable as pxt
import yfinance as yf

from pixelagent.openai import Agent


# Define a simple tool with a description
@pxt.udf
def stock_price(ticker: str) -> dict:
    """Retrieve the current stock price for a given ticker symbol."""
    stock = yf.Ticker(ticker)
    return stock.info


# Define a new tool to get a random trading action
@pxt.udf
def analyst_recommendation(ticker: str) -> str:
    """Randomly select a trading action: buy, sell, or hold."""
    return random.choice(["buy", "sell", "hold"])


# Create an agent with tools
agent = Agent(
    name="financial_analyst",
    system_prompt="You are a CFA working at a top-tier investment bank.",
    tools=pxt.tools(stock_price, analyst_recommendation),
    reset=True,
)

# Test chat and tool_call functionality
print("--------------")
print(agent.chat("Hi, how are you?"))
print("--------------")
print(agent.tool_call("Get NVIDIA and Apple stock price"))
print("--------------")
print(agent.chat("What was my last question?"))
print("--------------")
print(agent.chat("What's the recommendation for NVIDIA?"))

agent_memory = pxt.get_table("financial_analyst.memory")
print(agent_memory.select(agent_memory.content).collect())
