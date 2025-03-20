import pixeltable as pxt

from pixelagent.openai import Agent


# Define a simple tool with a description
@pxt.udf
def stock_price(ticker: str) -> dict:
    """Retrieve the current stock price for a given ticker symbol."""
    import yfinance as yf

    stock = yf.Ticker(ticker)
    info = stock.info
    return {"price": info.get("regularMarketPrice", "N/A")}


# Create the tools object with the UDF
tools = pxt.tools(stock_price)

# Create an agent with tools
agent = Agent(
    agent_name="openai_bot",
    system_prompt="You’re my assistant.",
    tools=tools,
    reset=True,
)

# Test chat and tool_call functionality
print(agent.chat("Hi, how are you?"))
print(agent.tool_call("Get NVIDIA and Apple  stock price"))
print(agent.chat("What was my last question?"))
