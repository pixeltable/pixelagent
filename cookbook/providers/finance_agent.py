from pixelagent import Agent
from pixelagent.llms import OpenAIModel
import yfinance as yf
import pixeltable as pxt
from typing import Optional

@pxt.udf
def stock_info(ticker: str) -> Optional[dict]:
    """Get stock info for a given ticker symbol."""
    stock = yf.Ticker(ticker)
    return stock.info

# print(stock_info("NVDA"))

model = OpenAIModel(model_name="gpt-4o-mini")
agent_with_tools = Agent(
    model=model,
    tools=[stock_info],
    system_prompt="You are a helpful financial assistant."
)

response = agent_with_tools.run("What is the stock price of NVDA today?")
print(f"Answer: {response.answer}")
print(f"Tool outputs: {response.tool_outputs}")