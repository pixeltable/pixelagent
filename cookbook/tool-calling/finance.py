import yfinance as yf

import pixeltable as pxt
from pxl.providers import openai_agent


from typing import Optional


@pxt.udf
def stock_info(ticker: str) -> Optional[dict]:
    """Get stock info for a given ticker symbol."""
    stock = yf.Ticker(ticker)
    return stock.info


# Persistent Agent with memory and tools
openai_agent.init(
    agent_name="Financial_Analyst",
    system_prompt="You are a financial analyst, who can access yahoo finance data. Help the user with their stock analysis.",
    model_name="gpt-4o-mini",
    agent_tools=pxt.tools(stock_info),
    reset_memory=True,  # set to true to delete the agent and start fresh
)

report = openai_agent.run(
    agent_name="Financial_Analyst",
    message="Fetch the latest stock information for Factset (Ticker: FDS), and create a 100 summary of the company as a financial report in markdown format.",
)

# inspect
print(report)
