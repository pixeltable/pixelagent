import yfinance as yf

import pixeltable as pxt
from pxl.agent import initialize_agent, run_agent

from typing import Optional

@pxt.udf
def stock_info(ticker: str) -> Optional[dict]:
    """Get stock info for a given ticker symbol."""
    stock = yf.Ticker(ticker)
    return stock.info

@pxt.udf
def analyst(prompt: str) -> str:
    """Get stock info for a given ticker symbol."""
    return run_agent("Financial_Analyst", prompt)

# Initialize the financial analyst agent
initialize_agent(
    agent_name="Financial_Analyst",
    system_prompt="You are a financial analyst at a large NYC hedgefund.",
    model_name="gpt-4o-mini",
    agent_tools=pxt.tools(stock_info),
    reset_memory=True  # set to true to delete the agent and start fresh
)

# Initialize the portfolio manager agent
initialize_agent(
    agent_name="Portfolio_Manager",
    system_prompt="""
    You are a portfolio manager use your financial analyst to help you manage the portfolio.
    You have access to a financial analyst who can help you with your research.
    Be sure to give the financial analyst a good amount of detail. Dont be afraid to ask for more information.
    """,
    model_name="gpt-4o-mini",
    agent_tools=pxt.tools(analyst),
    reset_memory=True  # set to true to delete the agent and start fresh
)

# Run the portfolio manager agent
report = run_agent("Portfolio_Manager", "What was the price change of FDS last week.")
print(report)


