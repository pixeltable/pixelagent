import yfinance as yf

import pixeltable as pxt
from pxl.agent import initialize_agent, run_agent

from typing import Optional

@pxt.udf
def stock_info(ticker: str) -> Optional[dict]:
    """Get stock info for a given ticker symbol."""
    stock = yf.Ticker(ticker)
    return stock.info

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
    agent_name="Self_Reflecting_Agent",
    system_prompt="""
    Critique the assistant's work. Determine if the the assistant fully answered the users question. If say say <OK> otherwise provide feedback to the assistant.
    """,
    model_name="gpt-4o-mini",
    reset_memory=True  # set to true to delete the agent and start fresh
)

# Run the portfolio manager agent
question = "What was the price change of FDS last week, and what happened?"
report = run_agent("Financial_Analyst", question)



