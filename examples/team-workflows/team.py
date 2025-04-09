import pixeltable as pxt

from pixelagent.openai import Agent

import yfinance as yf

# Define a simple tool with a description
@pxt.udf
def stock_price(ticker: str) -> dict:
    """Retrieve the current stock price for a given ticker symbol."""
    stock = yf.Ticker(ticker)
    return stock.info

financial_analyst = Agent(
    agent_name="financial_analyst",
    system_prompt="You are a CFA working at a top-tier investment bank.",
    tools=pxt.tools(stock_price),
    reset=True,
)

financial_analyst_udf = pxt.udf(financial_analyst, return_value=financial_analyst.answer)
portfolio_manager = Agent(
    agent_name="portfolio_manager",
    system_prompt="You are a CFA working at a top-tier investment bank.",
    tools=pxt.tools(financial_analyst_udf),
    reset=True,
)

# Test chat and tool_call functionality
print("--------------")
print(portfolio_manager.chat("Create the structure of a financial report for NVIDIA"))
print("--------------")
print(portfolio_manager.tool_call("Ask you financial analyst for a comprehensive analysis on NVIDIA"))
print("--------------")
print(portfolio_manager.chat("Finalize your report based on the analysis"))