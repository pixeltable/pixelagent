from typing import Dict, List

import yfinance as yf

from pixelagent.openai import Agent, tool


@tool
def get_stock_info(ticker: str) -> Dict:
    """Get basic information about a stock."""
    stock = yf.Ticker(ticker)
    return stock.info


@tool
def get_recommendations(ticker: str) -> List[Dict]:
    """Get recent analyst recommendations."""
    stock = yf.Ticker(ticker)
    recs = stock.recommendations
    if recs is not None:
        return recs.tail(5).to_dict("records")
    return []


agent = Agent(
    name="yfinance_analyst",
    system_prompt="You are a financial analyst, who can access yahoo finance data. Help the user with their stock analysis.",
    tools=[get_stock_info, get_recommendations],
    reset=True,
)

query = "Provide a 100 word summary of FDS stock"

response = agent.run(query)
print(response)
