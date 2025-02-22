from typing import Dict, List

import pixeltable as pxt
import yfinance as yf

from pixelagent.openai import Agent


@pxt.udf
def get_stock_info(ticker: str) -> Dict:
    """Get basic information about a stock."""
    stock = yf.Ticker(ticker)
    return stock.info


@pxt.udf
def get_price_history(ticker: str, period: str = "1mo") -> Dict:
    """Get historical price data for a stock."""
    stock = yf.Ticker(ticker)
    history = stock.history(period=period)
    return {
        "latest_price": float(history["Close"][-1]),
        "price_change": float(history["Close"][-1] - history["Close"][0]),
        "price_change_percent": float(
            (history["Close"][-1] - history["Close"][0]) / history["Close"][0] * 100
        ),
        "average_volume": float(history["Volume"].mean()),
    }


@pxt.udf
def get_recommendations(ticker: str) -> List[Dict]:
    """Get recent analyst recommendations."""
    stock = yf.Ticker(ticker)
    recs = stock.recommendations
    if recs is not None:
        return recs.tail(5).to_dict("records")
    return []


# Create tools collection
yfinance_tools = pxt.tools(get_stock_info, get_price_history, get_recommendations)

# Create agent with YFinance tools
agent = Agent(
    name="yfinance_analyst",
    system_prompt="You are a financial analyst, who can access yahoo finance data. Help the user with their stock analysis.",
    tools=yfinance_tools,
    reset=True,
)

# Example analysis
query = """Analyze NVIDIA (NVDA) stock:
1. Current price and recent performance
2. Analyst recommendations
Provide a brief investment summary."""

response = agent.run(query)
print("\nQuery:")
print(query)
print("\nAnalysis:")
print(response)
