from typing import Dict, List

import pixeltable as pxt
from pixelagent.openai import Agent

import yfinance as yf

@pxt.udf
def get_stock_info(ticker: str) -> Dict:
    """Get basic information about a stock."""
    stock = yf.Ticker(ticker)
    return stock.info

@pxt.udf
def get_price_history(ticker: str, period: str = '1mo') -> Dict:
    """Get historical price data for a stock."""
    stock = yf.Ticker(ticker)
    history = stock.history(period=period)
    return {
        'latest_price': float(history['Close'].iloc[-1]),
        'price_change': float(history['Close'].iloc[-1] - history['Close'].iloc[0]),
        'price_change_percent': float((history['Close'].iloc[-1] - history['Close'].iloc[0]) / history['Close'].iloc[0] * 100),
        'average_volume': float(history['Volume'].mean())
    }

@pxt.udf
def get_recommendations(ticker: str) -> List[Dict]:
    """Get recent analyst recommendations."""
    stock = yf.Ticker(ticker)
    recs = stock.recommendations
    if recs is not None:
        return recs.tail(5).to_dict('records')
    return []

# Create tools collection
yfinance_tools = pxt.tools(
    get_stock_info,
    get_price_history,
    get_recommendations
)

# Create agent with YFinance tools
analyst = Agent(
    name="yfinance_analyst", 
    system_prompt="You are a financial analyst, who can access yahoo finance data. Help the user with their stock analysis.", 
    tools=yfinance_tools,
    reset=True
)

@pxt.udf
def ask_analyst(query: str) -> str:
    """State your query to the analyst and get a response."""
    return analyst.run(query)

portfolio_manager = Agent(
    name="portfolio_manager", 
    system_prompt="Work with your analyst to build a report on stocks in your portfolio.", 
    tools=pxt.tools(ask_analyst),
    reset=True
)

response = portfolio_manager.run("I need a exhaustive report on PLTR stock. Make sure to include all the details.")
print("\nAnalysis:")
print(response)