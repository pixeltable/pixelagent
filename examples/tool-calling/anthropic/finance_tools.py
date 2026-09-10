"""Tool UDFs for finance.py.

They live in a module rather than beside the agent because Pixeltable refuses a
@pxt.udf defined in the global namespace of a script: a UDF has to be importable
by name so the stored computed column can find it again on a later run.
"""

import random

import pixeltable as pxt
import yfinance as yf


@pxt.udf
def stock_price(ticker: str) -> dict:
    """Retrieve the current stock price for a given ticker symbol."""
    stock = yf.Ticker(ticker)
    return stock.info


@pxt.udf
def analyst_recommendation(ticker: str) -> str:
    """Randomly select a trading action: buy, sell, or hold."""
    return random.choice(["buy", "sell", "hold"])
