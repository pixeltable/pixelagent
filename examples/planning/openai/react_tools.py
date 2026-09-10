"""Tool UDFs for react.py.

They live in a module rather than beside the agent because Pixeltable refuses a
@pxt.udf defined in the global namespace of a script: a UDF has to be importable
by name so a stored computed column can find it again on a later run.
"""


import pixeltable as pxt
import yfinance as yf


@pxt.udf
def stock_info(ticker: str) -> dict:
    """
    Retrieve comprehensive stock information for a given ticker symbol.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL' for Apple)

    Returns:
        dict: Dictionary containing stock information and metrics
    """
    stock = yf.Ticker(ticker)
    return stock.info
