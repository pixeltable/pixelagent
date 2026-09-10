"""Mock tool UDFs shared by the test suite.

These live in a module, not in `conftest.py`. Pixeltable rejects a `@pxt.udf`
defined in the global namespace of a script (`pixeltable/func/udf.py`); a UDF has
to be importable by module path so a stored computed column can resolve it later.
"""

import pixeltable as pxt


@pxt.udf
def stock_price_int(ticker: str) -> int:
    """Get stock information for a ticker symbol (returns integer)"""
    return 5


@pxt.udf
def stock_price_dict(ticker: str) -> dict:
    """Get stock information for a ticker symbol (returns dictionary)"""
    return {"regularMarketPrice": 5, "shortName": "NVIDIA Corporation"}
