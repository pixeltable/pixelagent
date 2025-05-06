"""
Common pytest fixtures and configurations for pixelagent tests.
"""
import pytest
import pixeltable as pxt


# Define UDFs at module level to avoid nested function errors
@pxt.udf
def stock_price_int(ticker: str) -> int:
    """Get stock information for a ticker symbol (returns integer)"""
    return 5


@pxt.udf
def stock_price_dict(ticker: str) -> dict:
    """Get stock information for a ticker symbol (returns dictionary)"""
    return {"regularMarketPrice": 5, "shortName": "NVIDIA Corporation"}


@pytest.fixture
def mock_stock_price():
    """Fixture to provide a mock stock price tool that returns a fixed value.
    
    This avoids external API dependencies during testing.
    """
    return stock_price_int


@pytest.fixture
def mock_stock_price_dict():
    """Fixture to provide a mock stock price tool that returns a dictionary.
    
    This is useful for testing OpenAI which expects a more complex return value.
    """
    return stock_price_dict
