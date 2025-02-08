# Pixel Agent

Build-your-own agent framework with Pixeltable.


## Why Pixeltable?

Pixeltable is a multimodal data infrastructure for building AI applications.

Pixelagent is a demonstration of the power of Pixeltable to build an agentic framework.

## Installation

```bash
pip install pixelagent openai
```

## Usage

```python
from pxl.providers import openai_agent

import yfinance as yf

@pxt.udf
def stock_info(ticker: str) -> Optional[dict]:
    """Get stock info for a given ticker symbol."""
    stock = yf.Ticker(ticker)
    return stock.info

# Persistent Agent with memory and tools
openai_agent.init(
    agent_name="Financial_Analyst",
    system_prompt="You are a financial analyst at a large NYC hedgefund.",
    model_name="gpt-4o-mini",
    agent_tools=pxt.tools(stock_info),
    reset_memory=False,
)

# Run the agent
result = openai_agent.run(
    agent_name="Financial_Analyst",
    message="""
    Fetch the latest stock information for Factset (Ticker: FDS).
    Create a 100 word summary of the company as a financial report in markdown format.
    """
)

print(result)

# Inspect agent history
inspect = pxt.get_table("Financial_Analyst")
df = inspect.collect().to_pandas()
print(df.head())
```
