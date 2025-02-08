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
    reset_memory=True
)

@pxt.udf
def analyst(prompt: str) -> str:
    """Get stock info for a given ticker symbol."""
    response = run_agent(
        agent_name="Financial_Analyst",
        message=prompt
    )
    return response

# Initialize the portfolio manager agent
initialize_agent(
    agent_name="Portfolio_Manager",
    system_prompt="""
    You are a portfolio manager use your financial analyst to help you manage the portfolio.
    You have access to a financial analyst who can help you with your research.
    Be sure to give the financial analyst a good amount of detail. Dont be afraid to ask for more information.
    """,
    model_name="gpt-4o-mini",
    agent_tools=pxt.tools(analyst),
    reset_memory=True
)

ticker = "Harley Davidson"
sections = [
    "Company Overview",
    "Key Metrics",
    "Conclusion"
]

run_agent(
    agent_name="Portfolio_Manager",
    message="Create the company pitch report structure. Here is the general structure: " + ",".join(sections)
)

for section in sections:
    print(f"Generating section: {section}")
    # Each section is automatically stored in the conversation.
    run_agent(
        agent_name="Portfolio_Manager",
        message=f"Work with your financial analyst to generate a detailed report on the company: {ticker} for the section: {section}"
    )

final_report = run_agent(
    agent_name="Portfolio_Manager",
    message="Great work. Finalize your report and make sure you follow your structure:" + ",".join(sections)
)


print(final_report)