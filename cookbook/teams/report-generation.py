from typing import Optional

import pixeltable as pxt
import yfinance as yf

from pxl.agent import openai_agent


@pxt.udf
def stock_info(ticker: str) -> Optional[dict]:
    """Get stock info for a given ticker symbol."""
    stock = yf.Ticker(ticker)
    return stock.info


# Initialize the financial analyst agent
openai_agent.init(
    agent_name="Financial_Analyst",
    system_prompt="You are a financial analyst at a large NYC hedgefund.",
    model_name="gpt-4o-mini",
    agent_tools=pxt.tools(stock_info),
    reset_memory=False,
)


@pxt.udf
def analyst(prompt: str) -> str:
    """Get stock info for a given ticker symbol."""
    return openai_agent.run("Financial_Analyst", prompt)


# Initialize the portfolio manager agent
openai_agent.init(
    agent_name="Portfolio_Manager",
    system_prompt="""

    You are a portfolio manager use your financial analyst to help you manage the portfolio.
    You have access to a financial analyst who can help you with your research. Use at your discretion.
    """,
    model_name="gpt-4o-mini",
    agent_tools=pxt.tools(analyst),
    reset_memory=False,
)



ticker = "Harley Davidson"
sections = ["Company Overview", "Key Metrics", "Conclusion"]

openai_agent.run(
    agent_name="Portfolio_Manager",
    message="Create the company pitch report structure. Here is the general structure: "
    + ",".join(sections),
)


for section in sections:
    print(f"Generating section: {section}")
    # Each section is automatically stored in the conversation.
    openai_agent.run(
        agent_name="Portfolio_Manager",
        message=f"Work with your financial analyst to generate a detailed report on the company: {ticker} for the section: {section}",
    )


final_report = openai_agent.run(
    agent_name="Portfolio_Manager",
    message="Finalize the report. Place your report in these brackets: <REPORT> </REPORT>",
)


print(final_report)
