import pixeltable as pxt
import yfinance as yf

from pxl.agent import openai_agent


@pxt.udf
def stock_info(ticker: str) -> dict:
    """Get stock info for a given ticker symbol."""
    stock = yf.Ticker(ticker)
    return stock.info


# Persistent Agent with memory and tools
openai_agent.init(
    agent_name="Financial_Analyst",
    system_prompt="You are a financial analyst, who can access yahoo finance data. Help the user with their stock analysis.",
    model_name="gpt-4o-mini",
    agent_tools=pxt.tools(stock_info),
    reset_memory=False,  # set to true to delete the agent history and start fresh
)

openai_agent.init(
    agent_name="Reflection",
    system_prompt="""
    Critique the assistant's response and provide feedback for improvement. 
    If the assistant's response is correct respond with <OK>
    """,
    model_name="gpt-4o-mini",
    agent_tools=pxt.tools(stock_info),
    reset_memory=False,  # set to true to delete the agent history and start fresh
)

# Run initial report
_report = openai_agent.run(
    agent_name="Financial_Analyst",
    message="Fetch the latest stock information for Factset (Ticker: FDS), and create a 100 summary of the company as a financial report in markdown format.",
)

max_iterations = 3

for i in range(max_iterations):
    msgs = openai_agent.get_messages("Financial_Analyst")

    critique = openai_agent.run(
        agent_name="Reflection",
        message=_report,
        additional_context=msgs,
    )

    print(critique)

    if "<OK>" in critique:
        break

    candidate_report = openai_agent.run(
        agent_name="Financial_Analyst",
        message=critique,
    )

    print(candidate_report)

    _report = candidate_report

print(_report)
