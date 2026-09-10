import pixeltable as pxt
from finance_tools import analyst_recommendation, stock_price

from pixelagent.anthropic import Agent

# Create an agent with tools
agent = Agent(
    name="financial_analyst",
    system_prompt="You are a CFA working at a top-tier investment bank.",
    tools=pxt.tools(stock_price, analyst_recommendation),
    reset=True,
)

# Test chat and tool_call functionality
print("--------------")
print(agent.chat("Hi, how are you?"))
print("--------------")
print(agent.tool_call("Get NVIDIA and Apple stock price"))
print("--------------")
print(agent.chat("What was my last question?"))
print("--------------")
print(agent.chat("What's the recommendation for NVIDIA?"))

agent_memory = pxt.get_table("financial_analyst.memory")
print(agent_memory.select(agent_memory.content).collect())
