import pixeltable as pxt

from pixelagent.openai import Agent

# Create an agent with tools
agent = Agent(
    agent_name="openai_agent", system_prompt="You’re my assistant.", reset=True
)

# Persistant chat and memory
print(agent.chat("Hi, how are you?"))
print(agent.chat("What was my last question?"))

# Easily access agent memory
memory = pxt.get_table("openai_agent.memory")
print(memory.collect())
