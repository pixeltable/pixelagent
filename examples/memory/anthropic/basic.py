import pixeltable as pxt

from pixelagent.anthropic import Agent

# Create an agent
agent = Agent(
    name="anthropic_agent", system_prompt="You’re my assistant.", reset=True
)

# Persistant chat and memory
print(agent.chat("Hi, how are you?"))
print(agent.chat("What was my last question?"))

# Easily access agent memory
memory = pxt.get_table("anthropic_agent.memory")
print(memory.collect())
