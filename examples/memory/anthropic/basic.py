from pixelagent.anthropic import Agent
import pixeltable as pxt

# Create an agent with tools
agent = Agent(
    agent_name="anthropic_bot",
    system_prompt="You’re my assistant.",
    reset=True
)

# Test chat and tool_call functionality
print(agent.chat("Hi, how are you?"))
print(agent.chat("What was my last question?"))