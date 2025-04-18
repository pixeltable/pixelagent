import pixeltable as pxt
from agent import Agent


@pxt.udf
def weather(city: str) -> str:
    """
    Returns the weather in a given city.
    """
    return f"The weather in {city} is sunny."


# Create an agent
agent = Agent(
    name="openai_agent",
    system_prompt="You’re my assistant.",
    tools=pxt.tools(weather),
    reset=True,
)

# Persistant chat and memory
print(agent.chat("Hi, how are you?"))
print(agent.chat("What was my last question?"))

# Tool call
print(agent.tool_call("Get weather in San Francisco"))
