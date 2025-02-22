import random

from pixelagent.openai import Agent, tool


@tool
def get_weather(location: str) -> str:
    """Get current temperature for a given location."""
    return str(random.uniform(10, 30))


@tool
def get_air_quality(location: str) -> str:
    """Get current air quality for a given location."""
    return str(random.uniform(10, 30))


tools = [get_weather, get_air_quality]

agent = Agent(
    agent_name="WeatherAgent",
    system_prompt="You are a helpful weather and air quality assistant.",
    tools=tools,
    reset=True,
)
result = agent.run("My name is John Doe")
print(result)
result = agent.run("I went to MIT")
print(result)
result = agent.run("What's the weather in Tokyo?")
print(result)
result = agent.run("Whats my name?")
print(result)
