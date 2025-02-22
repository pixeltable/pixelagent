from pixelagent.openai import Agent

agent = Agent(
    name="writer",
    system_prompt="You are a brilliant writer.",
)

result = agent.run("Write a haiku about dragons")
print(result)
