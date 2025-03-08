from pixelagent.anthropic import Agent

agent = Agent(
    name="writer",
    system_prompt="You are a brilliant writer.",
    model="claude-3-5-haiku-latest",
    reset=True
)

result = agent.run("What is the capital of France?")
print(result)
