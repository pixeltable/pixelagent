from pixelagent.anthropic import AgentX

agent = AgentX(
    name="writer",
    system_prompt="You are a brilliant writer.",
    model="claude-3-5-haiku-latest",
    reset=True
)

result = agent.execute("What is the capital of France?")
print(result)
