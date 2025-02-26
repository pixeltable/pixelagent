from pixelagent.openai import AgentX

agent = AgentX(
    name="writer",
    system_prompt="You are a brilliant writer.",
    model="gpt-4o-mini",
    reset=True
)

result = agent.execute("What is the capital of France?")
print(result)
