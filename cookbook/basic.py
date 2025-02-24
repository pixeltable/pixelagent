from pixelagent.openai import Agent

agent = Agent(
    name="writer",
    system_prompt="You are a brilliant writer.",
    model="gpt-4o-mini",
)

result = agent.run("What is the capital of France?")
print(result)
