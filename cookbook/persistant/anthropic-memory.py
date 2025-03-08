from pixelagent.anthropic import Agent

agent = Agent(
    name="anthropic_persistent_agent",
    system_prompt="You are a helpful assistant.",
    model="claude-3-7-sonnet-latest",
    reset=True
)

agent.run("What is the capital of France?")

agent.run("My name is John Doe.")

agent.run("What is my name?")

agent.run("I went to the store yesterday.")

agent.run("What did I do yesterday?")

res = agent.run("What was my first question?")

print(res)
