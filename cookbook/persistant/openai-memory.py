from pixelagent.openai import Agent

agent = Agent(
    name="openai_persistent_agent",
    system_prompt="You are a helpful assistant.",
    model="gpt-4o-mini",
)

agent.run("What is the capital of France?")

agent.run("My name is John Doe.")

agent.run("What is my name?")

agent.run("I went to the store yesterday.")

agent.run("What did I do yesterday?")

agent.run("What was my first question?")

