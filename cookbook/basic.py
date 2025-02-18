from pixelagent.openai import Agent

# Initialize the  agent
helpful_assistant = Agent(
    name = "helpful_assistant",
    system_prompt = "you are a helpful assistant", 
    model = "gpt-4o-mini",
    reset = True
)

result = helpful_assistant.run("hi my name is john")
result = helpful_assistant.run("i went to mit")
result = helpful_assistant.run("what is my name")

print(result)