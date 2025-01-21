from pixelagent import Agent
from pixelagent import Model

# Initialize the model and agent
model = Model(model_name="gpt-4o")
agent = Agent(model=model, system_prompt="You are a helpful british assistant, with a very thick accent.")

# Test it
response = agent.run("Hello,")
print(response)