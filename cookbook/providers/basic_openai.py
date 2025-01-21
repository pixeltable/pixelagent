from pixelagent import Agent
from pixelagent.llms import OpenAIModel

# Initialize the model and agent
model = OpenAIModel(model_name="gpt-4o-mini")
agent = Agent(model=model, system_prompt="You are a helpful assistant with a thick southern accent.")

# Test it
response = agent.run("Hello")
# Print the full conversation details
print("\nConversation Details:")
print("-" * 50)
print(f"Timestamp: {response.timestamp}")
print(f"\nSystem Prompt: {response.system_prompt}")
print(f"User Message: {response.user_prompt}")
print(f"\nAssistant's Response:\n{response.answer}")
print("-" * 50)