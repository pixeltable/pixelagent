from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
import time

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    description="you are a helpful assistant",
    add_history_to_messages=True,
    num_history_responses=3,
)
# Run the persistent agent
start_time = time.time()

result = agent.run("hi my name is john")
result = agent.run("i went to mit")
result: RunResponse = agent.run("what is my name")

end_time = time.time()
execution_time = end_time - start_time

print(result.content)
print(f"Total execution time: {execution_time:.2f} seconds")
