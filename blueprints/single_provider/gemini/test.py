from dotenv import load_dotenv
import pixeltable as pxt
from agent import Agent

# Load environment variables from .env file
load_dotenv()

@pxt.udf
def weather(city: str) -> str:
    """
    Returns the weather in a given city.
    """
    return f"The weather in {city} is sunny."


# Create an agent
agent = Agent(
    name="gemini_agent",
    system_prompt="You're my assistant.",
    tools=pxt.tools(weather),
    reset=True,
)

# Persistant chat and memory
print("=== FIRST CHAT ===")
response1 = agent.chat("Hi, how are you?")
print(f"Response: {response1}")

# Check what's in memory after first chat
print("\n=== MEMORY CONTENTS AFTER FIRST CHAT ===")
memory_contents = agent.memory.select(agent.memory.role, agent.memory.content, agent.memory.timestamp).order_by(agent.memory.timestamp).collect()
for row in memory_contents:
    print(f"Role: {row['role']}, Content: {row['content']}, Timestamp: {row['timestamp']}")

print("\n=== SECOND CHAT ===")
response2 = agent.chat("What was my last question?")
print(f"Response: {response2}")

# Check memory context that should be retrieved
print("\n=== MEMORY CONTENTS AFTER SECOND CHAT ===")
memory_contents = agent.memory.select(agent.memory.role, agent.memory.content, agent.memory.timestamp).order_by(agent.memory.timestamp).collect()
for row in memory_contents:
    print(f"Role: {row['role']}, Content: {row['content']}, Timestamp: {row['timestamp']}")

# Check what was actually passed to the agent table
print("\n=== AGENT TABLE CONTENTS ===")
agent_contents = agent.agent.select(agent.agent.user_message, agent.agent.memory_context, agent.agent.prompt, agent.agent.agent_response).collect()
for i, row in enumerate(agent_contents):
    print(f"Request {i+1}:")
    print(f"  User Message: {row['user_message']}")
    print(f"  Memory Context: {row['memory_context']}")
    print(f"  Prompt: {row['prompt'][:200]}...")  # First 200 chars
    print(f"  Response: {row['agent_response']}")
    print()

print("\n=== TOOL CALL ===")
tool_response = agent.tool_call("Get weather in San Francisco")
print(f"Tool Response: {tool_response}")
