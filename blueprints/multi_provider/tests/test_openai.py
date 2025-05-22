import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root_dir))

from blueprints.multi_provider.openai.agent import Agent as OpenAIAgent


import pixeltable as pxt

@pxt.udf
def weather(city: str) -> str:
    """
    Returns the weather in a given city.
    """
    return f"The weather in {city} is sunny with a high of 75°F."


def main():
    # Create an OpenAI agent with memory
    agent = OpenAIAgent(
        name="openai_test",
        system_prompt="You are a helpful assistant that can answer questions and use tools.",
        model="gpt-4o-mini",  # Use GPT-4o Mini
        n_latest_messages=10,  # Keep last 10 messages in context
        tools=pxt.tools(weather),  # Register the weather tool
        reset=True,  # Reset the agent's memory for testing
    )

    print("\n=== Testing Conversational Memory ===\n")
    
    # First conversation turn
    user_message = "Hello, my name is Bob."
    print(f"User: {user_message}")
    response = agent.chat(user_message)
    print(f"Agent: {response}\n")
    
    # Second conversation turn - the agent should remember the user's name
    user_message = "What's my name?"
    print(f"User: {user_message}")
    response = agent.chat(user_message)
    print(f"Agent: {response}\n")
    
    print("\n=== Testing Tool Calling ===\n")
    
    # Tool call
    user_message = "What's the weather in Seattle?"
    print(f"User: {user_message}")
    response = agent.tool_call(user_message)
    print(f"Agent: {response}\n")
    
    # Another tool call
    user_message = "How about the weather in Chicago?"
    print(f"User: {user_message}")
    response = agent.tool_call(user_message)
    print(f"Agent: {response}\n")
    
    print("\n=== Testing Memory After Tool Calls ===\n")
    
    # Regular chat after tool calls - should still remember the user's name
    user_message = "Do you still remember my name?"
    print(f"User: {user_message}")
    response = agent.chat(user_message)
    print(f"Agent: {response}\n")


if __name__ == "__main__":
    main()
