import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root_dir))

import pixeltable as pxt

from blueprints.multi_provider.minimax.agent import Agent as MiniMaxAgent


@pxt.udf
def weather(city: str) -> str:
    """
    Returns the weather in a given city.
    """
    return f"The weather in {city} is sunny with a high of 75°F."


def main():
    agent = MiniMaxAgent(
        name="minimax_test",
        system_prompt="You are a helpful assistant that can answer questions and use tools.",
        model="MiniMax-M3",
        region="global_en",
        n_latest_messages=10,
        tools=pxt.tools(weather),
        reset=True,
    )

    print("\n=== Testing Conversational Memory ===\n")

    user_message = "Hello, my name is Bob."
    print(f"User: {user_message}")
    response = agent.chat(user_message)
    print(f"Agent: {response}\n")

    user_message = "What's my name?"
    print(f"User: {user_message}")
    response = agent.chat(user_message)
    print(f"Agent: {response}\n")

    print("\n=== Testing Tool Calling ===\n")

    user_message = "What's the weather in Seattle?"
    print(f"User: {user_message}")
    response = agent.tool_call(user_message)
    print(f"Agent: {response}\n")

    user_message = "How about the weather in Chicago?"
    print(f"User: {user_message}")
    response = agent.tool_call(user_message)
    print(f"Agent: {response}\n")

    print("\n=== Testing Memory After Tool Calls ===\n")

    user_message = "Do you still remember my name?"
    print(f"User: {user_message}")
    response = agent.chat(user_message)
    print(f"Agent: {response}\n")


if __name__ == "__main__":
    main()
