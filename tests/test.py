from typing import List

import pixeltable as pxt

from pixelagent.openai import Agent


@pxt.udf
def stock_price(ticker: str) -> List[float]:
    """Get the current stock price for a given ticker symbol."""
    # This is a mock implementation for testing
    prices = {
        "AAPL": 175.34,
        "MSFT": 325.89,
        "GOOGL": 142.56,
        "AMZN": 178.23,
        "NVDA": 131.17,
    }
    return [prices.get(ticker.upper(), 0.0)]


def main():
    # Create a Bedrock agent with memory
    agent = Agent(
        name="openai_test",
        system_prompt="You are a helpful assistant that can answer questions and use tools.",
        model="gpt-4.1-2025-04-14",  # Use the Amazon Nova Pro model
        n_latest_messages=None,  # Unlimited memory to ensure all messages are included
        tools=pxt.tools(stock_price),  # Register the stock_price tool
        reset=True,  # Reset the agent's memory for testing
    )

    print("\n=== Testing Conversational Memory ===\n")
    
    # First conversation turn
    user_message = "Hello, my name is Alice."
    print(f"User: {user_message}")
    response = agent.chat(user_message)
    print(f"Agent: {response}\n")
    
    # Second conversation turn - the agent should remember the user's name
    user_message = "What's my name?"
    print(f"User: {user_message}")
    response = agent.chat(user_message)
    print(f"Agent: {response}\n")
    
    print("\n=== Testing Tool Calling (No Memory) ===\n")
    
    # Tool call - should not use memory from previous conversation
    user_message = "What is the stock price of NVDA today?"
    print(f"User: {user_message}")
    response = agent.tool_call(user_message)
    print(f"Agent: {response}\n")
    
    # Another tool call - should not remember previous tool call
    user_message = "What about stock price of AAPL?"
    print(f"User: {user_message}")
    response = agent.tool_call(user_message)
    print(f"Agent: {response}\n")
    
    print("\n=== Testing Memory After Tool Calls ===\n")
    
    # Regular chat after tool calls - should still remember the user's name
    user_message = "Do you still remember my name?"
    print(f"User: {user_message}")
    response = agent.chat(user_message)
    print(f"Agent: {response}\n")
    
    # Check if the memory contains all the messages
    print("\n=== Checking Memory Contents ===\n")
    memory_contents = agent.memory.select(
        agent.memory.role, 
        agent.memory.content
    ).order_by(agent.memory.timestamp, asc=True).collect()
    
    print("Memory contains the following messages:")
    for i in range(len(memory_contents)):
        role = memory_contents["role"][i]
        content = memory_contents["content"][i]
        print(f"{i+1}. {role}: {content[:50]}...")


if __name__ == "__main__":
    main()
