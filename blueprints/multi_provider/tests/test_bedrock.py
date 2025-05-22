import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root_dir))

from blueprints.multi_provider.bedrock.agent import Agent as BedrockAgent


import pixeltable as pxt

@pxt.udf
def stock_price(ticker: str) -> float:
    """Get the current stock price for a given ticker symbol."""
    # This is a mock implementation for testing
    prices = {
        "AAPL": 175.34,
        "MSFT": 325.89,
        "GOOGL": 142.56,
        "AMZN": 178.23,
        "NVDA": 131.17,
    }
    return prices.get(ticker.upper(), 0.0)


def main():
    # Create a Bedrock agent with memory
    agent = BedrockAgent(
        name="bedrock_test",
        system_prompt="You are a helpful assistant that can answer questions and use tools.",
        model="amazon.nova-pro-v1:0",  # Use the Amazon Nova Pro model
        n_latest_messages=10,  # Keep last 10 messages in context
        tools=pxt.tools(stock_price),  # Register the stock_price tool
        reset=True,  # Reset the agent's memory for testing
    )

    print("\n=== Testing Conversational Memory ===\n")
    
    # First conversation turn
    user_message = "Hello, my name is Charlie."
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
    user_message = "What is the stock price of NVDA today?"
    print(f"User: {user_message}")
    response = agent.tool_call(user_message)
    print(f"Agent: {response}\n")
    
    # Another tool call
    user_message = "What about AAPL?"
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
