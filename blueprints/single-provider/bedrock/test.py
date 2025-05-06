"""
Test script for the AWS Bedrock Agent Blueprint.

This script demonstrates how to create and use a Bedrock-powered agent
with both basic chat functionality and tool execution.

Prerequisites:
- AWS credentials configured (via AWS CLI, environment variables, or IAM role)
- Access to AWS Bedrock models
"""

import pixeltable as pxt
from agent import Agent

# Step 1: Create a simple chat agent
def test_basic_chat():
    print("\n=== Testing Basic Chat ===\n")
    
    # Initialize the agent
    agent = Agent(
        name="test_bedrock_chat",
        system_prompt="You are a helpful assistant who provides concise responses.",
        model="amazon.nova-pro-v1:0",  # Default Bedrock model
        n_latest_messages=5,  # Keep context window small for testing
        reset=True  # Start fresh
    )
    
    # Test a simple conversation
    messages = [
        "Hello! Who are you?",
        "What can you tell me about AWS Bedrock?",
        "What's the difference between Bedrock and other LLM services?",
    ]
    
    # Send messages and print responses
    for message in messages:
        print(f"User: {message}")
        response = agent.chat(message)
        print(f"Agent: {response}\n")
    
    print("Basic chat test completed successfully!")

# Step 2: Test tool execution
def test_tool_execution():
    print("\n=== Testing Tool Execution ===\n")
    
    # Define a simple calculator tool
    calculator_tools = pxt.tools([
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform a mathematical calculation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["add", "subtract", "multiply", "divide"],
                            "description": "The mathematical operation to perform"
                        },
                        "a": {
                            "type": "number",
                            "description": "The first number"
                        },
                        "b": {
                            "type": "number",
                            "description": "The second number"
                        }
                    },
                    "required": ["operation", "a", "b"]
                }
            }
        }
    ])
    
    # Define the tool implementation
    def calculate(operation, a, b):
        if operation == "add":
            return f"{a} + {b} = {a + b}"
        elif operation == "subtract":
            return f"{a} - {b} = {a - b}"
        elif operation == "multiply":
            return f"{a} * {b} = {a * b}"
        elif operation == "divide":
            if b == 0:
                return "Error: Division by zero"
            return f"{a} / {b} = {a / b}"
        else:
            return f"Unknown operation: {operation}"
    
    # Register the tool implementation
    calculator_tools.register_tool("calculate", calculate)
    
    # Create an agent with the calculator tool
    agent = Agent(
        name="test_bedrock_calculator",
        system_prompt="You are a helpful math assistant. Use the calculate tool to solve math problems.",
        model="amazon.nova-pro-v1:0",
        tools=calculator_tools,
        reset=True
    )
    
    # Test tool execution with different prompts
    prompts = [
        "What is 42 plus 17?",
        "Calculate 123 times 456",
        "What is 1000 divided by 10?",
        "If I have 85 and subtract 37, what do I get?"
    ]
    
    # Send prompts and print responses
    for prompt in prompts:
        print(f"User: {prompt}")
        response = agent.tool_call(prompt)
        print(f"Agent: {response}\n")
    
    print("Tool execution test completed successfully!")

# Step 3: Test conversation memory
def test_conversation_memory():
    print("\n=== Testing Conversation Memory ===\n")
    
    # Initialize the agent with memory
    agent = Agent(
        name="test_bedrock_memory",
        system_prompt="You are a helpful assistant with memory. Reference previous parts of our conversation when appropriate.",
        model="amazon.nova-pro-v1:0",
        n_latest_messages=10,  # Remember 10 previous messages
        reset=True
    )
    
    # Have a conversation that requires memory
    conversation = [
        "Hi, my name is Alex.",
        "I'm working on a project using AWS Bedrock.",
        "The project involves building a chatbot for customer service.",
        "What recommendations do you have for my project?",
        "Can you remember what my name is and what I'm building?",
    ]
    
    # Send messages and print responses
    for message in conversation:
        print(f"User: {message}")
        response = agent.chat(message)
        print(f"Agent: {response}\n")
    
    print("Conversation memory test completed successfully!")

if __name__ == "__main__":
    print("Starting AWS Bedrock Agent Blueprint Tests")
    print("==========================================")
    print("Note: These tests require AWS credentials with Bedrock access")
    
    try:
        # Run the tests
        test_basic_chat()
        test_tool_execution()
        test_conversation_memory()
        
        print("\nAll tests completed successfully!")
        print("You can now use the Bedrock agent in your own applications.")
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        print("\nPossible issues:")
        print("1. AWS credentials not configured correctly")
        print("2. No access to AWS Bedrock models")
        print("3. Network connectivity issues")
        print("\nPlease check your AWS configuration and try again.")
