# AWS Bedrock Agent for Pixelagent

This module implements an AWS Bedrock agent that inherits from the core BaseAgent class. The agent supports:

1. Conversational memory for regular chat interactions
2. Tool calling without conversational memory
3. Integration with AWS Bedrock's Claude models

## Implementation Details

The Bedrock agent implementation consists of three main files:

- `agent.py`: Contains the main Agent class that inherits from BaseAgent
- `utils.py`: Contains utility functions for formatting messages for Bedrock
- `test.py`: Contains a test script to verify the agent's functionality

### Important Implementation Notes

1. The BaseAgent's tool_call method uses separate timestamps for the user message and the assistant's response to ensure proper ordering in the memory table.
2. The create_messages function in utils.py ensures that the conversation always starts with a user message, which is required by Bedrock Claude models.
3. The agent supports unlimited memory (n_latest_messages=None) to maintain the entire conversation history.

### Agent Class

The Agent class implements two abstract methods from BaseAgent:

1. `_setup_chat_pipeline`: Configures the chat completion pipeline with conversational memory
2. `_setup_tools_pipeline`: Configures the tool execution pipeline without conversational memory

### Message Formatting

The `create_messages` function in `utils.py` formats the conversation history and current message for Bedrock Claude models. It ensures:

1. Messages are in the correct format with content as a list of objects
2. The conversation always starts with a user message
3. Images are properly encoded if included

## Usage

```python
from pixelagent.bedrock import Agent

# Create a Bedrock agent
agent = Agent(
    name="my_agent",
    system_prompt="You are a helpful assistant.",
    model="amazon.nova-pro-v1:0",  # Or any other Bedrock model
    n_latest_messages=10,  # Number of recent messages to include in context
    tools=None,  # Optional tools configuration
    reset=False,  # Whether to reset existing agent data
)

# Chat with the agent
response = agent.chat("Hello, my name is Alice.")
print(response)

# Use tools if configured
if agent.tools:
    tool_response = agent.tool_call("What is the stock price of AAPL?")
    print(tool_response)
```

## Testing

Run the test script to verify the agent's functionality:

```bash
python -m pixelagent.bedrock.test
```

The test script verifies:

1. Conversational memory for regular chat
2. Tool calling without conversational memory
3. Memory persistence across different interactions
