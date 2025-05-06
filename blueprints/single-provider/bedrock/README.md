# AWS Bedrock Agent Blueprint

This blueprint demonstrates how to create a conversational AI agent powered by AWS Bedrock models using Pixeltable for persistent memory, storage, orchestration, and tool execution.

## Features

- **Persistent Memory**: Maintains conversation history in a structured database
- **Tool Execution**: Supports function calling with AWS Bedrock models
- **Image Support**: Can process and respond to images
- **Configurable Context Window**: Control how many previous messages are included in the context

## Prerequisites

- Python 3.8+
- Pixeltable (`pip install pixeltable`)
- AWS Boto3 (`pip install boto3`)
- AWS credentials configured (via AWS CLI, environment variables, or IAM role)

## AWS Credentials Setup

Before using this blueprint, ensure you have:

1. An AWS account with access to AWS Bedrock
2. Proper IAM permissions to use Bedrock models
3. AWS credentials configured using one of these methods:
   - AWS CLI: Run `aws configure`
   - Environment variables: Set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_REGION`
   - IAM role (if running on AWS infrastructure)

## Quick Start

```python
from blueprints.single_provider.bedrock.agent import Agent

# Create a simple chat agent
agent = Agent(
    name="my_bedrock_agent",
    system_prompt="You are a helpful assistant.",
    model="amazon.nova-pro-v1:0",  # You can use other Bedrock models like "anthropic.claude-3-sonnet-20240229-v1:0"
    n_latest_messages=10,  # Number of recent messages to include in context
    reset=True  # Start with a fresh conversation history
)

# Chat with the agent
response = agent.chat("Hello, who are you?")
print(response)

# Chat with an image
from PIL import Image
img = Image.open("path/to/image.jpg")
response = agent.chat("What's in this image?", image=img)
print(response)
```

## Tool Execution Example

```python
import pixeltable as pxt

# Define tools
weather_tools = pxt.tools([
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    }
                },
                "required": ["location"]
            }
        }
    }
])

# Define tool implementation
def get_weather(location):
    # In a real application, this would call a weather API
    return f"It's sunny and 72°F in {location}"

# Create agent with tools
agent = Agent(
    name="weather_assistant",
    system_prompt="You are a helpful weather assistant.",
    model="amazon.nova-pro-v1:0",
    tools=weather_tools,
    reset=True
)

# Register tool implementation
weather_tools.register_tool("get_weather", get_weather)

# Use tool calling
response = agent.tool_call("What's the weather like in Seattle?")
print(response)
```

## Available Bedrock Models

This blueprint works with various AWS Bedrock models, including:

- `amazon.nova-pro-v1:0` (default)
- `anthropic.claude-3-sonnet-20240229-v1:0`
- `anthropic.claude-3-opus-20240229-v1:0`
- `anthropic.claude-3-haiku-20240307-v1:0`
- `meta.llama3-70b-instruct-v1:0`
- `meta.llama3-8b-instruct-v1:0`

Note that different models have different capabilities and pricing. Refer to the [AWS Bedrock documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html) for more details.

## Advanced Configuration

You can pass additional parameters to the Bedrock API using the `chat_kwargs` and `tool_kwargs` parameters:

```python
agent = Agent(
    name="advanced_agent",
    system_prompt="You are a helpful assistant.",
    model="amazon.nova-pro-v1:0",
    chat_kwargs={
        "temperature": 0.7,
        "max_tokens": 1000
    },
    tool_kwargs={
        "temperature": 0.2  # Lower temperature for more deterministic tool calls
    }
)
```

## How It Works

The agent uses Pixeltable to create and manage three tables:

1. **memory**: Stores all conversation history with timestamps
2. **agent**: Manages chat interactions and responses
3. **tools**: (Optional) Handles tool execution and responses

When you send a message to the agent, it:

1. Stores your message in the memory table
2. Triggers a pipeline that retrieves recent conversation history
3. Formats the messages for the Bedrock API
4. Gets a response from the Bedrock model
5. Stores the response in memory
6. Returns the response to you

Tool execution follows a similar pattern but includes additional steps to handle the tool calling handshake.
