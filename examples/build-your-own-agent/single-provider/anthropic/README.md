# Building an Anthropic Agent in Pixeltable: A Step-by-Step Guide

This tutorial walks you through building `agent.py`, a persistent conversational agent with memory using Pixeltable's automated data orchestration and storage—powered by Anthropic's Claude model. We'll build the agent step by step, explaining each component as we go. By the end, you'll have a complete implementation that can handle both chat and tool-calling capabilities.

The final code is available in `agent.py`, but we recommend following the tutorial to understand how each piece fits together.

## Prerequisites
- Install required packages:
  ```bash
  pip install pixeltable anthropic
  ```
- Set up your Anthropic API key:
  ```bash
  export ANTHROPIC_API_KEY='your-api-key'
  ```

## Step 1: Understanding the Building Blocks

The agent is built on three core components that inherit from the shared `BaseAgent` class:

1. **Memory Management**: Using Pixeltable tables to store conversation history with optional message limits
2. **Chat Pipeline**: A series of computed columns that process user input and generate responses
3. **Tool Execution**: Optional support for executing functions/tools during conversations

Key classes and functions:

- `Agent`: Main class that inherits from `BaseAgent` and implements Anthropic-specific logic
- `create_messages()`: UDF that formats conversation history for Claude, with support for images
- Pixeltable tables (inherited from `BaseAgent`):
  - `memory`: Stores all conversation history
  - `agent`: Manages chat interactions
  - `tools`: Handles tool execution (if enabled)

Here's the basic setup with imports and the Agent class initialization:

```python
import base64
import io
from datetime import datetime
from typing import Optional
from uuid import uuid4

import PIL
import pixeltable as pxt
import pixeltable.functions as pxtf

try:
    from pixeltable.functions.anthropic import invoke_tools, messages
except ImportError:
    raise ImportError("anthropic not found; run `pip install anthropic`")

@pxt.udf
def create_messages(memory_context: list[dict], current_message: str, image: Optional[PIL.Image.Image] = None) -> list[dict]:
    """Helper UDF to format conversation history and current message for Claude, with optional image support"""
    messages = memory_context.copy()
    
    # For text-only messages
    if not image:
        messages.append({"role": "user", "content": current_message})
        return messages
        
    # For messages with images
    # Creates content blocks with text and image in base64 format
    content_blocks = [
        {"type": "text", "text": current_message},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg"}}
    ]
    messages.append({"role": "user", "content": content_blocks})
    return messages

class Agent:
    def __init__(
        self,
        agent_name: str,
        system_prompt: str,
        model: str = "claude-3-5-sonnet-latest",
        n_latest_messages: Optional[int] = 10,
        tools: Optional[pxt.tools] = None,
        reset: bool = False,
        chat_kwargs: Optional[dict] = None,
        tool_kwargs: Optional[dict] = None,
    ):
        self.directory = agent_name
        self.system_prompt = system_prompt
        self.model = model
        self.n_latest_messages = n_latest_messages  # None for unlimited history
        self.tools = tools
        self.chat_kwargs = chat_kwargs or {}
        self.tool_kwargs = tool_kwargs or {}
```

## Step 2: Building the Memory and Chat Pipeline

The memory system consists of two main parts inherited from `BaseAgent`:

1. **Table Setup** (`_setup_tables()`):
   - Creates the `memory` table for persistent conversation history
   - Creates the `agent` table for managing chat completions
   - Optionally creates the `tools` table for tool execution
   - Uses Pixeltable's automated data orchestration

2. **Chat Pipeline** (`_setup_chat_pipeline()`):
   - Uses computed columns to create a 4-step processing pipeline:
     1. Retrieves recent messages based on `n_latest_messages` limit (supports unlimited history when set to None)
     2. Formats messages with system prompt for Claude
     3. Gets completion from Anthropic API
     4. Extracts the final response text

Here's how we implement the tables and pipeline:

```python
def _setup_tables(self):
    """Initialize the required Pixeltable tables for the agent"""
    # Create memory table for conversation history
    self.memory = pxt.create_table(
        f"{self.directory}.memory",
        {
            "message_id": pxt.String,   # Unique ID for each message
            "role": pxt.String,         # 'user' or 'assistant'
            "content": pxt.String,      # Message content
            "timestamp": pxt.Timestamp, # When the message was received
        },
        if_exists="ignore",
    )

    # Create agent table for managing chat interactions
    self.agent = pxt.create_table(
        f"{self.directory}.agent",
        {
            "message_id": pxt.String,    # Unique ID for each message
            "user_message": pxt.String,  # User's message content
            "timestamp": pxt.Timestamp,  # When the message was received
            "system_prompt": pxt.String, # System prompt for Claude
            "image": PIL.Image.Image,    # Optional image for multimodal input
        },
        if_exists="ignore",
    )

    # Create tools table if tools are configured
    if self.tools:
        self.tools_table = pxt.create_table(
            f"{self.directory}.tools",
            {
                "tool_invoke_id": pxt.String,  # Unique ID for each tool invocation
                "tool_prompt": pxt.String,     # Tool prompt for Claude
                "timestamp": pxt.Timestamp,    # When the tool was invoked
            },
            if_exists="ignore",
        )
        self._setup_tools_pipeline()

    # Set up chat pipeline
    self._setup_chat_pipeline()

def _setup_chat_pipeline(self):
    """Configure the chat completion pipeline using Pixeltable computed columns"""
    # Get recent messages from memory
    @pxt.query
    def get_recent_memory(current_timestamp: pxt.Timestamp) -> list[dict]:
        """Get recent messages from memory, respecting n_latest_messages limit if set"""
        query = (
            self.memory.where(self.memory.timestamp < current_timestamp)
            .order_by(self.memory.timestamp, asc=False)
            .select(role=self.memory.role, content=self.memory.content)
        )
        if self.n_latest_messages is not None:
            query = query.limit(self.n_latest_messages)
        return query

    # Add computed columns to process chat completion
    self.agent.add_computed_column(
        memory_context=get_recent_memory(self.agent.timestamp),
        if_exists="ignore",
    )

    # Create messages for Claude
    self.agent.add_computed_column(
        messages=create_messages(
            self.agent.memory_context, self.agent.user_message
        ),
        if_exists="ignore",
    )

    # Get Anthropic's API response
    self.agent.add_computed_column(
        api_response=messages(
            messages=self.agent.messages,
            model=self.model,
            system=self.system_prompt,
            **self.chat_kwargs,
        ),
        if_exists="ignore",
    )

    # Parse Claude's response
    self.agent.add_computed_column(
        agent_response=self.agent.api_response.content[0].text,
        if_exists="ignore",
    )
```

## Step 3: Adding Tool-Calling Support

Tool calling is implemented as a three-stage process using Pixeltable's computed columns:

1. **Initial Response**: 
   - Sends the user's request to Claude with available tools
   - Claude decides if and which tools to call

2. **Tool Execution**:
   - `invoke_tools()` executes the requested tools
   - Results are stored in the tools table

3. **Final Response**:
   - Tool results are sent back to Claude
   - Claude generates a final response incorporating tool outputs

Here's the implementation:

```python
def _setup_tools_pipeline(self):
    """Configure the tool call handshake pipeline"""
    # Get initial response from Claude with tool calls
    self.tools_table.add_computed_column(
        initial_response=messages(
            model=self.model,
            system=self.system_prompt,
            messages=[{"role": "user", "content": self.tools_table.tool_prompt}],
            tools=self.tools,
            **self.tool_kwargs,
        ),
        if_exists="ignore",
    )

    # Execute the tools
    self.tools_table.add_computed_column(
        tool_output=invoke_tools(self.tools, self.tools_table.initial_response),
        if_exists="ignore",
    )

    # Pass the tool results back to Claude
    self.tools_table.add_computed_column(
        tool_response_prompt=pxtf.string.format(
            "{0}: {1}", self.tools_table.tool_prompt, self.tools_table.tool_output
        ),
        if_exists="ignore",
    )

    # Get final API response with tool results
    self.tools_table.add_computed_column(
        final_response=messages(
            model=self.model,
            system=self.system_prompt,
            messages=[{"role": "user", "content": self.tools_table.tool_response_prompt}],
            **self.tool_kwargs,
        ),
        if_exists="ignore",
    )

    # Parse Claude's response
    self.tools_table.add_computed_column(
        tool_answer=self.tools_table.final_response.content[0].text,
        if_exists="ignore",
    )
```

## Using the Agent

To use the agent in your code:

1. **Initialize the Agent**:
```python
agent = Agent(
    agent_name="my_agent",
    system_prompt="You are a helpful assistant",
    model="claude-3-5-sonnet-latest",  # Latest Claude model
    n_latest_messages=10,  # None for unlimited history
)
```

2. **Chat with the Agent**:
```python
# Simple text chat
response = agent.chat("Hello, how are you?")

# Chat with an image
from PIL import Image
img = Image.open("path/to/image.jpg")
response = agent.chat("What's in this image?", image=img)

# Chat with tool execution
response = agent.tool_call("What's the weather in New York?")
```

3. **Access History**:
```python
# View recent conversations
memory = pxt.get_table("my_agent.memory")
print(memory.collect())
```

## Key Features

1. **Shared Base Implementation**:
   - Inherits from `BaseAgent` class for common functionality
   - Reduces code duplication with OpenAI agent

2. **Flexible Memory Management**:
   - Optional message limit with `n_latest_messages`
   - Support for unlimited history when set to None
   - Support for multimodal conversations with image handling

3. **Automated Data Management**:
   - Persistent storage using Pixeltable
   - Automatic dependency tracking
   - Efficient query optimization

For more details and advanced usage, refer to the complete implementation in `agent.py`.

## Testing the Agent

You can test this implementation using the provided `test.py` script, which demonstrates both chat and tool-calling capabilities of the agent.