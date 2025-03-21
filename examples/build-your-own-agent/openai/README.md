# Building an OpenAI Agent in Pixeltable: A Step-by-Step Guide

This tutorial walks you through building `agent.py`, a persistent conversational agent with memory using Pixeltable's automated data orchestration and storage—powered by OpenAI's GPT models. We'll build the agent step by step, explaining each component as we go. By the end, you'll have a complete implementation that can handle both chat and tool-calling capabilities.

The final code is available in `agent.py`, but we recommend following the tutorial to understand how each piece fits together.

## Prerequisites
- Install required packages:
  ```bash
  pip install pixeltable openai
  ```
- Set up your OpenAI API key:
  ```bash
  export OPENAI_API_KEY='your-api-key'
  ```

## Step 1: Understanding the Building Blocks

The agent is built on three core components:

1. **Memory Management**: Using Pixeltable tables to store conversation history and enable persistent memory across sessions
2. **Chat Pipeline**: A series of computed columns that process user input and generate responses
3. **Tool Execution**: Optional support for executing functions/tools during conversations

Key classes and functions:

- `Agent`: Main class that orchestrates all components
- `create_messages()`: UDF that formats conversation history for the LLM
- Pixeltable tables:
  - `memory`: Stores all conversation history
  - `agent`: Manages chat interactions
  - `tools`: Handles tool execution (if enabled)

The code shows the basic setup with imports and the Agent class initialization:

```python
from datetime import datetime
from typing import Optional
from uuid import uuid4

import pixeltable as pxt
import pixeltable.functions as pxtf

try:
    from pixeltable.functions.openai import chat_completions, invoke_tools
except ImportError:
    raise ImportError("openai not found; run `pip install openai`")

@pxt.udf
def create_messages(
    system_prompt: str, memory_context: list[dict], current_message: str
) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(memory_context.copy())
    messages.append({"role": "user", "content": current_message})
    return messages

class Agent:
    def __init__(
        self,
        agent_name: str,
        system_prompt: str,
        model: str = "gpt-4o-mini",
        n_latest_messages: Optional[int] = 10,
        tools: Optional[pxt.tools] = None,
        reset: bool = False,
        chat_kwargs: Optional[dict] = None,
        tool_kwargs: Optional[dict] = None,
    ):
        self.directory = agent_name
        self.system_prompt = system_prompt
        self.model = model
        self.n_latest_messages = n_latest_messages
        self.tools = tools
        self.chat_kwargs = chat_kwargs or {}
        self.tool_kwargs = tool_kwargs or {}
```

## Step 2: Building the Memory and Chat Pipeline

The memory system consists of two main parts:

1. **Table Setup** (`_setup_tables()`):
   - Creates the `memory` table to store all conversations with timestamps
   - Creates the `agent` table to manage chat completions
   - Optionally creates the `tools` table if tool support is enabled
   - All tables use Pixeltable's automated data orchestration

2. **Chat Pipeline** (`_setup_chat_pipeline()`):
   - Uses computed columns to create a 4-step processing pipeline:
     1. Retrieves recent messages based on `n_latest_messages` limit
     2. Formats messages with system prompt for the LLM
     3. Gets completion from OpenAI API
     4. Extracts the final response text

The implementation shows how Pixeltable's computed columns automatically handle dependencies:

```python
def _setup_tables(self):
    """
    Initialize the required Pixeltable tables for the agent.
    Creates three tables:
    1. memory: Stores conversation history
    2. agent: Manages chat completions
    3. tools: (Optional) Handles tool execution
    """
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
        # Set up tools pipeline
        self._setup_tools_pipeline()

    # Set up chat pipeline
    self._setup_chat_pipeline()

def _setup_chat_pipeline(self):
    """
    Configure the chat completion pipeline using Pixeltable computed columns
    1. Get recent messages from memory
    2. Create messages for GPT
    3. Get GPT's response
    4. Extract the response text
    """
    # Get recent messages from memory, respecting n_latest_messages limit if set
    @pxt.query
    def get_recent_memory(current_timestamp: pxt.Timestamp) -> list[dict]:
        query = (
            self.memory.where(self.memory.timestamp < current_timestamp)
            .order_by(self.memory.timestamp, asc=False)
            .select(role=self.memory.role, content=self.memory.content)
        )
        if self.n_latest_messages is not None:
            query = query.limit(self.n_latest_messages)
        return query

    # Add computed columns to agent table
    # Get recent messages from memory
    self.agent.add_computed_column(
        memory_context=get_recent_memory(self.agent.timestamp),
        if_exists="ignore",
    )
    # Create messages for Openai
    self.agent.add_computed_column(
        prompt=create_messages(
            self.agent.system_prompt,
            self.agent.memory_context,
            self.agent.user_message,
        ),
        if_exists="ignore",
    )
    # Get api response
    self.agent.add_computed_column(
        response=chat_completions(
            messages=self.agent.prompt, model=self.model, **self.chat_kwargs
        ),
        if_exists="ignore",
    )
    # Extract the response text
    self.agent.add_computed_column(
        agent_response=self.agent.response.choices[0].message.content,
        if_exists="ignore",
    )
```

## Step 3: Adding Tool-Calling Support

Tool calling is implemented as a three-stage process:

1. **Initial Response**: 
   - Sends the user's request to the LLM with available tools
   - LLM decides if and which tools to call

2. **Tool Execution**:
   - `invoke_tools()` executes the requested tools
   - Results are stored in the tools table

3. **Final Response**:
   - Tool results are sent back to the LLM
   - LLM generates a final response incorporating tool outputs

The pipeline is implemented using computed columns that automatically handle the tool calling handshake:

```python
def _setup_tools_pipeline(self):
    """
    Configure the tool call handshake pipeline using Pixeltable computed columns
    1. Get initial response from GPT with tool calls
    2. Execute the tools
    3. Pass the tool results back to GPT for final response
    """
    # Get initial response from GPT with tool calls
    self.tools_table.add_computed_column(
        initial_response=chat_completions(
            model=self.model,
            messages=[{"role": "user", "content": self.tools_table.tool_prompt}],
            tools=self.tools,
            **self.tool_kwargs
        ),
        if_exists="ignore",
    )

    # Execute the tools
    self.tools_table.add_computed_column(
        tool_output=invoke_tools(self.tools, self.tools_table.initial_response),
        if_exists="ignore",
    )

    # Pass the tool results back to GPT for final response
    self.tools_table.add_computed_column(
        tool_response_prompt=pxtf.string.format(
            "{0}: {1}", self.tools_table.tool_prompt, self.tools_table.tool_output
        ),
        if_exists="ignore",
    )

    # Get final API response from OpenAI with tool results
    self.tools_table.add_computed_column(
        final_response=chat_completions(
            model=self.model,
            messages=[{"role": "user", "content": self.tools_table.tool_response_prompt}],
            **self.chat_kwargs
        ),
        if_exists="ignore",
    )

    # Parse GPT's response
    self.tools_table.add_computed_column(
        tool_answer=self.tools_table.final_response.choices[0].message.content,
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
    model="gpt-4o-mini",  # or other OpenAI models
    n_latest_messages=10  # None for unlimited history
)
```

2. **Chat with the Agent**:
```python
# Simple chat
response = agent.chat("Hello, how are you?")

# Chat with tool execution
response = agent.tool_call("What's the weather in New York?")
```

3. **Access History**:
```python
# View recent conversations
memory = pxt.get_table("my_agent.memory")
print(memory.collect())
```

For more details and advanced usage, refer to the complete implementation in `agent.py`.

## Testing the Agent

You can test this implementation using the provided `test.py` script, which demonstrates both chat and tool-calling capabilities of the agent.

## Key Features

The implementation includes several key features:

1. **Shared Base Implementation**
   - Inherits common Pydantic fields from BaseAgent
   - Consistent interface across LLM providers
   - Abstract methods enforce proper implementation
   - Reduced code duplication

2. **Flexible Memory Management**
   - Stores all conversation history in Pixeltable
   - Supports both limited and unlimited memory (n_latest_messages=None)
   - Unique message IDs for reliable tracking

3. **Robust Tool Execution**
   - Clean handshake between LLM and tools
   - Structured tool response handling
   - Automatic tool result incorporation

4. **Data Orchestration**
   - Automated pipeline management
   - Persistent storage
   - Efficient query optimization

For more details and advanced usage, refer to the complete implementation in `agent.py`.