# Building an OpenAI Agent in Pixeltable: A Step-by-Step Guide

Learn how to build a persistent, tool-calling `Agent` with conversational memory using Pixeltable's data orchestration—powered by OpenAI's models. We'll start with a basic chat agent and then add tool-calling capabilities.

## Prerequisites
- Install required packages:
  ```bash
  pip install pixeltable openai
  ```
- Set up your OpenAI API key:
  ```bash
  export OPENAI_API_KEY='your-api-key'
  ```

## Step 1: Set Up the Agent Basics

First, let's create a basic `Agent` class with a name, system prompt, and model:

```python
from datetime import datetime
import pixeltable as pxt

class Agent:
    def __init__(self, agent_name: str, system_prompt: str, model: str = "gpt-4o-mini"):
        self.directory = agent_name  # Where data lives
        self.system_prompt = system_prompt  # Agent's behavior guide
        self.model = model  # OpenAI model

        # Set up a Pixeltable directory
        pxt.create_dir(self.directory, if_exists="ignore")
```

- **`agent_name`**: A unique ID (e.g., "my_agent") for storing the agent's data in Pixeltable.
- **`system_prompt`**: Defines the agent's behavior (e.g., "You're a helpful assistant").
- **`model`**: The OpenAI model to use, defaulting to "gpt-4o-mini".
- **`pxt.create_dir`**: Creates a storage space for our agent's data.

## Step 2: Add Memory for Chat History

Next, let's add a memory table to store the conversation history:

```python
from datetime import datetime
import pixeltable as pxt

class Agent:
    def __init__(self, agent_name: str, system_prompt: str, model: str = "gpt-4o-mini"):
        self.directory = agent_name
        self.system_prompt = system_prompt
        self.model = model
        pxt.create_dir(self.directory, if_exists="ignore")
        
        # Memory table for chat history
        self.memory = pxt.create_table(
            f"{self.directory}.memory",
            {"role": pxt.String, "content": pxt.String, "timestamp": pxt.Timestamp},
            if_exists="ignore"
        )
```

- **`memory` table**: Stores chat history with:
  - `role`: Who sent the message ("user" or "assistant")
  - `content`: The actual message text
  - `timestamp`: When the message was sent
- Pixeltable ensures this data persists automatically—no extra database setup needed.

## Step 3: Build the Chat Pipeline

Now let's create the chat functionality by adding an `agent` table to handle user messages and generate responses:

```python
from datetime import datetime
import pixeltable as pxt
from pixeltable.functions.openai import chat_completions

class Agent:
    def __init__(self, agent_name: str, system_prompt: str, model: str = "gpt-4o-mini"):
        self.directory = agent_name
        self.system_prompt = system_prompt
        self.model = model
        pxt.create_dir(self.directory, if_exists="ignore")
        
        # Set up tables
        self.memory = pxt.create_table(
            f"{self.directory}.memory",
            {"role": pxt.String, "content": pxt.String, "timestamp": pxt.Timestamp},
            if_exists="ignore"
        )
        self.agent = pxt.create_table(
            f"{self.directory}.agent",
            {"user_message": pxt.String, "timestamp": pxt.Timestamp},
            if_exists="ignore"
        )
        
        # Chat pipeline
        self.agent.add_computed_column(
            response=chat_completions(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.agent.user_message}
                ],
                model=self.model
            )
        )
        self.agent.add_computed_column(
            agent_response=self.agent.response.choices[0].message.content  # Extract response text
        )

    def chat(self, message: str) -> str:
        now = datetime.now()
        # Save user message to memory
        self.memory.insert([{"role": "user", "content": message, "timestamp": now}])
        # Process through agent table
        self.agent.insert([{"user_message": message, "timestamp": now}])
        # Fetch response
        result = self.agent.select(self.agent.agent_response).where(self.agent.user_message == message).collect()
        response = result["agent_response"][0]
        # Save response to memory
        self.memory.insert([{"role": "assistant", "content": response, "timestamp": now}])
        return response
```

What's happening here:
- We've added an `agent` table with:
  - `user_message`: The input from the user
  - `timestamp`: When the message was received
- **Chat pipeline**:
  - `response`: Uses OpenAI's `chat_completions` to send the system prompt and user message to the API
  - `agent_response`: Extracts the generated text from the API response
- **`chat` method**:
  1. Saves the user's message to memory
  2. Processes the message through the agent table
  3. Extracts the response
  4. Saves the response to memory
  5. Returns the response to the user

### Test the Chat Agent

```python
agent = Agent("chatbot", "You're a friendly assistant.")
print(agent.chat("Hi!"))  # "Hello! How can I help you today?"
```

## Step 4: Enhanced Agent with Tool-Calling and Context

Now let's build an enhanced version with tool-calling capabilities, context management, and improved configuration:

```python
from datetime import datetime
from typing import Optional
import uuid

import pixeltable as pxt

try:
    import openai
    from pixeltable.functions.openai import chat_completions, invoke_tools
except ImportError:
    raise ImportError("openai not found, run `pip install openai`")


# Build final prompt with tool results
@pxt.udf
def create_tool_prompt(question: str, tool_outputs: list[dict]) -> str:
    return f"QUESTION:\n{question}\n\n RESULTS:\n{tool_outputs}"


# Prompt builder
@pxt.udf
def create_messages(
    past_context: list[dict], current_message: str, system_prompt: str
) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(
        {"role": msg["role"], "content": msg["content"]} for msg in past_context
    )
    messages.append({"role": "user", "content": current_message})
    return messages


class Agent:
    def __init__(
        self,
        agent_name: str,
        system_prompt: str,
        model: str = "gpt-4o-mini",
        n_latest_messages: int = 10,
        tools: Optional[pxt.tools] = None,
        reset: bool = False,
        chat_kwargs: Optional[dict] = None,
        tool_kwargs: Optional[dict] = None,
    ):
        """
        Initialize the Agent with conversational and tool-calling capabilities using OpenAI's API.

        Args:
            agent_name: Unique name for the agent and its directory
            system_prompt: Instructions for the agent's behavior
            model: OpenAI model to use (default: 'gpt-4o-mini')
            n_latest_messages: Number of recent messages to include in context (default: 10)
            tools: Optional Pixeltable tools object for tool-calling
            reset: Whether to drop and recreate the directory (default: False)
            chat_kwargs: Optional dict of kwargs for OpenAI chat API in chat mode
            tool_kwargs: Optional dict of kwargs for OpenAI chat API in tool-calling mode
        """
        self.directory = agent_name
        self.system_prompt = system_prompt
        self.model = model
        self.n_latest_messages = n_latest_messages
        self.tools = tools
        self.chat_kwargs = chat_kwargs or {}
        self.tool_kwargs = tool_kwargs or {}

        if reset:
            pxt.drop_dir(self.directory, force=True)
        pxt.create_dir(self.directory, if_exists="ignore")

        self._setup_tables()

        self.memory = pxt.get_table(f"{self.directory}.memory")
        self.agent = pxt.get_table(f"{self.directory}.agent")
        self.tools_table = (
            pxt.get_table(f"{self.directory}.tools") if self.tools else None
        )

    def _setup_tables(self):
        # Memory table
        self.memory = pxt.create_table(
            f"{self.directory}.memory",
            {
                "uuid": pxt.String,
                "role": pxt.String,
                "content": pxt.String,
                "timestamp": pxt.Timestamp,
            },
            if_exists="ignore",
        )

        # Agent table
        self.agent = pxt.create_table(
            f"{self.directory}.agent",
            {
                "uuid": pxt.String,
                "user_message": pxt.String,
                "timestamp": pxt.Timestamp,
                "system_prompt": pxt.String,
            },
            if_exists="ignore",
        )

        if self.tools:
            self.tools_table = pxt.create_table(
                f"{self.directory}.tools",
                {
                    "uuid": pxt.String,
                    "tool_prompt": pxt.String,
                    "timestamp": pxt.Timestamp,
                },
                if_exists="ignore",
            )
            self._setup_tools_pipeline()

        self._setup_chat_pipeline()

    def _setup_chat_pipeline(self):
        """Setup the chat pipeline for OpenAI."""

        # Recent memory query
        @pxt.query
        def get_recent_memory(current_timestamp: pxt.Timestamp) -> list[dict]:
            return (
                self.memory.where(self.memory.timestamp < current_timestamp)
                .order_by(self.memory.timestamp, asc=False)
                .select(role=self.memory.role, content=self.memory.content)
                .limit(self.n_latest_messages)
            )

        # Chat pipeline
        self.agent.add_computed_column(
            memory_context=get_recent_memory(self.agent.timestamp),
            if_exists="ignore",
        )
        self.agent.add_computed_column(
            prompt=create_messages(
                self.agent.memory_context,
                self.agent.user_message,
                self.agent.system_prompt,
            ),
            if_exists="ignore",
        )
        self.agent.add_computed_column(
            response=chat_completions(
                messages=self.agent.prompt,
                model=self.model,
                **self.chat_kwargs
            ),
            if_exists="ignore",
        )
        self.agent.add_computed_column(
            agent_response=self.agent.response.choices[0].message.content,
            if_exists="ignore",
        )

    def _setup_tools_pipeline(self):
        """Setup the tool-calling pipeline for OpenAI."""
        # Initial response with tool call
        messages = [{"role": "user", "content": self.tools_table.tool_prompt}]
        self.tools_table.add_computed_column(
            initial_response=chat_completions(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice=self.tools.choice(required=True),
                **self.tool_kwargs
            ),
            if_exists="ignore",
        )

        # Extract tool input and invoke tool
        self.tools_table.add_computed_column(
            tool_output=invoke_tools(self.tools, self.tools_table.initial_response),
            if_exists="ignore",
        )
        
        # Format tool results for final response
        self.tools_table.add_computed_column(
            tool_response_prompt=create_tool_prompt(
                self.tools_table.tool_prompt, self.tools_table.tool_output
            ),
            if_exists="ignore",
        )
        
        # Generate final response based on tool results
        final_messages = [
            {"role": "user", "content": self.tools_table.tool_response_prompt},
        ]
        self.tools_table.add_computed_column(
            final_response=chat_completions(
                model=self.model,
                messages=final_messages,
                **self.tool_kwargs
            ),
            if_exists="ignore",
        )
        self.tools_table.add_computed_column(
            tool_answer=self.tools_table.final_response.choices[0].message.content,
            if_exists="ignore",
        )

    def chat(self, message: str) -> str:
        """
        Process a user message and return the agent's response.

        Args:
            message: The user's input message

        Returns:
            The agent's response as a string
        """
        now = datetime.now()
        generated_uuid = str(uuid.uuid4())

        self.memory.insert(
            [
                {
                    "uuid": generated_uuid,
                    "role": "user",
                    "content": message,
                    "timestamp": now,
                }
            ]
        )
        self.agent.insert(
            [
                {
                    "uuid": generated_uuid,
                    "user_message": message,
                    "timestamp": now,
                    "system_prompt": self.system_prompt,
                }
            ]
        )

        result = (
            self.agent.select(self.agent.agent_response)
            .where(self.agent.uuid == generated_uuid)
            .collect()
        )
        response = result["agent_response"][0]

        self.memory.insert(
            [
                {
                    "uuid": str(uuid.uuid4()),
                    "role": "assistant",
                    "content": response,
                    "timestamp": now,
                }
            ]
        )
        return response

    def tool_call(self, prompt: str) -> str:
        """
        Execute a tool call based on the user's prompt and store in memory.

        Args:
            prompt: The user's tool-related prompt

        Returns:
            The final answer after tool execution
        """
        if not self.tools:
            return "No tools configured for this agent."

        now = datetime.now()
        generated_uuid = str(uuid.uuid4())  # Generate a unique UUID

        self.memory.insert(
            [
                {
                    "uuid": generated_uuid,
                    "role": "user",
                    "content": prompt,
                    "timestamp": now,
                }
            ]
        )
        self.tools_table.insert(
            [
                {
                    "uuid": generated_uuid,
                    "tool_prompt": prompt,
                    "timestamp": now,
                }
            ]
        )

        result = (
            self.tools_table.select(self.tools_table.tool_answer)
            .where(self.tools_table.uuid == generated_uuid)
            .collect()
        )
        tool_answer = result["tool_answer"][0]

        self.memory.insert(
            [
                {
                    "uuid": str(uuid.uuid4()),
                    "role": "assistant",
                    "content": tool_answer,
                    "timestamp": now,
                }
            ]
        )
        return tool_answer
```

## New Features and Improvements

The updated Agent class includes several significant improvements:

1. **UUID-based tracking**:
   - Each message now has a unique identifier through the `uuid` column
   - This enables reliable message tracking even with concurrent requests
   - The `.where(uuid == generated_uuid)` pattern ensures we retrieve the correct result

2. **Better code organization**:
   - Class structure refactored with private methods (`_setup_tables`, `_setup_chat_pipeline`, `_setup_tools_pipeline`)
   - More detailed docstrings and function annotations
   - Clear separation of concerns between tables, pipelines, and API interactions

3. **Enhanced configurability**:
   - `n_latest_messages` can now be configured as a parameter (previously hardcoded)
   - New `reset` parameter for dropping and recreating tables if needed
   - `chat_kwargs` and `tool_kwargs` dictionaries allow for passing additional parameters to the OpenAI API
   - `if_exists="ignore"` added to computed columns to avoid errors when adding columns multiple times

4. **Improved message handling**:
   - Changed order to get the most recent messages first with `asc=False`
   - System prompt is stored with each message in the agent table for flexibility
   - Added `create_messages` UDF to properly build the message array with context

5. **Better tool result processing**:
   - Added `create_tool_prompt` UDF to format tool outputs for better final responses
   - Two-step process: first get tool output, then format and summarize results

### Test the Enhanced Agent

```python
# Define a tool
@pxt.udf
def say_hello(name: str) -> str:
    return f"Hello, {name}!"

# Create agent with tools and custom API parameters
tools = pxt.tools(say_hello)
agent = Agent(
    "toolbot", 
    "You're a helpful assistant.", 
    tools=tools,
    n_latest_messages=5,  # Smaller context window
    chat_kwargs={"temperature": 0.7},  # More creative in chat
    tool_kwargs={"temperature": 0.2}  # More deterministic with tools
)

# Test chat and tools
print(agent.chat("Hi!"))  # "Hello! How can I help you today?"
print(agent.tool_call("Say hello to Alice"))  # "Hello, Alice!"
print(agent.chat("What was that last result?"))  # Remembers tool result via memory
```

## Why This Works

- **Modular Code**: Improved class design makes extending functionality easier
- **Configuration Options**: Fine-tune agent behavior without changing code
- **Efficient Context Management**: Control memory usage by adjusting `n_latest_messages`
- **Separate API Parameters**: Optimize different aspects of the agent with `chat_kwargs` and `tool_kwargs`
- **UUID Tracking**: Ensure reliable results even with concurrent operations

## Why Pixeltable Rocks for Agents

- **Persistence**: Memory and results are automatically saved without additional code
- **Orchestration**: UDFs and data pipelines handle complexity elegantly
- **Efficiency**: Token-saving tool calls and incremental updates keep it fast and cost-effective
- **Error Handling**: Better error recovery with `if_exists="ignore"` for computed columns

## Next Steps

- Add more tools to `pxt.tools()` for richer functionality
- Adjust `n_latest_messages` for different context sizes based on your needs
- Customize `create_tool_prompt` for fancier outputs or better presentation
- Use the new `reset=True` parameter to quickly iterate on agent designs

You've built a powerful agent with minimal code! All the code is here—ready to tweak and scale up for your specific use cases. Happy coding!