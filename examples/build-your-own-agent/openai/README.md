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

## Step 4: Add Tool-Calling Capabilities

Now let's extend our agent to support tool-calling by adding a `tools` parameter and a dedicated `tools_table`:

```python
from datetime import datetime
from typing import Optional
import pixeltable as pxt
from pixeltable.functions.openai import chat_completions, invoke_tools

class Agent:
    def __init__(self, agent_name: str, system_prompt: str, model: str = "gpt-4o-mini", tools=None):
        self.directory = agent_name
        self.system_prompt = system_prompt
        self.model = model
        self.tools = tools  # Optional tools
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
        
        # Tools table (if tools provided)
        if self.tools:
            self.tools_table = pxt.create_table(
                f"{self.directory}.tools",
                {"tool_prompt": pxt.String, "timestamp": pxt.Timestamp},
                if_exists="ignore"
            )
            # Tools pipeline (no chat history sent!)
            self.tools_table.add_computed_column(
                initial_response=chat_completions(
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": self.tools_table.tool_prompt}
                    ],
                    model=self.model,
                    tools=self.tools,
                    tool_choice=self.tools.choice(required=True)
                )
            )
            self.tools_table.add_computed_column(
                tool_output=invoke_tools(self.tools, self.tools_table.initial_response)
            )
            self.tools_table.add_computed_column(
                tool_answer=self.tools_table.tool_output  # Simplified output
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
            agent_response=self.agent.response.choices[0].message.content
        )

    def chat(self, message: str) -> str:
        now = datetime.now()
        self.memory.insert([{"role": "user", "content": message, "timestamp": now}])
        self.agent.insert([{"user_message": message, "timestamp": now}])
        result = self.agent.select(self.agent.agent_response).where(self.agent.user_message == message).collect()
        response = result["agent_response"][0]
        self.memory.insert([{"role": "assistant", "content": response, "timestamp": now}])
        return response

    def tool_call(self, prompt: str) -> str:
        if not self.tools:
            return "No tools configured."
        now = datetime.now()
        # Save prompt to memory
        self.memory.insert([{"role": "user", "content": prompt, "timestamp": now}])
        # Process tool call
        self.tools_table.insert([{"tool_prompt": prompt, "timestamp": now}])
        result = self.tools_table.select(self.tools_table.tool_answer).where(self.tools_table.tool_prompt == prompt).collect()
        tool_answer = result["tool_answer"][0]
        # Save answer to memory
        self.memory.insert([{"role": "assistant", "content": tool_answer, "timestamp": now}])
        return tool_answer
```

Key additions in this step:
- **Tools parameter**: The `__init__` method now accepts an optional `tools` parameter.
- **Tools table**: Created only if tools are provided.
- **Tools pipeline**:
  1. `initial_response`: Sends only the system prompt and tool prompt (no chat history) to OpenAI with `tool_choice` set to require using a tool.
  2. `tool_output`: Uses `invoke_tools` to execute the tool based on OpenAI's response.
  3. `tool_answer`: Captures the tool result for the user.
- **`tool_call` method**: Processes tool calls while keeping memory in sync.

Note that we don't send chat history with tool calls to save on tokens, since OpenAI charges based on token usage. The memory table still keeps track of the full conversation.

### Test the Tool-Calling

```python
# Define a tool
@pxt.udf
def say_hello(name: str) -> str:
    return f"Hello, {name}!"

# Create agent with tools
tools = pxt.tools(say_hello)
agent = Agent("toolbot", "You're a helpful assistant.", tools=tools)

# Test chat and tools
print(agent.chat("Hi!"))  # "Hello! How can I help you today?"
print(agent.tool_call("Say hello to Alice"))  # "Hello, Alice!"
```

## Why This Works

- **Chat**: The agent table and chat method handle conversations, with memory tracking history.
- **Tools**: The tools table and tool_call method enable tool use, optimized for token usage.
- **Pixeltable Magic**: Automates data flow with tables and computed columns.

## Why Pixeltable Rocks for Agents

- **Persistence**: Chat and tool data are automatically saved in tables. No manual database work—Pixeltable keeps everything durable and queryable.
- **Data Orchestration**: It manages tool-call handshakes (e.g., `invoke_tools`) and async OpenAI calls. Pipelines flow effortlessly.
- **Token Efficiency**: By skipping chat history in tool-calling while still syncing via memory, you save on token costs.
- **Incremental Updates**: `.insert` triggers event-driven processing—add data, and only the new stuff updates. Fast, efficient, and scalable.

## Next Steps

- Add context window management by limiting memory to recent messages
- Create more sophisticated tools and chain them together
- Add better formatting for tool results
- Build specialized agents for different domains

You've built a powerful OpenAI-powered agent with minimal code! All the code is here—ready to tweak and scale up for your specific use cases. Happy building!