# Building an Anthropic Agent in Pixeltable: A Step-by-Step Guide

Build a persistent, conversational agent with memory using Pixeltable's automated data orchestration—powered by Anthropic's Claude model. This guide starts with a basic chat agent that remembers conversations, then progressively adds tool-calling capabilities.

## Prerequisites
- Install required packages:
  ```bash
  pip install pixeltable anthropic
  ```
- Set up your Anthropic API key:
  ```bash
  export ANTHROPIC_API_KEY='your-api-key'
  ```

## Step 1: Set Up the Agent Basics

First, let's create a basic `Agent` class with a name, system prompt, and model:

```python
from datetime import datetime
import pixeltable as pxt

class Agent:
    def __init__(
        self,
        agent_name: str,
        system_prompt: str,
        model: str = "claude-3-5-sonnet-latest"
    ):
        self.directory = agent_name  # Directory for agent's data
        self.system_prompt = system_prompt  # Defines agent's behavior
        self.model = model  # Anthropic model
        self.n_latest_messages = 10  # Context window size
        pxt.create_dir(self.directory, if_exists="ignore")
```

- **`agent_name`**: A unique ID (e.g., "my_agent") for storing data in Pixeltable.
- **`system_prompt`**: Guides the agent's behavior (e.g., "You're a helpful assistant").
- **`n_latest_messages`**: Limits chat context to the last 10 messages.

## Step 2: Build the Chat Agent with Memory

Now we'll add a `memory` table and chat pipeline, introducing an async `create_messages` UDF for efficient message handling:

```python
from datetime import datetime
import pixeltable as pxt
try:
    import anthropic
    from pixeltable.functions.anthropic import messages
except ImportError:
    raise ImportError("anthropic not found, run `pip install anthropic`")

# Async UDF for building chat messages
@pxt.udf
async def create_messages(past_context: list[dict], current_message: str) -> list[dict]:
    messages = [
        {"role": msg["role"], "content": msg["content"]} for msg in past_context
    ]
    messages.append({"role": "user", "content": current_message})
    return messages

class Agent:
    def __init__(
        self,
        agent_name: str,
        system_prompt: str,
        model: str = "claude-3-5-sonnet-latest"
    ):
        self.directory = agent_name
        self.system_prompt = system_prompt
        self.model = model
        self.n_latest_messages = 10
        pxt.create_dir(self.directory, if_exists="ignore")
        
        # Memory table
        self.memory = pxt.create_table(
            f"{self.directory}.memory",
            {"role": pxt.String, "content": pxt.String, "timestamp": pxt.Timestamp},
            if_exists="ignore"
        )
        
        # Agent table for chat
        self.agent = pxt.create_table(
            f"{self.directory}.agent",
            {"user_message": pxt.String, "timestamp": pxt.Timestamp},
            if_exists="ignore"
        )
        
        # Chat pipeline with context
        @pxt.query
        def get_recent_memory(current_timestamp: pxt.Timestamp) -> list[dict]:
            return (
                self.memory.where(self.memory.timestamp < current_timestamp)
                .order_by(self.memory.timestamp)
                .select(role=self.memory.role, content=self.memory.content)
                .limit(self.n_latest_messages)
            )
        
        self.agent.add_computed_column(memory_context=get_recent_memory(self.agent.timestamp))
        self.agent.add_computed_column(
            prompt=create_messages(self.agent.memory_context, self.agent.user_message)
        )
        self.agent.add_computed_column(
            response=messages(
                messages=self.agent.prompt,
                model=self.model,
                system=self.system_prompt
            )
        )
        self.agent.add_computed_column(
            agent_response=self.agent.response.content[0].text  # Extract text
        )
    
    def chat(self, message: str) -> str:
        now = datetime.now()
        self.memory.insert([{"role": "user", "content": message, "timestamp": now}])
        self.agent.insert([{"user_message": message, "timestamp": now}])
        result = self.agent.select(self.agent.agent_response).where(self.agent.user_message == message).collect()
        response = result["agent_response"][0]
        self.memory.insert([{"role": "assistant", "content": response, "timestamp": now}])
        return response
```

Let's break down what we've added:

- **`memory` table**: Stores chat history with columns for `role`, `content`, and `timestamp`.
- **`create_messages`**: An async UDF that builds the message list from context and current input—great for parallel processing.
- **Chat pipeline**:
  1. `get_recent_memory`: A query that fetches the last 10 messages from memory.
  2. `prompt`: Uses `create_messages` to prepare input for the Anthropic API.
  3. `response`: Calls Anthropic's messages API.
  4. `agent_response`: Extracts the text from the response.
- **`chat` method**: Manages message flow and updates memory with both user inputs and agent responses.

### Test the Chat Agent

```python
agent = Agent("chatbot", "You're a friendly assistant.")
print(agent.chat("Hi!"))  # "Hello! How can I help you?"
print(agent.chat("What did I say?"))  # Remembers "Hi!" in context
```

## Step 3: Add Tool-Calling Capabilities

Now that we have a functional chat agent with memory, let's extend it to support tool-calling. We'll need to:
1. Update our `Agent` class to accept tools
2. Add a dedicated tools table and pipeline
3. Create a new method for tool calling

```python
from datetime import datetime
from typing import Optional
import pixeltable as pxt
try:
    import anthropic
    from pixeltable.functions.anthropic import invoke_tools, messages
except ImportError:
    raise ImportError("anthropic not found, run `pip install anthropic`")

# Async UDF for building chat messages
@pxt.udf
async def create_messages(past_context: list[dict], current_message: str) -> list[dict]:
    messages = [
        {"role": msg["role"], "content": msg["content"]} for msg in past_context
    ]
    messages.append({"role": "user", "content": current_message})
    return messages

# Async UDF for formatting tool results
@pxt.udf
async def format_tool_results(
    original_prompt: str, tool_inputs: list[dict], tool_outputs: dict
) -> str:
    result = f"Original prompt: {original_prompt}\n"
    result += "Tool information:\n"
    for tool_name, outputs in tool_outputs.items():
        inputs = [ti for ti in tool_inputs if ti.get("name") == tool_name]
        for i, output in enumerate(outputs):
            if i < len(inputs):
                result += f"Tool: {tool_name}\n"
                result += f"Input: {inputs[i]['input']}\n"
                result += f"Output: {output}\n"
                result += "----\n"
    return result.rstrip()

class Agent:
    def __init__(
        self,
        agent_name: str,
        system_prompt: str,
        model: str = "claude-3-5-sonnet-latest",
        tools: Optional[pxt.tools] = None
    ):
        self.directory = agent_name
        self.system_prompt = system_prompt
        self.model = model
        self.tools = tools  # Now we introduce tools parameter
        self.n_latest_messages = 10
        pxt.create_dir(self.directory, if_exists="ignore")
        
        # Memory table
        self.memory = pxt.create_table(
            f"{self.directory}.memory",
            {"role": pxt.String, "content": pxt.String, "timestamp": pxt.Timestamp},
            if_exists="ignore"
        )
        
        # Agent table for chat
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
            self.tools_table.add_computed_column(
                initial_response=messages(
                    messages=[{"role": "user", "content": self.tools_table.tool_prompt}],
                    model=self.model,
                    system=self.system_prompt,
                    tools=self.tools
                )
            )
            self.tools_table.add_computed_column(
                tool_input=self.tools_table.initial_response.content
            )
            self.tools_table.add_computed_column(
                tool_output=invoke_tools(self.tools, self.tools_table.initial_response)
            )
            self.tools_table.add_computed_column(
                formatted_results=format_tool_results(
                    self.tools_table.tool_prompt,
                    self.tools_table.tool_input,
                    self.tools_table.tool_output
                )
            )
            self.tools_table.add_computed_column(
                final_response=messages(
                    messages=[{"role": "user", "content": self.tools_table.formatted_results}],
                    model=self.model,
                    system=self.system_prompt
                )
            )
            self.tools_table.add_computed_column(
                tool_answer=self.tools_table.final_response.content[0].text
            )
        
        # Chat pipeline with context
        @pxt.query
        def get_recent_memory(current_timestamp: pxt.Timestamp) -> list[dict]:
            return (
                self.memory.where(self.memory.timestamp < current_timestamp)
                .order_by(self.memory.timestamp)
                .select(role=self.memory.role, content=self.memory.content)
                .limit(self.n_latest_messages)
            )
        
        self.agent.add_computed_column(memory_context=get_recent_memory(self.agent.timestamp))
        self.agent.add_computed_column(
            prompt=create_messages(self.agent.memory_context, self.agent.user_message)
        )
        self.agent.add_computed_column(
            response=messages(
                messages=self.agent.prompt,
                model=self.model,
                system=self.system_prompt
            )
        )
        self.agent.add_computed_column(
            agent_response=self.agent.response.content[0].text
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
        self.memory.insert([{"role": "user", "content": prompt, "timestamp": now}])
        self.tools_table.insert([{"tool_prompt": prompt, "timestamp": now}])
        result = self.tools_table.select(self.tools_table.tool_answer).where(self.tools_table.tool_prompt == prompt).collect()
        tool_answer = result["tool_answer"][0]
        self.memory.insert([{"role": "assistant", "content": tool_answer, "timestamp": now}])
        return tool_answer
```

Key additions in this step:

- **New tools parameter**: We've updated the `__init__` method to accept an optional `tools` parameter.
- **`format_tool_results` UDF**: A new async UDF for formatting tool outputs in a readable way.
- **Tools pipeline**:
  1. `initial_response`: Sends only the `tool_prompt` (no chat history) to Anthropic, saving tokens.
  2. `tool_input`: Extracts tool call details from the response.
  3. `tool_output`: Executes the tool with `invoke_tools`.
  4. `formatted_results`: Uses the async UDF to format the output nicely.
  5. `final_response` & `tool_answer`: Gets a polished response from Anthropic.
- **`tool_call` method**: Allows invoking tools while keeping memory in sync.

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
print(agent.chat("Hi!"))  # "Hello! How can I assist you?"
print(agent.tool_call("Say hello to Alice"))  # Formatted output, e.g., "Hello, Alice!"
print(agent.chat("What was that last result?"))  # Remembers tool result via memory
```

## Why This Works

- **Chat**: Uses `create_messages` and a 10-message context window for efficient, relevant responses.
- **Tools**: Optimizes token usage by skipping chat history in tool calls, with `format_tool_results` enhancing outputs.
- **Pixeltable**: Manages persistence and pipelines seamlessly.

## Why Pixeltable Rocks for Agents

- **Persistence**: Memory and results are automatically saved without additional code.
- **Orchestration**: Async UDFs and data pipelines handle complexity elegantly.
- **Efficiency**: Token-saving tool calls and incremental updates keep it fast and cost-effective.

## Next Steps

- Add more tools to `pxt.tools()` for richer functionality.
- Adjust `n_latest_messages` for different context sizes based on your needs.
- Customize `format_tool_results` for fancier outputs or better presentation.

You've built a powerful agent with minimal code! All the code is here—ready to tweak and scale up for your specific use cases. Happy coding!