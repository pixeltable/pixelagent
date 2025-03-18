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

Now we'll add a `memory` table and chat pipeline, introducing a `create_messages` UDF for efficient message handling:

```python
from datetime import datetime
import pixeltable as pxt
try:
    import anthropic
    from pixeltable.functions.anthropic import messages
except ImportError:
    raise ImportError("anthropic not found, run `pip install anthropic`")

# UDF for building chat messages
@pxt.udf
def create_messages(past_context: list[dict], current_message: str) -> list[dict]:
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
- **`create_messages`**: A UDF that builds the message list from context and current input.
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

# UDF for building chat messages
@pxt.udf
def create_messages(past_context: list[dict], current_message: str) -> list[dict]:
    messages = [
        {"role": msg["role"], "content": msg["content"]} for msg in past_context
    ]
    messages.append({"role": "user", "content": current_message})
    return messages

# UDF for formatting tool results
@pxt.udf
def format_tool_results(
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
        n_latest_messages: int = 10,
        tools: Optional[pxt.tools] = None,
        reset: bool = False,
        chat_kwargs: Optional[dict] = None,
        tool_kwargs: Optional[dict] = None,
    ):
        """
        Initialize the Agent with conversational and tool-calling capabilities using Anthropic's API.

        Args:
            agent_name: Unique name for the agent and its directory
            system_prompt: Instructions for the agent's behavior
            model: Anthropic model to use (default: 'claude-3-5-sonnet-latest')
            n_latest_messages: Number of recent messages to include in context (default: 10)
            tools: Optional Pixeltable tools object for tool-calling
            reset: Whether to drop and recreate the directory (default: False)
            chat_kwargs: Optional dict of kwargs for Anthropic messages API in chat mode
            tool_kwargs: Optional dict of kwargs for Anthropic messages API in tool-calling mode
        """
        self.directory = agent_name
        self.system_prompt = system_prompt
        self.model = model
        self.n_latest_messages = n_latest_messages
        self.tools = tools
        self.chat_kwargs = chat_kwargs or {}  # Default to empty dict if None
        self.tool_kwargs = tool_kwargs or {}  # Default to empty dict if None

        # Setup Pixeltable environment
        if reset:
            pxt.drop_dir(self.directory, force=True)

        pxt.create_dir(self.directory, if_exists="ignore")

        # Initialize tables
        self._setup_tables()

        # Table references
        self.memory = pxt.get_table(f"{self.directory}.memory")
        self.agent = pxt.get_table(f"{self.directory}.agent")
        self.tools_table = (
            pxt.get_table(f"{self.directory}.tools") if self.tools else None
        )
    
    def _setup_tables(self):
        """Setup memory, agent, and tools tables."""
        # Memory table for chat and tool history
        self.memory = pxt.create_table(
            f"{self.directory}.memory",
            {"role": pxt.String, "content": pxt.String, "timestamp": pxt.Timestamp},
            if_exists="ignore",
        )

        # Agent table for chat pipeline
        self.agent = pxt.create_table(
            f"{self.directory}.agent",
            {"user_message": pxt.String, "timestamp": pxt.Timestamp},
            if_exists="ignore",
        )

        # Tools table for tool-calling (if tools are provided)
        if self.tools:
            self.tools_table = pxt.create_table(
                f"{self.directory}.tools",
                {"tool_prompt": pxt.String, "timestamp": pxt.Timestamp},
                if_exists="ignore",
            )
            self._setup_tools_pipeline()

        # Setup chat pipeline
        self._setup_chat_pipeline()
    
    def _setup_chat_pipeline(self):
        """Setup the chat pipeline for Anthropic."""

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
            if_exists="ignore"
        )
        self.agent.add_computed_column(
            prompt=create_messages(self.agent.memory_context, self.agent.user_message),
            if_exists="ignore"
        )
        self.agent.add_computed_column(
            response=messages(
                messages=self.agent.prompt,
                model=self.model,
                system=self.system_prompt,
                **self.chat_kwargs  # Use chat-specific kwargs
            ),
            if_exists="ignore"
        )
        self.agent.add_computed_column(
            agent_response=self.agent.response.content[0].text,  # Anthropic response structure
            if_exists="ignore"
        )
    
    def _setup_tools_pipeline(self):
        """Setup the tool-calling pipeline for Anthropic."""
        # Initial response with tool call
        self.tools_table.add_computed_column(
            initial_response=messages(
                model=self.model,
                system=self.system_prompt,
                messages=[{"role": "user", "content": self.tools_table.tool_prompt}],
                tools=self.tools,
                **self.tool_kwargs  # Use tool-specific kwargs
            ),
            if_exists="ignore"
        )

        # Extract tool input from response
        self.tools_table.add_computed_column(
            tool_input=self.tools_table.initial_response.content,
            if_exists="ignore"
        )

        # Invoke tools
        self.tools_table.add_computed_column(
            tool_output=invoke_tools(self.tools, self.tools_table.initial_response),
            if_exists="ignore"
        )

        self.tools_table.add_computed_column(
            formatted_results=format_tool_results(
                self.tools_table.tool_prompt,
                self.tools_table.tool_input,
                self.tools_table.tool_output,
            ),
            if_exists="ignore"
        )

        # Final response from LLM
        self.tools_table.add_computed_column(
            final_response=messages(
                model=self.model,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": self.tools_table.formatted_results}
                ],
                **self.tool_kwargs  # Use tool-specific kwargs
            ),
            if_exists="ignore"
        )
        self.tools_table.add_computed_column(
            tool_answer=self.tools_table.final_response.content[0].text,
            if_exists="ignore"
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
        self.memory.insert([{"role": "user", "content": message, "timestamp": now}])
        self.agent.insert([{"user_message": message, "timestamp": now}])

        result = (
            self.agent.select(self.agent.agent_response)
            .where(self.agent.user_message == message)
            .collect()
        )
        response = result["agent_response"][0]

        self.memory.insert(
            [{"role": "assistant", "content": response, "timestamp": now}]
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

        # Insert the tool prompt into memory as a user message
        self.memory.insert([{"role": "user", "content": prompt, "timestamp": now}])

        # Process the tool call
        self.tools_table.insert([{"tool_prompt": prompt, "timestamp": now}])
        result = (
            self.tools_table.select(self.tools_table.tool_answer)
            .where(self.tools_table.tool_prompt == prompt)
            .collect()
        )
        tool_answer = result["tool_answer"][0]

        # Insert the tool answer into memory as an assistant message
        self.memory.insert(
            [{"role": "assistant", "content": tool_answer, "timestamp": now}]
        )

        return tool_answer
```

## New Features and Improvements

The updated Agent class includes several significant improvements:

1. **Better code organization**:
   - Class structure refactored with private methods (`_setup_tables`, `_setup_chat_pipeline`, `_setup_tools_pipeline`)
   - More detailed docstrings and function annotations
   - Clear separation of concerns between tables, pipelines, and API interactions

2. **Enhanced configurability**:
   - `n_latest_messages` can now be configured as a parameter (previously hardcoded to 10)
   - New `reset` parameter for dropping and recreating tables if needed
   - `chat_kwargs` and `tool_kwargs` dictionaries allow for passing additional parameters to Anthropic's API
   - `if_exists="ignore"` added to computed columns to avoid errors when adding columns multiple times

3. **Improved message ordering**:
   - Changed `order_by(self.memory.timestamp)` to `order_by(self.memory.timestamp, asc=False)` to get the most recent messages first
   - This ensures the most recent context is included in the conversation history

4. **Simplified UDFs**:
   - Removed `async` declarations from UDFs for better compatibility and simpler usage
   - This makes the code more straightforward to understand and maintain

5. **Better table management**:
   - Using `pxt.get_table()` to retrieve table references after creation
   - Using full table initialization in the `_setup_tables()` method
   - Improved error handling for missing tools

### Test the Tool-Calling

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
    chat_kwargs={"max_tokens": 500},  # Limit token usage in chat
    tool_kwargs={"temperature": 0.2}  # Lower temperature for tool calls
)

# Test chat and tools
print(agent.chat("Hi!"))  # "Hello! How can I assist you?"
print(agent.tool_call("Say hello to Alice"))  # Formatted output, e.g., "Hello, Alice!"
print(agent.chat("What was that last result?"))  # Remembers tool result via memory
```

## Why This Works

- **Modular Code**: Improved class design makes extending functionality easier
- **Configuration Options**: Fine-tune agent behavior without changing code
- **Efficient Context Management**: Control memory usage by adjusting `n_latest_messages`
- **Separate API Parameters**: Optimize different aspects of the agent with `chat_kwargs` and `tool_kwargs`
- **Reversed Message Order**: Getting the most recent messages first ensures the most relevant context is used

## Why Pixeltable Rocks for Agents

- **Persistence**: Memory and results are automatically saved without additional code
- **Orchestration**: UDFs and data pipelines handle complexity elegantly
- **Efficiency**: Token-saving tool calls and incremental updates keep it fast and cost-effective
- **Error Handling**: Better error recovery with `if_exists="ignore"` for computed columns

## Next Steps

- Add more tools to `pxt.tools()` for richer functionality
- Adjust `n_latest_messages` for different context sizes based on your needs
- Customize `format_tool_results` for fancier outputs or better presentation
- Use the new `reset=True` parameter to quickly iterate on agent designs

You've built a powerful agent with minimal code! All the code is here—ready to tweak and scale up for your specific use cases. Happy coding!