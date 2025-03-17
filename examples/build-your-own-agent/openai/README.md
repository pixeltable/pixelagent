
---

### How to Build an OpenAI Agent in Pixeltable with OpenAI: A Step-by-Step Guide

Here’s how to create an `Agent` class with Pixeltable using OpenAI’s API for chatting and tool-calling. We’ll keep it minimal and clear, giving you a blueprint to build on. Assumes you’ve installed Pixeltable and OpenAI (`pip install pixeltable openai`). Let’s get started!

#### Step 1: Set Up the Basics
Start with a basic class. The `Agent` needs a name, a system prompt, and an OpenAI model (defaulting to `gpt-4o-mini`).

```python
from datetime import datetime
import pixeltable as pxt

class Agent:
    def __init__(self, agent_name: str, system_prompt: str, model: str = "gpt-4o-mini"):
        self.directory = agent_name  # Where data lives
        self.system_prompt = system_prompt  # Agent’s behavior guide
        self.model = model  # OpenAI model

        # Set up a Pixeltable directory
        pxt.create_dir(self.directory, if_exists="ignore")
```

- **What’s happening?**
  - `agent_name` creates a Pixeltable directory for the agent’s data.
  - `system_prompt` defines the agent’s role (e.g., "You’re a helpful assistant").
  - `pxt.create_dir` prepares the storage space, ignoring if it already exists.

---

#### Step 2: Add Memory for Chat History
The agent needs memory to track conversations. Let’s create a `memory` table.

```python
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

- **What’s happening?**
  - The `memory` table stores `role` ("user" or "assistant"), `content` (message text), and `timestamp`.
  - Pixeltable ensures persistence—no extra database setup needed.

---

#### Step 3: Build the Chat Pipeline
Let’s add an `agent` table to handle user messages and generate responses using OpenAI’s `chat_completions`.

```python
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

- **What’s happening?**
  - The `agent` table uses `chat_completions` to send the system prompt and user message to OpenAI.
  - `add_computed_column` builds the pipeline: message → OpenAI → response text.
  - The `chat` method saves the input and output to `memory` and returns the response.
  - Pixeltable ties it all together seamlessly.

---

#### Step 4: Add Tool-Calling
Now, let’s add tool-calling with a `tools` table. We’ll avoid sending chat history to save on token costs, just like before.

```python
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
                tool_answer=self.tool_output  # Simplified output
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

- **What’s happening?**
  - `tools` is optional (e.g., from `pxt.tools()`).
  - The `tools_table` pipeline:
    1. `chat_completions` sends only the `tool_prompt` and system prompt—no chat history—to OpenAI with tools enabled.
    2. `invoke_tools` runs the tool based on OpenAI’s response.
    3. `tool_answer` holds the result (simplified here).
  - **Why no chat history?** Tool-calling hits the LLM API twice (prompt → tool → response), and OpenAI charges by token. Skipping history cuts costs, and we sync it via `memory` for chat to use later.
  - `tool_call` saves both prompt and result to `memory`.

---

#### Step 5: Test It Out!
Try your OpenAI-powered agent:

```python
# Define a tool
@pxt.udf
def say_hello(name: str) -> str:
    return f"Hello, {name}!"

# Create the agent
tools = pxt.tools(say_hello)
agent = Agent(
    agent_name="my_agent",
    system_prompt="You’re a friendly assistant.",
    tools=tools
)

# Test it
print(agent.chat("Hi there!"))  # "Hello! How can I help you today?"
print(agent.tool_call("Say hello to Bob"))  # "Hello, Bob!"
```

---

### Why This Works
- **Chat**: The `agent` table and `chat` method handle conversations, with `memory` tracking history.
- **Tool-Calling**: The `tools_table` and `tool_call` method enable tool use, managed by Pixeltable.
- **Pixeltable Magic**: Automates data flow with tables and computed columns.

### Extend It!
Add these as needed:
- **Context**: Limit `memory` to the last 10 messages with a query.
- **Formatting**: Use `create_tool_prompt` to format tool results.
- **More Tools**: Add multiple tools to `pxt.tools()`.

---

### Why Pixeltable is the Agent Framework to Build On
Pixeltable is your agent-building superpower. Here’s why:

- **Persistence**: Chat and tool data are auto-saved in tables. No manual database work—Pixeltable keeps it all durable and queryable.

- **Data Orchestration**: It manages tool-call handshakes (e.g., `invoke_tools`) and async OpenAI calls. Pipelines flow effortlessly, and it skips chat history in tool-calling to save tokens while syncing via memory.

- **Incremental Updates**: `.insert` triggers event-driven processing—add data, and only the new stuff updates. Fast, efficient, and scalable.

Pixeltable’s your agent’s OS: it handles state, optimizes costs (like token trimming), and frees you to innovate. Build chats, tools, or teams—Pixeltable’s got you covered.

---

Now you’ve got an OpenAI-powered blueprint! Tweak it, add tools, or scale it up. Pixeltable does the heavy lifting—go create something amazing!