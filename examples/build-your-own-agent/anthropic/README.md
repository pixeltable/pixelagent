
---

### How to Build an Anthropic Agent in Pixeltable: A Step-by-Step Guide

Here’s how you can create an `Agent` class with Pixeltable that can chat with users and call tools. We’ll keep it minimal—focusing on the essentials—and show you how to extend it later. Let’s dive in!

#### Step 1: Set Up the Basics
Start with a simple class structure. The `Agent` needs a name (to organize its data), a system prompt (to guide its behavior), and a model (we’ll use Anthropic’s Claude by default).

```python
from datetime import datetime
import pixeltable as pxt

class Agent:
    def __init__(self, agent_name: str, system_prompt: str, model: str = "claude-3-5-sonnet-latest"):
        self.directory = agent_name  # Where data will live
        self.system_prompt = system_prompt  # How the agent should behave
        self.model = model  # The AI model to use

        # Create a directory in Pixeltable for this agent
        pxt.create_dir(self.directory, if_exists="ignore")
```

- **What’s happening?**
  - `agent_name` becomes a directory in Pixeltable to store all the agent’s data.
  - `system_prompt` tells the agent its personality or purpose (e.g., "You’re a helpful assistant").
  - Pixeltable’s `create_dir` sets up a space for our tables. The `if_exists="ignore"` means it won’t throw an error if the directory already exists.

---

#### Step 2: Add Memory for Chat History
For the agent to chat, it needs to remember past messages. Let’s create a memory table to store them.

```python
class Agent:
    def __init__(self, agent_name: str, system_prompt: str, model: str = "claude-3-5-sonnet-latest"):
        self.directory = agent_name
        self.system_prompt = system_prompt
        self.model = model
        pxt.create_dir(self.directory, if_exists="ignore")
        
        # Set up the memory table
        self.memory = pxt.create_table(
            f"{self.directory}.memory",
            {"role": pxt.String, "content": pxt.String, "timestamp": pxt.Timestamp},
            if_exists="ignore"
        )
```

- **What’s happening?**
  - We create a table called `memory` with columns: `role` (e.g., "user" or "assistant"), `content` (the message text), and `timestamp` (when it happened).
  - Pixeltable handles persistence—messages are saved automatically.

---

#### Step 3: Build the Chat Pipeline
Now, let’s make the agent respond to messages. We’ll create an `agent` table to process user input and generate responses.

```python
from pixeltable.functions.anthropic import messages

class Agent:
    def __init__(self, agent_name: str, system_prompt: str, model: str = "claude-3-5-sonnet-latest"):
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
        
        # Set up the chat pipeline
        self.agent.add_computed_column(
            response=messages(
                messages=[{"role": "user", "content": self.agent.user_message}],
                model=self.model,
                system=self.system_prompt
            )
        )
        self.agent.add_computed_column(
            agent_response=self.agent.response.content[0].text  # Extract the text response
        )

    def chat(self, message: str) -> str:
        now = datetime.now()
        # Save the user’s message to memory
        self.memory.insert([{"role": "user", "content": message, "timestamp": now}])
        # Process the message through the agent table
        self.agent.insert([{"user_message": message, "timestamp": now}])
        # Get the response
        result = self.agent.select(self.agent.agent_response).where(self.agent.user_message == message).collect()
        response = result["agent_response"][0]
        # Save the agent’s response to memory
        self.memory.insert([{"role": "assistant", "content": response, "timestamp": now}])
        return response
```

- **What’s happening?**
  - The `agent` table takes a `user_message` and uses Pixeltable’s `messages` function (from Anthropic) to get a response.
  - `add_computed_column` defines the pipeline: input → Anthropic API → response text.
  - The `chat` method:
    1. Saves the user’s message to `memory`.
    2. Inserts it into the `agent` table to get a response.
    3. Saves the response to `memory` and returns it.
  - Pixeltable automates the data flow—super simple!

---

#### Step 4: Add Tool-Calling
Let’s enable the agent to call tools. We’ll add a `tools` table and a method to handle tool requests. Note: we *don’t* send the full chat history to the tool-calling protocol to save on token costs—more on that below.

```python
from pixeltable.functions.anthropic import messages, invoke_tools

class Agent:
    def __init__(self, agent_name: str, system_prompt: str, model: str = "claude-3-5-sonnet-latest", tools=None):
        self.directory = agent_name
        self.system_prompt = system_prompt
        self.model = model
        self.tools = tools  # Optional tools object
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
            # Set up the tools pipeline (no chat history sent here!)
            self.tools_table.add_computed_column(
                response=messages(
                    messages=[{"role": "user", "content": self.tools_table.tool_prompt}],
                    model=self.model,
                    system=self.system_prompt,
                    tools=self.tools
                )
            )
            self.tools_table.add_computed_column(
                tool_output=invoke_tools(self.tools, self.tools_table.response)
            )
            self.tools_table.add_computed_column(
                tool_answer=self.tools_table.tool_output  # Simplified for this example
            )
        
        # Set up the chat pipeline
        self.agent.add_computed_column(
            response=messages(
                messages=[{"role": "user", "content": self.agent.user_message}],
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
        # Save the prompt to memory
        self.memory.insert([{"role": "user", "content": prompt, "timestamp": now}])
        # Process the tool call
        self.tools_table.insert([{"tool_prompt": prompt, "timestamp": now}])
        result = self.tools_table.select(self.tools_table.tool_answer).where(self.tools_table.tool_prompt == prompt).collect()
        tool_answer = result["tool_answer"][0]
        # Save the answer to memory for chat to pick up
        self.memory.insert([{"role": "assistant", "content": tool_answer, "timestamp": now}])
        return tool_answer
```

- **What’s happening?**
  - `tools` is an optional argument (e.g., from `pxt.tools()`).
  - In the `tools_table` pipeline:
    1. `messages` sends only the current `tool_prompt` to Anthropic with tool info—no chat history!
    2. `invoke_tools` executes the tool based on the response.
    3. `tool_answer` holds the result.
  - **Why no chat history?** Tool-calling is a two-step process (prompt → tool execution → response), and LLM APIs charge by token count. Sending the full history doubles the cost for no gain here. Instead, we insert the tool prompt and answer into `memory`, so the `chat` method can use it later.
  - The `tool_call` method saves both the prompt and result to `memory`, keeping the interaction seamless.

---

#### Step 5: Test It Out!
Here’s how to use your new `Agent`:

```python
# Define a simple tool
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

# Chat with it
print(agent.chat("Hi there!"))  # "Hello! How can I assist you today?"
print(agent.tool_call("Say hello to Alice"))  # "Hello, Alice!"
```

---

### Why This Works
- **Chat**: The `agent` table and `chat` method handle basic conversation, with memory keeping the history.
- **Tool-Calling**: The `tools_table` and `tool_call` method let the agent use external functions, orchestrated by Pixeltable.
- **Pixeltable Magic**: Tables and computed columns automate data flow—no manual juggling required!

### Extend It!
This is just the core. Add these as you need:
- **Context**: Limit memory to the last 10 messages (use a query).
- **Formatting**: Pretty-print tool results (like `format_tool_results`).
- **More Tools**: Pass multiple tools to `pxt.tools()`.

---

### Why Pixeltable is the Agent Framework to Build On
Pixeltable isn’t just a tool—it’s the backbone that makes agent-building effortless and powerful. Here’s why it stands out:

- **Persistence**: Your agent’s memory and responses are automatically saved in tables. No need to manage databases or files—Pixeltable keeps everything durable and ready to query, so your agent never forgets (unless you want it to!).

- **Data Orchestration**: Pixeltable handles the messy stuff like tool-call handshakes (e.g., `invoke_tools`) and async API calls (e.g., to Anthropic). It wires up your pipelines—input to output—so you can focus on what your agent does, not how it juggles data. Plus, it skips sending chat history to tool-calling, saving tokens and costs while still syncing everything through memory.

- **Incremental Updates**: With `.insert`, Pixeltable processes new data event-driven style. Add a message or tool call, and the system updates only what’s needed—no re-running everything. It’s fast, efficient, and scales as your agent grows.

Think of Pixeltable as your agent’s operating system: it manages state, flows data, and optimizes costs (like trimming token usage in tool calls). Whether you’re adding memory, tools, or multi-agent teams, Pixeltable has your back.

---

Now you’ve got a blueprint! Build your own agent—tweak the prompt, add tools, or scale it up. Pixeltable handles the heavy lifting, so you can focus on the fun stuff. Happy coding!