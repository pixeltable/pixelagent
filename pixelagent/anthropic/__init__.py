from datetime import datetime
from typing import Optional

import pixeltable as pxt

try:
    import anthropic
    from pixeltable.functions.anthropic import invoke_tools, messages
except ImportError:
    raise ImportError("anthropic not found, run `pip install anthropic`")


# Format tool results
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


# Prompt builder (no system prompt here)
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
        model: str = "claude-3-5-sonnet-latest",
        n_latest_messages: int = 10,
        tools: Optional[pxt.tools] = None,
        reset: bool = False,
        chat_kwargs: Optional[dict] = None,  # New: Separate kwargs for chat
        tool_kwargs: Optional[dict] = None,  # New: Separate kwargs for tool calls
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
                .order_by(self.memory.timestamp)
                .select(role=self.memory.role, content=self.memory.content)
                .limit(self.n_latest_messages)
            )

        # Chat pipeline
        self.agent.add_computed_column(
            memory_context=get_recent_memory(self.agent.timestamp)
        )
        self.agent.add_computed_column(
            prompt=create_messages(self.agent.memory_context, self.agent.user_message)
        )
        self.agent.add_computed_column(
            response=messages(
                messages=self.agent.prompt,
                model=self.model,
                system=self.system_prompt,
                **self.chat_kwargs  # Use chat-specific kwargs
            )
        )
        self.agent.add_computed_column(
            agent_response=self.agent.response.content[0].text  # Anthropic response structure
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
            )
        )

        # Extract tool input from response
        self.tools_table.add_computed_column(
            tool_input=self.tools_table.initial_response.content
        )

        # Invoke tools
        self.tools_table.add_computed_column(
            tool_output=invoke_tools(self.tools, self.tools_table.initial_response)
        )

        self.tools_table.add_computed_column(
            formatted_results=format_tool_results(
                self.tools_table.tool_prompt,
                self.tools_table.tool_input,
                self.tools_table.tool_output,
            )
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
            )
        )
        self.tools_table.add_computed_column(
            tool_answer=self.tools_table.final_response.content[0].text
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