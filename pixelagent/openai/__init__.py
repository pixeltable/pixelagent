from datetime import datetime
from typing import Optional

import pixeltable as pxt

try:
    import openai
    from pixeltable.functions.openai import chat_completions, invoke_tools
except ImportError:
    raise ImportError("openai not found, run `pip install openai`")


# Build final prompt with tool results
@pxt.udf
async def create_tool_prompt(question: str, tool_outputs: list[dict]) -> str:
    return f"QUESTION:\n{question}\n\n RESULTS:\n{tool_outputs}"


# Prompt builder
@pxt.udf
async def create_messages(
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
        chat_kwargs: Optional[dict] = None,  # New: Separate kwargs for chat
        tool_kwargs: Optional[dict] = None,  # New: Separate kwargs for tool calls
    ):
        """
        Initialize the Agent with conversational and tool-calling capabilities.

        Args:
            agent_name: Unique name for the agent and its directory
            system_prompt: Instructions for the agent's behavior
            model: OpenAI model to use (default: 'gpt-4o-mini')
            n_latest_messages: Number of recent messages to include in context (default: 10)
            tools: Optional Pixeltable tools object for tool-calling
            reset: Whether to drop and recreate the directory (default: False)
            chat_kwargs: Optional dict of kwargs for OpenAI chat_completions API in chat mode
            tool_kwargs: Optional dict of kwargs for OpenAI chat_completions API in tool-calling mode
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
            {
                "user_message": pxt.String,
                "timestamp": pxt.Timestamp,
                "system_prompt": pxt.String,
            },
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
        """Setup the chat pipeline."""

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
                **self.chat_kwargs  # Use chat-specific kwargs
            ),
            if_exists="ignore",
        )
        self.agent.add_computed_column(
            agent_response=self.agent.response.choices[0].message.content,
            if_exists="ignore",
        )

    def _setup_tools_pipeline(self):
        """Setup the tool-calling pipeline."""
        messages = [{"role": "user", "content": self.tools_table.tool_prompt}]

        # Initial response with tool call
        self.tools_table.add_computed_column(
            initial_response=chat_completions(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice=self.tools.choice(required=True),
                **self.tool_kwargs  # Use tool-specific kwargs
            ),
            if_exists="ignore",
        )

        # Invoke tools
        self.tools_table.add_computed_column(
            tool_output=invoke_tools(self.tools, self.tools_table.initial_response),
            if_exists="ignore",
        )

        self.tools_table.add_computed_column(
            tool_response_prompt=create_tool_prompt(
                self.tools_table.tool_prompt, self.tools_table.tool_output
            ),
            if_exists="ignore",
        )

        # Final response from LLM
        final_messages = [
            {"role": "user", "content": self.tools_table.tool_response_prompt},
        ]
        self.tools_table.add_computed_column(
            final_response=chat_completions(
                model=self.model,
                messages=final_messages,
                **self.tool_kwargs  # Use tool-specific kwargs
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
        self.memory.insert([{"role": "user", "content": message, "timestamp": now}])
        self.agent.insert(
            [
                {
                    "user_message": message,
                    "timestamp": now,
                    "system_prompt": self.system_prompt,
                }
            ]
        )

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