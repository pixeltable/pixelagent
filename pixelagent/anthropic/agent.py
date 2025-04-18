from typing import Optional

import pixeltable as pxt
import pixeltable.functions as pxtf

from pixelagent.core.base import BaseAgent

from .utils import create_messages

try:
    from pixeltable.functions.anthropic import invoke_tools, messages
except ImportError:
    raise ImportError("anthropic not found; run `pip install anthropic`")


class Agent(BaseAgent):
    """
    Anthropic-specific implementation of the BaseAgent.

    This agent uses Anthropic's Claude API for generating responses and handling tools.
    It inherits common functionality from BaseAgent including:
    - Table setup and management
    - Memory persistence
    - Base chat and tool call implementations

    The agent supports both limited and unlimited conversation history through
    the n_latest_messages parameter.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        model: str = "claude-3-5-sonnet-latest",
        n_latest_messages: Optional[int] = 10,
        tools: Optional[pxt.tools] = None,
        reset: bool = False,
        chat_kwargs: Optional[dict] = None,
        tool_kwargs: Optional[dict] = None,
    ):
        # Initialize the base agent with all common parameters
        super().__init__(
            name=name,
            system_prompt=system_prompt,
            model=model,
            n_latest_messages=n_latest_messages,  # None for unlimited history
            tools=tools,
            reset=reset,
            chat_kwargs=chat_kwargs,
            tool_kwargs=tool_kwargs,
        )

    def _setup_chat_pipeline(self):
        """
        Configure the chat completion pipeline using Pixeltable's computed columns.
        This method implements the abstract method from BaseAgent.

        The pipeline consists of 4 steps:
        1. Retrieve recent messages from memory
        2. Format messages for Claude
        3. Get completion from Anthropic
        4. Extract the response text

        Note: The pipeline automatically handles memory limits based on n_latest_messages.
        When set to None, it maintains unlimited conversation history.
        """

        # Step 1: Define a query to get recent messages
        @pxt.query
        def get_recent_memory(current_timestamp: pxt.Timestamp) -> list[dict]:
            """
            Get recent messages from memory, respecting n_latest_messages limit if set.
            Messages are ordered by timestamp (newest first).
            Returns all messages if n_latest_messages is None.
            """
            query = (
                self.memory.where(self.memory.timestamp < current_timestamp)
                .order_by(self.memory.timestamp, asc=False)
                .select(role=self.memory.role, content=self.memory.content)
            )
            if self.n_latest_messages is not None:
                query = query.limit(self.n_latest_messages)
            return query

        # Step 2: Add computed columns to process the conversation
        # First, get the conversation history
        self.agent.add_computed_column(
            memory_context=get_recent_memory(self.agent.timestamp), if_exists="ignore"
        )

        # Format messages for Claude (simpler than OpenAI as system prompt is passed separately)
        self.agent.add_computed_column(
            messages=create_messages(
                self.agent.memory_context,
                self.agent.user_message,
                self.agent.image,
            ),
            if_exists="ignore",
        )

        # Get Claude's API response (note: system prompt passed directly to messages())
        self.agent.add_computed_column(
            response=messages(
                messages=self.agent.messages,
                model=self.model,
                system=self.system_prompt,  # Claude handles system prompt differently
                **self.chat_kwargs,
            ),
            if_exists="ignore",
        )

        # Extract the final response text from Claude's specific response format
        self.agent.add_computed_column(
            agent_response=self.agent.response.content[0].text, if_exists="ignore"
        )

    def _setup_tools_pipeline(self):
        """
        Configure the tool execution pipeline using Pixeltable's computed columns.
        This method implements the abstract method from BaseAgent.

        The pipeline has 4 stages:
        1. Get initial response from Claude with potential tool calls
        2. Execute any requested tools
        3. Format tool results for follow-up
        4. Get final response incorporating tool outputs

        Note: Claude's tool calling format differs slightly from OpenAI's,
        but the overall flow remains the same thanks to BaseAgent abstraction.
        """
        # Stage 1: Get initial response with potential tool calls
        self.tools_table.add_computed_column(
            initial_response=messages(
                model=self.model,
                system=self.system_prompt,  # Include system prompt for consistent behavior
                messages=[{"role": "user", "content": self.tools_table.tool_prompt}],
                tools=self.tools,  # Pass available tools to Claude
                **self.tool_kwargs,
            ),
            if_exists="ignore",
        )

        # Stage 2: Execute any tools that Claude requested
        self.tools_table.add_computed_column(
            tool_output=invoke_tools(self.tools, self.tools_table.initial_response),
            if_exists="ignore",
        )

        # Stage 3: Format tool results for follow-up
        self.tools_table.add_computed_column(
            tool_response_prompt=pxtf.string.format(
                "{0}: {1}", self.tools_table.tool_prompt, self.tools_table.tool_output
            ),
            if_exists="ignore",
        )

        # Stage 4: Get final response incorporating tool results
        self.tools_table.add_computed_column(
            final_response=messages(
                model=self.model,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": self.tools_table.tool_response_prompt}
                ],
                **self.tool_kwargs,
            ),
            if_exists="ignore",
        )

        # Extract the final response text from Claude's format
        self.tools_table.add_computed_column(
            tool_answer=self.tools_table.final_response.content[0].text,
            if_exists="ignore",
        )
