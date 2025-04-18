from typing import Optional

import pixeltable as pxt
import pixeltable.functions as pxtf

from pixelagent.core.base import BaseAgent

from .utils import create_messages

try:
    from pixeltable.functions.openai import chat_completions, invoke_tools
except ImportError:
    raise ImportError("openai not found; run `pip install openai`")


class Agent(BaseAgent):
    """
    OpenAI-specific implementation of the BaseAgent.

    This agent uses OpenAI's chat completion API for generating responses and handling tools.
    It inherits common functionality from BaseAgent including:
    - Table setup and management
    - Memory persistence
    - Base chat and tool call implementations
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        model: str = "gpt-4o-mini",
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
        2. Format messages with system prompt
        3. Get completion from OpenAI
        4. Extract the response text
        """

        # Step 1: Define a query to get recent messages
        @pxt.query
        def get_recent_memory(current_timestamp: pxt.Timestamp) -> list[dict]:
            """
            Get recent messages from memory, respecting n_latest_messages limit if set.
            Messages are ordered by timestamp (newest first).
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
            memory_context=get_recent_memory(self.agent.timestamp),
            if_exists="ignore",
        )

        # Format messages for OpenAI with system prompt
        self.agent.add_computed_column(
            prompt=create_messages(
                self.agent.system_prompt,
                self.agent.memory_context,
                self.agent.user_message,
                self.agent.image,
            ),
            if_exists="ignore",
        )

        # Get OpenAI's API response
        self.agent.add_computed_column(
            response=chat_completions(
                messages=self.agent.prompt, model=self.model, **self.chat_kwargs
            ),
            if_exists="ignore",
        )

        # Extract the final response text
        self.agent.add_computed_column(
            agent_response=self.agent.response.choices[0].message.content,
            if_exists="ignore",
        )

    def _setup_tools_pipeline(self):
        """
        Configure the tool execution pipeline using Pixeltable's computed columns.
        This method implements the abstract method from BaseAgent.

        The pipeline has 4 stages:
        1. Get initial response from OpenAI with potential tool calls
        2. Execute any requested tools
        3. Format tool results for follow-up
        4. Get final response incorporating tool outputs
        """
        # Stage 1: Get initial response with potential tool calls
        self.tools_table.add_computed_column(
            initial_response=chat_completions(
                model=self.model,
                messages=[{"role": "user", "content": self.tools_table.tool_prompt}],
                tools=self.tools,  # Pass available tools to OpenAI
                **self.tool_kwargs,
            ),
            if_exists="ignore",
        )

        # Stage 2: Execute any tools that OpenAI requested
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
            final_response=chat_completions(
                model=self.model,
                messages=[
                    {"role": "user", "content": self.tools_table.tool_response_prompt},
                ],
                **self.tool_kwargs,
            ),
            if_exists="ignore",
        )

        # Extract the final response text
        self.tools_table.add_computed_column(
            tool_answer=self.tools_table.final_response.choices[0].message.content,
            if_exists="ignore",
        )
