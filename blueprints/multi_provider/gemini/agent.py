from typing import Optional

import pixeltable as pxt
import pixeltable.functions as pxtf

from ..core.base import BaseAgent

from .utils import create_content

try:
    from pixeltable.functions.gemini import generate_content, invoke_tools
except ImportError:
    raise ImportError("google.genai not found; run `pip install google-genai`")


class Agent(BaseAgent):
    """
    Google Gemini-specific implementation of the BaseAgent.

    This agent uses Google Gemini's generate_content API for generating responses and handling tools.
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
        model: str = "gemini-2.0-flash",
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
        2. Format content for Gemini
        3. Get completion from Google Gemini
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

        # Format content for Gemini without system prompt
        self.agent.add_computed_column(
            prompt=create_content(
                self.agent.memory_context,
                self.agent.user_message,
            ),
            if_exists="ignore",
        )

        # Prepare config with system instruction
        chat_config = self.chat_kwargs.copy() if self.chat_kwargs else {}
        chat_config['system_instruction'] = self.system_prompt

        # Get Gemini's API response (note: contents parameter instead of messages)
        self.agent.add_computed_column(
            response=generate_content(
                contents=self.agent.prompt,
                model=self.model,
                config=chat_config,
            ),
            if_exists="ignore",
        )

        # Extract the final response text from Gemini's specific response format
        self.agent.add_computed_column(
            agent_response=self.agent.response['candidates'][0]['content']['parts'][0]['text'], 
            if_exists="ignore"
        )

    def _setup_tools_pipeline(self):
        """
        Configure the tool execution pipeline using Pixeltable's computed columns.
        This method implements the abstract method from BaseAgent.

        The pipeline has 4 stages:
        1. Get initial response from Gemini with potential tool calls
        2. Execute any requested tools
        3. Format tool results for follow-up
        4. Get final response incorporating tool outputs

        Note: Gemini's tool calling format uses the same structure as other providers
        thanks to BaseAgent abstraction.
        """
        # Prepare config with system instruction for tools
        tool_config = self.tool_kwargs.copy() if self.tool_kwargs else {}
        tool_config['system_instruction'] = self.system_prompt
        
        # Stage 1: Get initial response with potential tool calls
        self.tools_table.add_computed_column(
            initial_response=generate_content(
                contents=self.tools_table.tool_prompt,
                model=self.model,
                tools=self.tools,  # Pass available tools to Gemini
                config=tool_config,
            ),
            if_exists="ignore",
        )

        # Stage 2: Execute any tools that Gemini requested
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
            final_response=generate_content(
                contents=self.tools_table.tool_response_prompt,
                model=self.model,
                config=tool_config,
            ),
            if_exists="ignore",
        )

        # Extract the final response text from Gemini's format
        self.tools_table.add_computed_column(
            tool_answer=self.tools_table.final_response['candidates'][0]['content']['parts'][0]['text'],
            if_exists="ignore",
        )
