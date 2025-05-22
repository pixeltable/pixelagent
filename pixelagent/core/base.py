from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional
from uuid import uuid4

import pixeltable as pxt
from PIL import Image


class BaseAgent(ABC):
    """
    An Base agent powered by LLM model with persistent memory and tool execution capabilities.

    This base agent gets inherited by other agent classes (see pixelagent/anthropic/agent.py and pixelagent/anthropic/agent.py).

    The agent maintains three key tables in Pixeltable:
    1. memory: Stores all conversation history with timestamps
    2. agent: Manages chat interactions and responses
    3. tools: (Optional) Handles tool execution and responses

    Key Features:
    - Persistent conversation memory with optional message limit
    - Tool execution support
    - Structured data storage and orchestration using Pixeltable
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        model: str,
        n_latest_messages: Optional[int] = 10,
        tools: Optional[pxt.tools] = None,
        reset: bool = False,
        chat_kwargs: Optional[dict] = None,
        tool_kwargs: Optional[dict] = None,
    ):
        """
        Initialize the agent with the specified configuration.

        Args:
            name: Unique name for the agent (used for table names)
            system_prompt: System prompt that guides LLM's behavior
            model: LLM model to use
            n_latest_messages: Number of recent messages to include in context (None for unlimited)
            tools: Optional tools configuration for function calling
            reset: If True, deletes existing agent data
            chat_kwargs: Additional kwargs for chat completion
            tool_kwargs: Additional kwargs for tool execution
        """
        self.directory = name
        self.system_prompt = system_prompt
        self.model = model
        self.n_latest_messages = n_latest_messages
        self.tools = tools
        self.chat_kwargs = chat_kwargs or {}
        self.tool_kwargs = tool_kwargs or {}

        # Set up or reset the agent's database
        if reset:
            pxt.drop_dir(self.directory, if_not_exists="ignore", force=True)

        # Create agent directory if it doesn't exist
        pxt.create_dir(self.directory, if_exists="ignore")

        # Set up tables
        self._setup_tables()

        # Get references to the created tables
        self.memory = pxt.get_table(f"{self.directory}.memory")
        self.agent = pxt.get_table(f"{self.directory}.agent")
        self.tools_table = (
            pxt.get_table(f"{self.directory}.tools") if self.tools else None
        )

    def _setup_tables(self):
        """
        Initialize the required Pixeltable tables for the agent.
        Creates three tables:
        1. memory: Stores conversation history
        2. agent: Manages chat completions
        3. tools: (Optional) Handles tool execution
        """
        # Create memory table for conversation history
        self.memory = pxt.create_table(
            f"{self.directory}.memory",
            {
                "message_id": pxt.String,  # Unique ID for each message
                "role": pxt.String,  # 'user' or 'assistant'
                "content": pxt.String,  # Message content
                "timestamp": pxt.Timestamp,  # When the message was received
            },
            if_exists="ignore",
        )

        # Create agent table for managing chat interactions
        self.agent = pxt.create_table(
            f"{self.directory}.agent",
            {
                "message_id": pxt.String,  # Unique ID for each message
                "user_message": pxt.String,  # User's message content
                "timestamp": pxt.Timestamp,  # When the message was received
                "system_prompt": pxt.String,  # System prompt for Claude
                "image": pxt.Image,  # Optional image attachment
            },
            if_exists="ignore",
        )

        # Create tools table if tools are configured
        if self.tools:
            self.tools_table = pxt.create_table(
                f"{self.directory}.tools",
                {
                    "tool_invoke_id": pxt.String,  # Unique ID for each tool invocation
                    "tool_prompt": pxt.String,  # Tool prompt for Claude
                    "timestamp": pxt.Timestamp,  # When the tool was invoked
                },
                if_exists="ignore",
            )
            # Set up tools pipeline
            self._setup_tools_pipeline()

        # Set up chat pipeline
        self._setup_chat_pipeline()

    @abstractmethod
    def _setup_chat_pipeline(self):
        """To be implemented by subclasses"""
        raise NotImplementedError

    @abstractmethod
    def _setup_tools_pipeline(self):
        """To be implemented by subclasses"""
        raise NotImplementedError

    def chat(self, message: str, image: Optional[Image.Image] = None) -> str:
        """
        Send a message to the agent and get its response.

        This method:
        1. Stores the user message in memory
        2. Triggers the chat completion pipeline
        3. Stores the assistant's response in memory
        4. Returns the response

        Args:
            message: The user's message

        Returns:
            The agent's response
        """
        now = datetime.now()

        # Generate unique IDs for the message pair
        user_message_id = str(uuid4())
        assistant_message_id = str(uuid4())

        # Store user message in memory
        self.memory.insert(
            [
                {
                    "message_id": user_message_id,
                    "role": "user",
                    "content": message,
                    "timestamp": now,
                }
            ]
        )

        # Store user message in agent table (which triggers the chat pipeline)
        self.agent.insert(
            [
                {
                    "message_id": user_message_id,
                    "user_message": message,
                    "timestamp": now,
                    "system_prompt": self.system_prompt,
                    "image": image,
                }
            ]
        )

        # Get LLM's response from agent table
        result = (
            self.agent.select(self.agent.agent_response)
            .where(self.agent.message_id == user_message_id)
            .collect()
        )
        response = result["agent_response"][0]

        # Store LLM's response in memory
        self.memory.insert(
            [
                {
                    "message_id": assistant_message_id,
                    "role": "assistant",
                    "content": response,
                    "timestamp": now,
                }
            ]
        )
        return response

    def tool_call(self, prompt: str) -> str:
        """
        Execute a tool call with the given prompt.

        This method:
        1. Stores the user prompt in memory
        2. Triggers the tool call handshake pipeline
        3. Stores the tool's response in memory
        4. Returns the response

        Args:
            prompt: The user's prompt

        Returns:
            The tool's response
        """
        if not self.tools:
            return "No tools configured for this agent."

        # Use separate timestamps for user and assistant messages
        user_timestamp = datetime.now()
        user_message_id = str(uuid4())
        tool_invoke_id = str(uuid4())
        assistant_message_id = str(uuid4())

        # Store user message in memory
        self.memory.insert(
            [
                {
                    "message_id": user_message_id,
                    "role": "user",
                    "content": prompt,
                    "timestamp": user_timestamp,
                }
            ]
        )

        # Store user prompt in tools table (which triggers the tool call handshake pipeline)
        self.tools_table.insert(
            [
                {
                    "tool_invoke_id": tool_invoke_id,
                    "tool_prompt": prompt,
                    "timestamp": user_timestamp,
                }
            ]
        )

        # Get tool answer from tools table
        result = (
            self.tools_table.select(self.tools_table.tool_answer)
            .where(self.tools_table.tool_invoke_id == tool_invoke_id)
            .collect()
        )
        tool_answer = result["tool_answer"][0]

        # Store LLM's response in memory with a slightly later timestamp
        assistant_timestamp = datetime.now()
        self.memory.insert(
            [
                {
                    "message_id": assistant_message_id,
                    "role": "assistant",
                    "content": tool_answer,
                    "timestamp": assistant_timestamp,
                }
            ]
        )
        return tool_answer
