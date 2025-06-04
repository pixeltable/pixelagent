"""
Tutorial: Building Your Own Google Gemini-powered Agent with Pixeltable

This example demonstrates how to create a conversational AI agent using Google Gemini's LLM model
and Pixeltable for persistent memory, storage, orchestration, and tool execution. The agent can maintain conversation
history and execute tools while keeping track of all interactions in a structured database.
"""

from datetime import datetime
from typing import Optional
from uuid import uuid4

import PIL
import pixeltable as pxt
import pixeltable.functions as pxtf

try:
    from pixeltable.functions.gemini import generate_content, invoke_tools
except ImportError:
    raise ImportError("google.genai not found; run `pip install google-genai`")


@pxt.udf
def create_content(
    system_prompt: str,
    memory_context: list[dict],
    current_message: str,
    image: Optional[PIL.Image.Image] = None,
) -> str:
    
    # Build the conversation context as a text string
    context = f"System: {system_prompt}\n\n"
    
    # Add memory context
    for msg in memory_context:
        context += f"{msg['role'].title()}: {msg['content']}\n"
    
    # Add current message
    context += f"User: {current_message}\n"
    context += "Assistant: "
    
    return context


class Agent:
    """
    A Google Gemini agent powered by their LLM models with persistent memory and tool execution capabilities.

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
        model: str = "gemini-2.0-flash",
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
                "system_prompt": pxt.String,  # System prompt for LLM
                "image": pxt.Image,          # Optional image attachment
            },
            if_exists="ignore",
        )

        # Create tools table if tools are configured
        if self.tools:
            self.tools_table = pxt.create_table(
                f"{self.directory}.tools",
                {
                    "tool_invoke_id": pxt.String,  # Unique ID for each tool invocation
                    "tool_prompt": pxt.String,  # Tool prompt for Gemini
                    "timestamp": pxt.Timestamp,  # When the tool was invoked
                },
                if_exists="ignore",
            )
            # Set up tools pipeline
            self._setup_tools_pipeline()

        # Set up chat pipeline
        self._setup_chat_pipeline()

    def _setup_chat_pipeline(self):
        """
        Configure the chat completion pipeline using Pixeltable computed columns
        1. Get recent messages from memory
        2. Create messages for Gemini
        3. Get Gemini's response
        4. Extract the response text
        """

        # Get recent messages from memory, respecting n_latest_messages limit if set
        @pxt.query
        def get_recent_memory(current_timestamp: pxt.Timestamp) -> list[dict]:
            query = (
                self.memory.where(self.memory.timestamp < current_timestamp)
                .order_by(self.memory.timestamp, asc=False)
                .select(role=self.memory.role, content=self.memory.content)
            )
            if self.n_latest_messages is not None:
                query = query.limit(self.n_latest_messages)
            return query

        # Add computed columns to agent table
        # Get recent messages from memory
        self.agent.add_computed_column(
            memory_context=get_recent_memory(self.agent.timestamp),
            if_exists="ignore",
        )
        # Create content for Gemini
        self.agent.add_computed_column(
            prompt=create_content(
                self.agent.system_prompt,
                self.agent.memory_context,
                self.agent.user_message,
            ),
            if_exists="ignore",
        )
        # Get api response
        self.agent.add_computed_column(
            response=generate_content(
                contents=self.agent.prompt, model=self.model, **self.chat_kwargs
            ),
            if_exists="ignore",
        )
        # Extract the response text
        self.agent.add_computed_column(
            agent_response=self.agent.response['candidates'][0]['content']['parts'][0]['text'],
            if_exists="ignore",
        )

    def _setup_tools_pipeline(self):
        """
        Configure the tool call handshake pipeline using Pixeltable computed columns
        1. Get initial response from Gemini with tool calls
        2. Execute the tools
        3. Pass the tool results back to Gemini for final response
        """
        # Get initial response from Gemini with tool calls
        self.tools_table.add_computed_column(
            initial_response=generate_content(
                contents=self.tools_table.tool_prompt,
                model=self.model,
                tools=self.tools,
                **self.tool_kwargs,
            ),
            if_exists="ignore",
        )
        # Execute the tools
        self.tools_table.add_computed_column(
            tool_output=invoke_tools(self.tools, self.tools_table.initial_response),
            if_exists="ignore",
        )
        # Pass the tool results back to Gemini for final response
        self.tools_table.add_computed_column(
            tool_response_prompt=pxtf.string.format(
                "{0}: {1}", self.tools_table.tool_prompt, self.tools_table.tool_output
            ),
            if_exists="ignore",
        )
        # Get final response from Gemini with tool results
        self.tools_table.add_computed_column(
            final_response=generate_content(
                contents=self.tools_table.tool_response_prompt,
                model=self.model,
                **self.tool_kwargs,
            ),
            if_exists="ignore",
        )
        # Extract the final response text
        self.tools_table.add_computed_column(
            tool_answer=self.tools_table.final_response['candidates'][0]['content']['parts'][0]['text'],
            if_exists="ignore",
        )

    def chat(self, message: str, image: Optional[PIL.Image.Image] = None) -> str:
        """
        Send a message to the agent and get its response.

        This method:
        1. Stores the user message in memory
        2. Triggers the chat completion pipeline
        3. Stores the assistant's response in memory
        4. Returns the response

        Args:
            message: The user's message
            image: Optional image attachment

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

        now = datetime.now()
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
                    "timestamp": now,
                }
            ]
        )

        # Store user prompt in tools table (which triggers the tool call handshake pipeline)
        self.tools_table.insert(
            [
                {
                    "tool_invoke_id": tool_invoke_id,
                    "tool_prompt": prompt,
                    "timestamp": now,
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

        # Store LLM's response in memory
        self.memory.insert(
            [
                {
                    "message_id": assistant_message_id,
                    "role": "assistant",
                    "content": tool_answer,
                    "timestamp": now,
                }
            ]
        )
        return tool_answer
