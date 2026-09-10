from datetime import datetime
from typing import Optional

import pixeltable as pxt
from PIL import Image

from . import schema
from .spec import ProviderSpec


class BaseAgent:
    """
    An agent with persistent memory and tool execution, backed by Pixeltable.

    The schema is declared in pixelagent/core/schema.py and reconciled on
    construction; this class is the Python facade over it. A turn is a row:
    `chat()` inserts one and reads back what the pipeline computed.

    Two tables, plus a third when tools are configured:
      memory: every message, user and assistant
      agent:  one row per turn, and the chat pipeline
      tools:  the tool-calling handshake

    Subclasses supply a ProviderSpec and defaults; they contain no pipeline
    logic of their own.
    """

    spec: ProviderSpec

    def __init__(
        self,
        name: str,
        system_prompt: str,
        model: str,
        n_latest_messages: Optional[int] = 10,
        tools: pxt.Tools | None = None,
        reset: bool = False,
        chat_kwargs: Optional[dict] = None,
        tool_kwargs: Optional[dict] = None,
        max_tokens: int = 4096,
    ):
        """
        Args:
            name: unique name for the agent; also its Pixeltable directory
            system_prompt: system prompt guiding the model's behaviour
            model: the model to call. Part of the schema, so changing it for an
                existing name raises SchemaConflict (pass reset=True instead).
            n_latest_messages: how many recent messages to include as context;
                None for the whole conversation
            tools: optional tools for function calling. Also part of the schema.
            reset: drop any existing data for this name first
            chat_kwargs: extra provider options for chat turns
            tool_kwargs: extra provider options for tool turns
            max_tokens: cap on generated tokens; required by Anthropic
        """
        self.directory = name
        self.system_prompt = system_prompt
        self.model = model
        self.n_latest_messages = n_latest_messages
        self.tools = tools
        self.chat_kwargs = chat_kwargs or {}
        self.tool_kwargs = tool_kwargs or {}
        self.max_tokens = max_tokens

        schema.build(
            self.directory,
            self.spec,
            model=self.model,
            tools=self.tools,
            reset=reset,
        )

        self.memory = pxt.get_table(f"{self.directory}.memory")
        self.agent = pxt.get_table(f"{self.directory}.agent")
        self.tools_table = (
            pxt.get_table(f"{self.directory}.tools") if self.tools else None
        )

    @property
    def _history_limit(self) -> int:
        if self.n_latest_messages is None:
            return schema.UNLIMITED_HISTORY
        return self.n_latest_messages

    def chat(
        self,
        message: str,
        image: Optional[Image.Image] = None,
        *,
        conversation_id: str = "default",
        system_prompt: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send a message and return the reply.

        The per-call overrides are possible because these settings are columns
        rather than baked into the pipeline, so varying them costs no rebuild.
        """
        now = datetime.now()
        status = self.agent.insert(
            [
                {
                    "conversation_id": conversation_id,
                    "user_message": message,
                    "system_prompt": (
                        system_prompt
                        if system_prompt is not None
                        else self.system_prompt
                    ),
                    "model_kwargs": (
                        model_kwargs if model_kwargs is not None else self.chat_kwargs
                    )
                    or None,
                    "max_tokens": (
                        max_tokens if max_tokens is not None else self.max_tokens
                    ),
                    "n_latest": self._history_limit,
                    "image": image,
                    "timestamp": now,
                }
            ],
            return_rows=True,
        )
        response = status.rows[0]["agent_response"]

        # Memory is written only once the turn has produced a reply. Writing the
        # user's message first, as this did before 0.2.0, meant a failed turn
        # left an unanswered message in history permanently.
        self.memory.insert(
            [
                {
                    "conversation_id": conversation_id,
                    "role": "user",
                    "content": message,
                    "timestamp": now,
                },
                {
                    "conversation_id": conversation_id,
                    "role": "assistant",
                    "content": response,
                    "timestamp": now,
                },
            ]
        )
        return response

    def tool_call(
        self,
        prompt: str,
        *,
        conversation_id: str = "default",
        system_prompt: Optional[str] = None,
    ) -> str:
        """Run the tool-calling handshake and return the final answer."""
        if not self.tools:
            return "No tools configured for this agent."

        now = datetime.now()
        status = self.tools_table.insert(
            [
                {
                    "tool_prompt": prompt,
                    "system_prompt": (
                        system_prompt
                        if system_prompt is not None
                        else self.system_prompt
                    ),
                    "model_kwargs": self.tool_kwargs or None,
                    "max_tokens": self.max_tokens,
                    "timestamp": now,
                }
            ],
            return_rows=True,
        )
        tool_answer = status.rows[0]["tool_answer"]

        self.memory.insert(
            [
                {
                    "conversation_id": conversation_id,
                    "role": "user",
                    "content": prompt,
                    "timestamp": now,
                },
                {
                    "conversation_id": conversation_id,
                    "role": "assistant",
                    "content": tool_answer,
                    "timestamp": datetime.now(),
                },
            ]
        )
        return tool_answer
