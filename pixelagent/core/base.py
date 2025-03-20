from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional
from uuid import uuid4

import pixeltable as pxt


class BaseAgent(ABC):
    def __init__(
        self,
        agent_name: str,
        system_prompt: str,
        model: str,
        n_latest_messages: Optional[int] = 10,
        tools: Optional[pxt.tools] = None,
        reset: bool = False,
        chat_kwargs: Optional[dict] = None,
        tool_kwargs: Optional[dict] = None,
    ):
        self.directory = agent_name
        self.system_prompt = system_prompt
        self.model = model
        self.n_latest_messages = n_latest_messages
        self.tools = tools
        self.chat_kwargs = chat_kwargs or {}
        self.tool_kwargs = tool_kwargs or {}

        if reset:
            pxt.drop_dir(self.directory, force=True)

        pxt.create_dir(self.directory, if_exists="ignore")

        self._setup_tables()

        self.memory = pxt.get_table(f"{self.directory}.memory")
        self.agent = pxt.get_table(f"{self.directory}.agent")
        self.tools_table = (
            pxt.get_table(f"{self.directory}.tools") if self.tools else None
        )

    def _setup_tables(self):
        self.memory = pxt.create_table(
            f"{self.directory}.memory",
            {
                "message_id": pxt.String,
                "role": pxt.String,
                "content": pxt.String,
                "timestamp": pxt.Timestamp,
            },
            if_exists="ignore",
        )

        self.agent = pxt.create_table(
            f"{self.directory}.agent",
            {
                "message_id": pxt.String,
                "user_message": pxt.String,
                "timestamp": pxt.Timestamp,
                "system_prompt": pxt.String,
            },
            if_exists="ignore",
        )

        if self.tools:
            self.tools_table = pxt.create_table(
                f"{self.directory}.tools",
                {
                    "tool_invoke_id": pxt.String,
                    "tool_prompt": pxt.String,
                    "timestamp": pxt.Timestamp,
                },
                if_exists="ignore",
            )
            self._setup_tools_table()

        self._setup_chat_pipeline()

    @abstractmethod
    def _setup_chat_pipeline(self):
        """To be implemented by subclasses"""
        raise NotImplementedError

    @abstractmethod
    def _setup_tools_table(self):
        """To be implemented by subclasses"""
        raise NotImplementedError

    def chat(self, message: str) -> str:
        now = datetime.now()

        user_message_id = str(uuid4())
        assistant_message_id = str(uuid4())

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
        self.agent.insert(
            [
                {
                    "message_id": user_message_id,
                    "user_message": message,
                    "timestamp": now,
                    "system_prompt": self.system_prompt,
                }
            ]
        )

        result = (
            self.agent.select(self.agent.agent_response)
            .where(self.agent.message_id == user_message_id)
            .collect()
        )
        response = result["agent_response"][0]

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

        self.tools_table.insert(
            [
                {
                    "tool_invoke_id": tool_invoke_id,
                    "tool_prompt": prompt,
                    "timestamp": now,
                }
            ]
        )

        result = (
            self.tools_table.select(self.tools_table.tool_answer)
            .where(self.tools_table.tool_invoke_id == tool_invoke_id)
            .collect()
        )
        tool_answer = result["tool_answer"][0]

        # Store assistant response in memory
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
