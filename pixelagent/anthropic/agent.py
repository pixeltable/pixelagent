from typing import Optional

import pixeltable as pxt
import pixeltable.functions as pxtf

from pixelagent.core.base import BaseAgent

from .utils import create_messages


try:
    import anthropic
    from pixeltable.functions.anthropic import invoke_tools, messages
except ImportError:
    raise ImportError("anthropic not found; run `pip install anthropic`")


class Agent(BaseAgent):
    def __init__(
        self,
        agent_name: str,
        system_prompt: str,
        model: str = "claude-3-5-sonnet-latest",
        n_latest_messages: Optional[int] = 10,
        tools: Optional[pxt.tools] = None,
        reset: bool = False,
        chat_kwargs: Optional[dict] = None,
        tool_kwargs: Optional[dict] = None,
    ):
        super().__init__(
            agent_name=agent_name,
            system_prompt=system_prompt,
            model=model,
            n_latest_messages=n_latest_messages,
            tools=tools,
            reset=reset,
            chat_kwargs=chat_kwargs,
            tool_kwargs=tool_kwargs,
        )

    def _setup_chat_pipeline(self):
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

        self.agent.add_computed_column(
            memory_context=get_recent_memory(self.agent.timestamp), if_exists="ignore"
        )

        self.agent.add_computed_column(
            messages=create_messages(
                self.agent.memory_context, self.agent.user_message
            ),
            if_exists="ignore",
        )

        self.agent.add_computed_column(
            response=messages(
                messages=self.agent.messages,
                model=self.model,
                system=self.system_prompt,
                **self.chat_kwargs,
            ),
            if_exists="ignore",
        )

        self.agent.add_computed_column(
            agent_response=self.agent.response.content[0].text, if_exists="ignore"
        )

    def _setup_tools_table(self):
        self.tools_table.add_computed_column(
            initial_response=messages(
                model=self.model,
                system=self.system_prompt,
                messages=[{"role": "user", "content": self.tools_table.tool_prompt}],
                tools=self.tools,
                **self.tool_kwargs,
            ),
            if_exists="ignore",
        )
        self.tools_table.add_computed_column(
            tool_output=invoke_tools(self.tools, self.tools_table.initial_response),
            if_exists="ignore",
        )
        self.tools_table.add_computed_column(
            tool_response_prompt=pxtf.string.format(
                "{0}: {1}", self.tools_table.tool_prompt, self.tools_table.tool_output
            ),
            if_exists="ignore",
        )
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
        self.tools_table.add_computed_column(
            tool_answer=self.tools_table.final_response.content[0].text,
            if_exists="ignore",
        )
