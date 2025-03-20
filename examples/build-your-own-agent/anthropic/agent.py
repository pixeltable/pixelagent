from datetime import datetime
from typing import Optional
from uuid import uuid4

import pixeltable as pxt

try:
    from pixeltable.functions.anthropic import invoke_tools, messages
except ImportError:
    raise ImportError("anthropic not found; run `pip install anthropic`")


@pxt.udf
def format_tool_results(
    original_prompt: str, tool_inputs: list[dict], tool_outputs: dict
) -> str:
    result = f"Original prompt: {original_prompt}\nTool information:\n"

    for tool in tool_inputs:
        tool_name = tool.get("name")
        input_data = tool.get("input")
        outputs = tool_outputs.get(tool_name, ["No output"])
        for output_data in outputs:
            result += (
                f"Tool: {tool_name}\nInput: {input_data}\nOutput: {output_data}\n----\n"
            )

    return result.rstrip()


@pxt.udf
def create_messages(memory_context: list[dict], current_message: str) -> list[dict]:
    messages = memory_context.copy()
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

        self.__setup_tables()

        self.memory = pxt.get_table(f"{self.directory}.memory")
        self.agent = pxt.get_table(f"{self.directory}.agent")
        self.tools_table = (
            pxt.get_table(f"{self.directory}.tools") if self.tools else None
        )

    def __setup_tables(self):
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
            self.__setup_tools_table()

        self.__setup_chat_pipeline()

    def __setup_chat_pipeline(self):
        @pxt.query
        def get_recent_memory(current_timestamp: pxt.Timestamp) -> list[dict]:
            return (
                self.memory.where(self.memory.timestamp < current_timestamp)
                .order_by(self.memory.timestamp, asc=False)
                .select(role=self.memory.role, content=self.memory.content)
                .limit(self.n_latest_messages)
            )

        self.agent.add_computed_column(
            memory_context=get_recent_memory(self.agent.timestamp), if_exists="ignore"
        )
        self.agent.add_computed_column(
            prompt=create_messages(self.agent.memory_context, self.agent.user_message),
            if_exists="ignore",
        )
        self.agent.add_computed_column(
            response=messages(
                messages=self.agent.prompt,
                model=self.model,
                system=self.system_prompt,
                **self.chat_kwargs,
            ),
            if_exists="ignore",
        )
        self.agent.add_computed_column(
            agent_response=self.agent.response.content[0].text, if_exists="ignore"
        )

    def __setup_tools_table(self):
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
            tool_input=self.tools_table.initial_response.content, if_exists="ignore"
        )
        self.tools_table.add_computed_column(
            tool_output=invoke_tools(self.tools, self.tools_table.initial_response),
            if_exists="ignore",
        )
        self.tools_table.add_computed_column(
            formatted_results=format_tool_results(
                self.tools_table.tool_prompt,
                self.tools_table.tool_input,
                self.tools_table.tool_output,
            ),
            if_exists="ignore",
        )
        self.tools_table.add_computed_column(
            final_response=messages(
                model=self.model,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": self.tools_table.formatted_results}
                ],
                **self.tool_kwargs,
            ),
            if_exists="ignore",
        )
        self.tools_table.add_computed_column(
            tool_answer=self.tools_table.final_response.content[0].text,
            if_exists="ignore",
        )

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
