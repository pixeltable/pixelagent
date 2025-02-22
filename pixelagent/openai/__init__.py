import inspect
import json
from datetime import datetime
from functools import wraps
from typing import Callable, Dict, List

import pixeltable as pxt
from openai import OpenAI
from duckduckgo_search import DDGS


def tool(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    sig = inspect.signature(func)
    parameters = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    required_params = []
    for param_name, param in sig.parameters.items():
        param_type = "string" if param.annotation == str else "integer" if param.annotation == int else "string"
        param_schema = {"type": param_type}
        if func.__doc__:
            param_schema["description"] = f"Parameter {param_name}"
        parameters["properties"][param_name] = param_schema
        if param.default == inspect.Parameter.empty:
            required_params.append(param_name)

    if required_params:
        parameters["required"] = required_params

    tool_dict = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__.strip() if func.__doc__ else f"Calls {func.__name__}",
            "parameters": parameters,
            "strict": False,
        },
    }
    wrapper.tool_definition = tool_dict
    return wrapper


def setup_pixeltable(name: str, reset: bool = False):
    tables = [i for i in pxt.list_tables() if i.startswith(name)]
    if reset or len(tables) == 0:
        pxt.drop_dir(name, force=True)
        pxt.create_dir(name)

        messages = pxt.create_table(
            f"{name}.messages",
            {
                "message_id": pxt.IntType(),
                "role": pxt.StringType(),
                "content": pxt.StringType(),
                "timestamp": pxt.TimestampType(),
            },
            primary_key="message_id",
        )

        chat = pxt.create_table(
            f"{name}.chat",
            {
                "system_prompt": pxt.StringType(),
                "user_input": pxt.StringType(),
                "response": pxt.StringType(nullable=True),
            },
        )

        tool_calls = pxt.create_table(
            f"{name}.tool_calls",
            {
                "tool_call_id": pxt.StringType(),
                "message_id": pxt.IntType(),
                "tool_name": pxt.StringType(),
                "arguments": pxt.JsonType(),
                "result": pxt.StringType(nullable=True),
                "timestamp": pxt.TimestampType(),
            },
        )
    else:
        messages = pxt.get_table(f"{name}.messages")
        chat = pxt.get_table(f"{name}.chat")
        tool_calls = pxt.get_table(f"{name}.tool_calls")

    return messages, chat, tool_calls


class Agent:
    def __init__(
        self,
        name: str,
        system_prompt: str,
        model: str = "gpt-4o-mini",
        tools: List[Callable] = None,
        reset: bool = False,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.client = OpenAI()
        self.model = model
        self.tools = tools if tools else []
        self.reset = reset
        self.tool_definitions = [tool.tool_definition for tool in self.tools]
        self.available_tools = {tool.__name__: tool for tool in self.tools}

        self.messages_table, self.chat_table, self.tool_calls_table = setup_pixeltable(
            name, reset
        )
        self.message_counter = 0

    def get_history(self) -> List[Dict]:
        """Fetch conversation history from Pixeltable."""
        result = self.messages_table.select(
            self.messages_table.role, self.messages_table.content
        ).collect()
        return [{"role": row["role"], "content": row["content"]} for row in result]

    def process_tool_calls(self, completion, message_id: int) -> List[Dict]:
        tool_calls = completion.choices[0].message.tool_calls
        if not tool_calls:
            return []

        results = []
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            if function_name in self.available_tools:
                func = self.available_tools[function_name]
                try:
                    result = func(**arguments)
                    result_str = str(result)
                    results.append({"tool_call_id": tool_call.id, "result": result_str})
                except Exception as e:
                    result_str = f"Error: {str(e)}"
                    results.append({"tool_call_id": tool_call.id, "result": result_str})
            else:
                result_str = f"Error: Function {function_name} not found"
                results.append({"tool_call_id": tool_call.id, "result": result_str})

        self.tool_calls_table.insert(
            [
                {
                    "tool_call_id": tool_call.id,
                    "message_id": message_id,
                    "tool_name": function_name,
                    "arguments": arguments,
                    "result": result_str,
                    "timestamp": datetime.now(),
                }
            ]
        )

        return results

    def run(self, user_input: str) -> str:
        self.message_counter += 1

        message_id = self.message_counter
        history = self.get_history()

        messages = (
            [{"role": "system", "content": self.system_prompt}]
            + history
            + [{"role": "user", "content": user_input}]
        )

        self.messages_table.insert(
            [
                {
                    "message_id": message_id,
                    "role": "user",
                    "content": user_input,
                    "timestamp": datetime.now(),
                }
            ]
        )

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tool_definitions if self.tools else None,
        )

        if (
            not hasattr(completion.choices[0].message, "tool_calls")
            or not completion.choices[0].message.tool_calls
        ):
            response = completion.choices[0].message.content
        else:
            tool_results = self.process_tool_calls(completion, message_id)
            messages.extend([completion.choices[0].message.to_dict()])
            for result in tool_results:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": result["tool_call_id"],
                        "content": result["result"],
                    }
                )

            final_completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tool_definitions if self.tools else None,
            )
            response = final_completion.choices[0].message.content

        self.message_counter += 1
        self.messages_table.insert(
            [
                {
                    "message_id": self.message_counter,
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now(),
                }
            ]
        )

        self.chat_table.insert(
            [
                {
                    "system_prompt": self.system_prompt,
                    "user_input": user_input,
                    "response": response,
                }
            ]
        )

        return response