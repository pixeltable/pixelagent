import inspect
import json
from datetime import datetime
from functools import wraps
from typing import Callable, Dict, List, Optional, Type, Union, get_type_hints

from openai import OpenAI
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich import print as rprint

from pixelagent.utils import setup_pixeltable

def tool(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    # Get function signature and type hints
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)

    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,  # Strict mode disallows unspecified properties
    }

    for param_name, param in sig.parameters.items():
        # Determine parameter type from type hints or default to string
        param_type = type_hints.get(param_name, str)
        if param_type == str:
            schema_type = "string"
        elif param_type == int:
            schema_type = "integer"
        elif param_type == float:
            schema_type = "number"
        elif param_type == bool:
            schema_type = "boolean"
        else:
            schema_type = "string"  # Fallback for unsupported types

        param_schema = {"type": schema_type}
        if func.__doc__:
            param_schema["description"] = f"Parameter {param_name}"

        parameters["properties"][param_name] = param_schema
        if param.default == inspect.Parameter.empty:
            parameters["required"].append(param_name)

    tool_dict = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__.strip()
            if func.__doc__
            else f"Calls {func.__name__}",
            "parameters": parameters,
            "strict": False,  # Enable strict mode
        },
    }
    wrapper.tool_definition = tool_dict
    return wrapper

class Agent:
    def __init__(
        self,
        name: str,
        system_prompt: str,
        model: str = "gpt-4o-mini",
        tools: List[Callable] = None,
        structured_output: Optional[Type[BaseModel]] = None,
        reset: bool = False,
        enable_observability: bool = True,  # New parameter
        **default_kwargs,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.client = OpenAI()
        self.model = model
        self.tools = tools if tools else []
        self.structured_output = structured_output
        self.reset = reset
        self.default_kwargs = default_kwargs  # Store defaults
        self.tool_definitions = [tool.tool_definition for tool in self.tools]
        self.available_tools = {tool.__name__: tool for tool in self.tools}
        self.messages_table, self.chat_table, self.tool_calls_table = setup_pixeltable(
            name, reset
        )
        self.message_counter = 0
        
        # Observability settings
        self.enable_observability = enable_observability
        self.console = Console() if enable_observability else None

    def display_message(self, role: str, content: str, attachments: Optional[str] = None):
        """Display a message with rich formatting."""
        if not self.enable_observability:
            return
            
        if role == "system":
            self.console.print(Panel(content, title="System", border_style="yellow"))
        elif role == "user":
            self.console.print(Panel(content, title="User", border_style="green"))
            if attachments:
                self.console.print(Panel(f"[Attachment: {attachments}]", border_style="green"))
        elif role == "assistant":
            try:
                # Try to parse as markdown
                md = Markdown(content)
                self.console.print(Panel(md, title="Assistant", border_style="blue"))
            except:
                # Fallback to plain text
                self.console.print(Panel(content, title="Assistant", border_style="blue"))
        elif role == "tool":
            self.console.print(Panel(content, title="Tool Result", border_style="purple"))

    def display_thinking(self, message: str):
        """Display a thinking/processing message."""
        if not self.enable_observability:
            return
        self.console.print(f"[dim italic]{message}[/dim italic]")

    def display_tool_call(self, tool_name: str, arguments: Dict, result: str):
        """Display a tool call in a table format."""
        if not self.enable_observability:
            return
            
        table = Table(title=f"Tool Call: {tool_name}")
        table.add_column("Arguments", style="cyan")
        table.add_column("Result", style="green")
        
        args_str = json.dumps(arguments, indent=2)
        table.add_row(args_str, result)
        
        self.console.print(table)
        
    def display_history(self):
        """Display the full conversation history."""
        if not self.enable_observability:
            return
            
        history = self.get_history()
        self.console.print(Panel("Conversation History", border_style="bold"))
        
        for msg in history:
            self.display_message(msg["role"], msg["content"])

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
            
            if self.enable_observability:
                self.display_thinking(f"Calling tool: {function_name}")

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

            if self.enable_observability:
                self.display_tool_call(function_name, arguments, result_str)
                
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

    def run(
        self, user_input: str, attachments: Optional[str] = None, **kwargs
    ) -> Union[str, BaseModel]:
        self.message_counter += 1
        kwargs = {**self.default_kwargs, **kwargs}
        message_id = self.message_counter
        
        # Display system prompt on first run
        if self.message_counter == 1 and self.enable_observability:
            self.display_message("system", self.system_prompt)
        
        # Display user input
        if self.enable_observability:
            self.display_message("user", user_input, attachments)
        
        history = self.get_history()

        user_content = [{"type": "text", "text": user_input}]
        if attachments:
            user_content.append(
                {"type": "image_url", "image_url": {"url": attachments}}
            )

        messages = (
            [{"role": "system", "content": self.system_prompt}]
            + history
            + [{"role": "user", "content": user_content}]
        )

        self.messages_table.insert(
            [
                {
                    "message_id": message_id,
                    "role": "user",
                    "content": user_input,
                    "attachments": attachments,
                    "timestamp": datetime.now(),
                }
            ]
        )

        if self.enable_observability:
            self.display_thinking(f"Thinking... (using model: {self.model})")

        # Use beta endpoint for structured outputs if specified
        completion_method = (
            self.client.beta.chat.completions.parse
            if self.structured_output
            else self.client.chat.completions.create
        )

        completion_kwargs = {
            "model": self.model,
            "messages": messages,
            "tools": self.tool_definitions if self.tools else None,
            **kwargs,
        }
        if self.structured_output:
            completion_kwargs["response_format"] = self.structured_output

        completion = completion_method(**completion_kwargs)

        if (
            not hasattr(completion.choices[0].message, "tool_calls")
            or not completion.choices[0].message.tool_calls
        ):
            response = (
                completion.choices[0].message.parsed
                if self.structured_output
                else completion.choices[0].message.content
            )
        else:
            if self.enable_observability:
                self.display_thinking("Processing tool calls...")
                
            tool_results = self.process_tool_calls(completion, message_id)
            messages.append(completion.choices[0].message.to_dict())
            for result in tool_results:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": result["tool_call_id"],
                        "content": result["result"],
                    }
                )
                if self.enable_observability:
                    self.display_message("tool", result["result"])

            if self.enable_observability:
                self.display_thinking("Generating final response...")
                
            final_completion = completion_method(
                model=self.model,
                messages=messages,
                tools=self.tool_definitions if self.tools else None,
                response_format=self.structured_output
                if self.structured_output
                else None,
                **kwargs,
            )
            response = (
                final_completion.choices[0].message.parsed
                if self.structured_output
                else final_completion.choices[0].message.content
            )

        # Log assistant response
        self.message_counter += 1
        response_content = (
            json.dumps(response.dict()) if self.structured_output else response
        )
        
        # Display assistant response
        if self.enable_observability:
            self.display_message("assistant", response_content)
            
        self.messages_table.insert(
            [
                {
                    "message_id": self.message_counter,
                    "role": "assistant",
                    "content": response_content,
                    "timestamp": datetime.now(),
                }
            ]
        )

        self.chat_table.insert(
            [
                {
                    "system_prompt": self.system_prompt,
                    "user_input": user_input,
                    "response": response_content,
                }
            ]
        )

        return response
