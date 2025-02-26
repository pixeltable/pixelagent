import json
from datetime import datetime
from typing import Callable, Dict, List, Optional, Type, Union

from openai import OpenAI
from pydantic import BaseModel

from pixelagent.core import setup_pixeltable
from pixelagent.core.display import PixelAgentDisplay

class Agent:
    def __init__(
        self,
        name: str,
        system_prompt: str,
        model: str = "gpt-4o-mini",
        tools: List[Callable] = None,
        structured_output: Optional[Type[BaseModel]] = None,
        reset: bool = False,
        debug: bool = True,
        **default_kwargs,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.client = OpenAI()
        self.model = model
        self.tools = tools if tools else []
        self.structured_output = structured_output
        self.reset = reset
        self.default_kwargs = default_kwargs
        self.tool_definitions = [tool.tool_definition for tool in self.tools]
        self.available_tools = {tool.__name__: tool for tool in self.tools}
        self.messages_table, self.tool_calls_table = setup_pixeltable(
            name, reset
        )
        self.message_counter = 0
        
        # Observability settings
        self.debug = debug
        self.display = PixelAgentDisplay(debug)

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
            
            if self.debug:
                self.display.display_thinking(f"Calling tool: {function_name}")

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

            if self.debug:
                self.display.display_tool_call(function_name, arguments, result_str)
                
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
        if self.message_counter == 1 and self.debug:
            self.display.display_message("system", self.system_prompt)
        
        # Display user input
        if self.debug:
            self.display.display_message("user", user_input, attachments)
        
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


        if self.debug:
            self.display.display_thinking(f"Thinking... (using model: {self.model})")

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
            if self.debug:
                self.display.display_thinking("Processing tool calls...")
                
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
                if self.debug:
                    self.display.display_message("tool", result["result"])

            if self.debug:
                self.display.display_thinking("Generating final response...")
                
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
        if self.debug:
            self.display.display_message("assistant", response_content)
            
        self.messages_table.insert(
            [
                {
                    "message_id": self.message_counter,
                    "system_prompt": self.system_prompt,
                    "user_input": user_input,
                    "response": response_content,                    
                    "role": "assistant",
                    "content": response_content,
                    "timestamp": datetime.now(),
                }
            ]
        )


        return response
