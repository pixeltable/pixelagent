import os
from typing import Dict, List, Any, Callable
import anthropic
from pixelagent.anthropic.utils import tool
from pixelagent.core import setup_pixeltable
from pixelagent.core.display import PixelAgentDisplay
import json
from datetime import datetime

class Agent:
    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: List[Callable] = None,
        reset: bool = False,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: str = None,
        debug: bool = True
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = {tool_func.name: tool_func for tool_func in (tools or [])}
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.messages_table, self.tool_calls_table = setup_pixeltable(
            name, tool_calls_table=True, reset=reset
        )
        self.message_counter = 0
        self.debug = debug
        self.display = PixelAgentDisplay(debug)

    def _format_tools(self) -> List[Dict]:
        return [tool_func.to_dict() for tool_func in self.tools.values()]

    def get_history(self) -> List[Dict]:
        result = self.messages_table.select(
            self.messages_table.role,
            self.messages_table.content
        ).order_by(self.messages_table.message_id).collect()
        return [{"role": row["role"], "content": row["content"]} for row in result]

    def process_tool_calls(self, response, message_id: int) -> List[Dict]:
        tool_uses = [content for content in response.content if content.type == "tool_use"]
        if not tool_uses:
            return []

        results = []
        for tool_use in tool_uses:
            tool_name, tool_input = tool_use.name, tool_use.input
            
            if self.debug:
                self.display.display_thinking(f"Calling tool: {tool_name}")

            try:
                result_str = (str(self.tools[tool_name](**tool_input)) 
                            if tool_name in self.tools 
                            else f"Error: Tool {tool_name} not found")
            except Exception as e:
                result_str = f"Error: {str(e)}"

            if self.debug:
                self.display.display_tool_call(tool_name, tool_input, result_str)

            self.tool_calls_table.insert([{
                "tool_call_id": tool_use.id,
                "message_id": message_id,
                "tool_name": tool_name,
                "arguments": tool_input,
                "result": result_str,
                "timestamp": datetime.now(),
            }])

            results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": result_str
            })
        return results

    def run(self, user_input: str) -> str:
        self.message_counter += 1
        message_id = self.message_counter

        if self.message_counter == 1 and self.debug:
            self.display.display_message("system", self.system_prompt)

        if self.debug:
            self.display.display_message("user", user_input)

        self.messages_table.insert([{
            "message_id": message_id,
            "system_prompt": self.system_prompt,
            "user_input": user_input,
            "response": None,
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now(),
        }])

        messages = self.get_history() + [{"role": "user", "content": user_input}]

        if self.debug:
            self.display.display_thinking(f"Thinking... (using model: {self.model})")

        response = self.client.messages.create(
            model=self.model,
            system=self.system_prompt,
            messages=messages,
            tools=self._format_tools(),
            max_tokens=1000
        )

        final_content = [content.text for content in response.content if content.type == "text"]
        tool_results = self.process_tool_calls(response, message_id)

        if tool_results:
            if self.debug:
                self.display.display_thinking("Processing tool calls...")
            
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            
            if self.debug:
                self.display.display_thinking("Generating final response...")
                
            final_response = self.client.messages.create(
                model=self.model,
                system=self.system_prompt,
                messages=messages,
                tools=self._format_tools(),
                max_tokens=1000
            )
            final_content = [content.text for content in final_response.content 
                            if content.type == "text"]
            response_content = "".join(final_content)  # Changed to just the text
        else:
            response_content = "".join(final_content)  # Changed to just the text

        self.message_counter += 1
        assistant_message_id = self.message_counter
        self.messages_table.insert([{
            "message_id": assistant_message_id,
            "system_prompt": self.system_prompt,
            "user_input": user_input,
            "response": response_content,
            "role": "assistant",
            "content": response_content,  # Store just the text
            "timestamp": datetime.now(),
        }])

        final_response = "".join(final_content)
        if self.debug:
            self.display.display_message("assistant", final_response)
        return final_response