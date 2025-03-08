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
        reset: bool = True,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: str = None,
        debug: bool = True
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = {tool_func.name: tool_func for tool_func in tools or []}
        self.model = model
        self.reset = reset
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.messages = [] if reset else None
        self.messages_table, self.tool_calls_table = setup_pixeltable(
            name, tool_calls_table=True, reset=reset
        )
        self.message_counter = 0
        self.debug = debug
        self.display = PixelAgentDisplay(debug)

    def _format_tools(self) -> List[Dict]:
        """Format tools for Anthropic API."""
        return [tool_func.to_dict() for tool_func in self.tools.values()]

    def _run_tool(self, tool_name: str, tool_input: Dict, message_id: int) -> Any:
        """Execute the specified tool with given input and log to Pixeltable."""
        if self.debug:
            self.display.display_thinking(f"Calling tool: {tool_name}")

        if tool_name not in self.tools:
            error_msg = f"Tool {tool_name} not found"
            if self.debug:
                self.display.display_tool_call(tool_name, tool_input, error_msg)
            return {"error": error_msg}

        try:
            result = self.tools[tool_name](**tool_input)
            result_str = str(result)
            
            # Log tool call to Pixeltable
            self.tool_calls_table.insert([{
                "tool_call_id": f"{message_id}_{tool_name}",
                "message_id": message_id,
                "tool_name": tool_name,
                "arguments": tool_input,
                "result": result_str,
                "timestamp": datetime.now(),
            }])
            
            if self.debug:
                self.display.display_tool_call(tool_name, tool_input, result_str)
            return result
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            if self.debug:
                self.display.display_tool_call(tool_name, tool_input, error_msg)
            return {"error": error_msg}

    def get_history(self) -> List[Dict]:
        """Fetch conversation history from Pixeltable."""
        result = self.messages_table.select(
            self.messages_table.role, self.messages_table.content
        ).collect()
        return [{"role": row["role"], "content": row["content"]} for row in result]

    def run(self, query: str) -> str:
        """Process a query and return the response."""
        self.message_counter += 1
        message_id = self.message_counter

        if self.reset or self.messages is None:
            self.messages = []

        # Display system prompt on first run
        if self.message_counter == 1 and self.debug:
            self.display.display_message("system", self.system_prompt)

        # Display user input
        if self.debug:
            self.display.display_message("user", query)

        # Log user query to Pixeltable
        self.messages_table.insert([{
            "message_id": message_id,
            "system_prompt": self.system_prompt,
            "user_input": query,
            "response": None,
            "role": "user",
            "content": query,
            "timestamp": datetime.now(),
        }])

        self.messages.append({"role": "user", "content": query})

        if self.debug:
            self.display.display_thinking(f"Thinking... (using model: {self.model})")

        while True:
            response = self.client.messages.create(
                model=self.model,
                system=self.system_prompt,
                messages=self.messages,
                tools=self._format_tools(),
                max_tokens=1000
            )

            final_content = []
            tool_uses = []

            # Process all content blocks in the response
            for content in response.content:
                if content.type == "text":
                    final_content.append({"type": "text", "text": content.text})
                elif content.type == "tool_use":
                    tool_uses.append(content)

            # Log and add assistant's response to message history
            response_content = json.dumps([c.to_dict() for c in response.content])
            self.messages.append({"role": "assistant", "content": response.content})
            
            self.message_counter += 1
            assistant_message_id = self.message_counter
            self.messages_table.insert([{
                "message_id": assistant_message_id,
                "system_prompt": self.system_prompt,
                "user_input": query,
                "response": response_content,
                "role": "assistant",
                "content": response_content,
                "timestamp": datetime.now(),
            }])

            # Handle tool uses
            if tool_uses:
                if self.debug:
                    self.display.display_thinking("Processing tool calls...")
                
                tool_results = []
                for tool_use in tool_uses:
                    tool_name = tool_use.name
                    tool_input = tool_use.input
                    tool_result = self._run_tool(tool_name, tool_input, assistant_message_id)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": str(tool_result)
                    })

                # Submit all tool results
                self.messages.append({
                    "role": "user",
                    "content": tool_results
                })
                continue

            # Display and return final response
            final_response = "".join(item["text"] for item in final_content if "text" in item)
            if self.debug:
                self.display.display_message("assistant", final_response)
            return final_response