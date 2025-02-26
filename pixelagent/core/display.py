from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
import json
from typing import Dict, Optional

class PixelAgentDisplay:
    def __init__(self, debug: bool = True):
        self.debug = debug
        self.console = Console() if debug else None
    
    def display_message(self, role: str, content: str, attachments: Optional[str] = None):
        """Display a message with rich formatting."""
        if not self.debug:
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
        if not self.debug:
            return
        self.console.print(f"[dim italic]{message}[/dim italic]")

    def display_tool_call(self, tool_name: str, arguments: Dict, result: str):
        """Display a tool call in a table format."""
        if not self.debug:
            return
            
        table = Table(title=f"Tool Call: {tool_name}")
        table.add_column("Arguments", style="cyan")
        table.add_column("Result", style="green")
        
        args_str = json.dumps(arguments, indent=2)
        table.add_row(args_str, result)
        
        self.console.print(table)
        
    def display_history(self, history):
        """Display the full conversation history."""
        if not self.debug:
            return
            
        self.console.print(Panel("Conversation History", border_style="bold"))
        
        for msg in history:
            self.display_message(msg["role"], msg["content"]) 