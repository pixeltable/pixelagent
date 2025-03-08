from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.text import Text
import json
from typing import Dict, Optional

class PixelAgentDisplay:
    def __init__(self, debug: bool = True):
        self.debug = debug
        self.console = Console() if debug else None
    
    def display_message(self, role: str, content: str, attachments: Optional[str] = None):
        """Display a message with appropriate formatting based on its role."""
        if not self.debug:
            return
            
        if role == "system":
            self.console.print(Panel(
                content, 
                title="[yellow bold]System[/]", 
                border_style="yellow", 
                subtitle="Configuration"
            ))
        elif role == "user":
            self.console.print(Panel(
                Text(content, style="green"), 
                title="[green bold]User Input[/]", 
                border_style="green", 
                subtitle="Query"
            ))
            if attachments:
                self.console.print(Panel(
                    f"📎 Attachment: {attachments}", 
                    border_style="green", 
                    title="[green]Additional Context[/]"
                ))
        elif role == "assistant":
            try:
                md = Markdown(content)
                self.console.print(Panel(
                    md, 
                    title="[blue bold]PixelAgent[/]", 
                    border_style="blue", 
                    subtitle="Response"
                ))
            except:
                self.console.print(Panel(
                    Text(content, style="blue"), 
                    title="[blue bold]PixelAgent[/]", 
                    border_style="blue", 
                    subtitle="Response"
                ))
        elif role == "tool":
            self.console.print(Panel(
                content, 
                title="[purple bold]Tool Output[/]", 
                border_style="purple", 
                subtitle="Function Result"
            ))

    def display_thinking(self, message: str):
        """Display the agent's thinking process."""
        if not self.debug:
            return
        self.console.print(f"[dim italic cyan]⏳ Processing: {message}...[/]")

    def display_tool_call(self, tool_name: str, arguments: Dict, result: str):
        """Display tool call details with inputs and outputs."""
        if not self.debug:
            return
            
        table = Table(title=f"[purple bold]Tool Call: {tool_name}[/]", border_style="purple")
        table.add_column("Parameters", style="cyan", justify="left")
        table.add_column("Result", style="green", justify="left")
        
        args_str = json.dumps(arguments, indent=2)
        table.add_row(Text(args_str, style="cyan"), Text(result, style="green"))
        
        self.console.print(Panel(
            table, 
            border_style="purple", 
            subtitle="[purple]Function Execution[/]"
        ))
        
    def display_history(self, history):
        """Display the conversation history."""
        if not self.debug:
            return
            
        self.console.print(Panel(
            Text("Conversation History", style="bold white"), 
            border_style="bold blue", 
            title="[blue bold]Session Log[/]",
            subtitle="Complete Interaction"
        ))
        
        for msg in history:
            self.display_message(msg["role"], msg["content"])