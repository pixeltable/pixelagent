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
        """Drop a message with AgentX swagger—locked and loaded!"""
        if not self.debug:
            return
            
        if role == "system":
            self.console.print(Panel(
                content, 
                title="[yellow bold]AgentX Core[/]", 
                border_style="yellow bold", 
                subtitle="⚙️  System Boost ⚙️"
            ))
        elif role == "user":
            self.console.print(Panel(
                Text(content, style="green"), 
                title="[green bold]Your Command[/]", 
                border_style="green bold", 
                subtitle="🚀 Launched by You 🚀"
            ))
            if attachments:
                self.console.print(Panel(
                    f"📎 Attachment: {attachments}", 
                    border_style="green bold", 
                    title="[green italic]Extra Input[/]"
                ))
        elif role == "assistant":
            try:
                md = Markdown(content)
                self.console.print(Panel(
                    md, 
                    title="[blue bold]AgentX Output[/]", 
                    border_style="blue bold", 
                    subtitle="✨ Powered Up ✨"
                ))
            except:
                self.console.print(Panel(
                    Text(content, style="blue"), 
                    title="[blue bold]AgentX Output[/]", 
                    border_style="blue bold", 
                    subtitle="✨ Powered Up ✨"
                ))
        elif role == "tool":
            self.console.print(Panel(
                content, 
                title="[purple bold]Power Surge[/]", 
                border_style="purple bold", 
                subtitle="💥 AgentX Flex 💥"
            ))

    def display_thinking(self, message: str):
        """Flash the AgentX grind—processing in style!"""
        if not self.debug:
            return
        self.console.print(f"[dim italic cyan]⏳ AgentX Grinding: {message}...[/]")

    def display_tool_call(self, tool_name: str, arguments: Dict, result: str):
        """Flex the AgentX power stats—sleek and sharp!"""
        if not self.debug:
            return
            
        table = Table(title=f"[purple bold]Power Surge: {tool_name} ⚡[/]", border_style="purple")
        table.add_column("🔧 Inputs", style="cyan bold", justify="center")
        table.add_column("🎯 Output", style="green bold", justify="center")
        
        args_str = json.dumps(arguments, indent=2)
        table.add_row(Text(args_str, style="cyan"), Text(result, style="green"))
        
        self.console.print(Panel(
            table, 
            border_style="purple bold", 
            subtitle="[purple italic]🔥 Power Unleashed 🔥[/]"
        ))
        
    def display_history(self, history):
        """Replay the AgentX saga—full run stats!"""
        if not self.debug:
            return
            
        self.console.print(Panel(
            Text("AgentX Run Log 📜", style="bold white"), 
            border_style="bold red", 
            title="[red bold]X-Factor Replay[/]",
            subtitle="📈 Full History 📈"
        ))
        
        for msg in history:
            self.display_message(msg["role"], msg["content"])